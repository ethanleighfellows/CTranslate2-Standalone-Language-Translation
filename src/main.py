from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import site
import re
import threading

# On Windows, try to auto-discover nvidia-cublas-cu12 and nvidia-cudnn-cu12 DLLs
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    try:
        paths = site.getsitepackages() + [site.getusersitepackages()]
        for p in paths:
            cublas_bin = os.path.join(p, 'nvidia', 'cublas', 'bin')
            if os.path.exists(cublas_bin):
                os.add_dll_directory(cublas_bin) # type: ignore
            cudnn_bin = os.path.join(p, 'nvidia', 'cudnn', 'bin')
            if os.path.exists(cudnn_bin):
                os.add_dll_directory(cudnn_bin) # type: ignore
    except Exception:
        pass
import tempfile
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Increase CSV max field size to prevent errors on giant prompts
csv.field_size_limit(1048576)

# Create a local models directory for caching HuggingFace downloads
MODELS_DIR = Path(__file__).parent.parent / "translation_models"

class ModelNotFoundError(Exception):
    """Raised when a requested CTranslate2 model is not installed."""

def _model_dir(source_lang: str, target_lang: str) -> Path:
    return MODELS_DIR / f"{source_lang}-{target_lang}"

def get_stored_model_name(source_lang: str, target_lang: str) -> str | None:
    meta_path = _model_dir(source_lang, target_lang) / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("model_name")
        except Exception:
            pass
    return None

def model_exists(source_lang: str, target_lang: str) -> bool:
    d = _model_dir(source_lang, target_lang)
    return d.is_dir() and (d / "model.bin").exists()


def _hf_model_info(model_name: str):
    """Thin wrapper around ``huggingface_hub.model_info`` for testability."""
    from huggingface_hub import model_info  # type: ignore
    return model_info(model_name)


_HF_LANG_ALIASES: dict[str, str] = {
    "ja": "jap",
}
"""Helsinki-NLP uses non-standard language codes for some models.
Map ISO 639-1 codes to the codes Helsinki-NLP actually uses on HuggingFace."""


def _resolve_hf_model_name(source_lang: str, target_lang: str) -> str:
    """Resolve the HuggingFace model identifier for a language pair.

    Tries all combinations of standard ISO codes and Helsinki-NLP aliases
    across both opus-mt- and opus-mt-tc-big- prefixes. Returns the first
    candidate that exists on HuggingFace.
    """
    src_codes = {source_lang, _HF_LANG_ALIASES.get(source_lang, source_lang)}
    tgt_codes = {target_lang, _HF_LANG_ALIASES.get(target_lang, target_lang)}

    candidates = []
    for s in src_codes:
        for t in tgt_codes:
            candidates.append(f"Helsinki-NLP/opus-mt-{s}-{t}")
            candidates.append(f"Helsinki-NLP/opus-mt-tc-big-{s}-{t}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    for name in candidates:
        try:
            _hf_model_info(name)
            logger.info("Resolved HuggingFace model: %s", name)
            return name
        except Exception:
            logger.debug("Model candidate not found: %s", name)
            continue

    tried = ", ".join(candidates)
    raise RuntimeError(
        f"No Helsinki-NLP model found for {source_lang}→{target_lang}. "
        f"Tried: {tried}"
    )

def convert_model(
    source_lang: str,
    target_lang: str,
    quantization: str = "int8",
    progress_callback: callable | None = None,  # type: ignore[type-arg]
) -> Path:
    """Download a Helsinki-NLP MarianMT model and convert it to CTranslate2 format.

    *progress_callback*, if provided, is called with ``(stage, message)``
    tuples at each meaningful step so the caller can relay progress to a UI.
    """
    def _log(stage: str, message: str) -> None:
        logger.info(message)
        if progress_callback:
            progress_callback(stage, message)

    _log("init", "Checking ctranslate2 dependency...")
    try:
        import ctranslate2  # type: ignore
    except ImportError:
        raise ImportError(
            "ctranslate2 is not installed. "
            "Install with: pip install ctranslate2 transformers sentencepiece"
        )

    _log("init", "Checking transformers dependency...")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError:
        raise ImportError(
            "transformers is not installed. "
            "Install with: pip install ctranslate2 transformers sentencepiece"
        )

    _log("resolve", f"Resolving HuggingFace model for {source_lang}→{target_lang}...")
    model_name = _resolve_hf_model_name(source_lang, target_lang)
    _log("resolve", f"Using model: {model_name}")

    output_dir = _model_dir(source_lang, target_lang)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        _log("download", f"Downloading tokenizer for {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download tokenizer for '{model_name}': {e}"
            ) from e

        _log("download", f"Downloading model weights for {model_name} (this may take a few minutes)...")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model '{model_name}' from HuggingFace: {e}"
            ) from e

        _log("download", "Saving model to temporary directory...")
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        _log("convert", f"Converting to CTranslate2 format (quantization={quantization})...")
        converter = ctranslate2.converters.TransformersConverter(tmp_dir)
        converter.convert(str(output_dir), quantization=quantization, force=True)

    _log("finalize", "Saving tokenizer for offline inference...")
    tokenizer.save_pretrained(str(output_dir))

    _log("finalize", "Writing model metadata...")
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model_name": model_name,
        "quantization": quantization,
        "converted_at": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")

    _log("complete", f"Model {source_lang}→{target_lang} ready at {output_dir}")
    return output_dir

_model_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()

# MarianMT hard limit on input tokens (including </s>).
_MODEL_TOKEN_LIMIT = 512

# Chunk budget: MarianMT was trained on sentence pairs, not paragraphs.
# When given long inputs it compresses/summarises rather than translating
# faithfully. Keeping chunks to ~3-4 sentences (<=150 tokens) keeps the model
# in its faithful translation zone while still well below the 512-token
# hard encoder limit.
_MAX_CHUNK_TOKENS = 150

# Maximum tokens the decoder is allowed to generate per chunk.
# Must be well above _MAX_CHUNK_TOKENS to accommodate verbose target languages
# (e.g. French is 15-25% more verbose than English).
_MAX_DECODING_LENGTH = 1024

# Sentence-ending pattern for primary chunking
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

# Clause-level pattern for secondary splitting of oversized sentences
_CLAUSE_RE = re.compile(r'(?<=[,;:—])\s+')


def get_translator(source_lang: str, target_lang: str, device: str = "cpu"):
    """Lazy-load and cache a CTranslate2 Translator + HuggingFace tokenizer."""
    cache_key = f"{source_lang}-{target_lang}-{device}"

    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

    if not model_exists(source_lang, target_lang):
        logger.info("Model for %s→%s not found locally. Downloading and converting...", source_lang, target_lang)
        convert_model(source_lang, target_lang)

    try:
        import ctranslate2  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        raise ImportError(
            "ctranslate2 and transformers are not installed. "
            "Install with: pip install ctranslate2 transformers sentencepiece"
        )

    model_path = str(_model_dir(source_lang, target_lang))
    model_name = (
        get_stored_model_name(source_lang, target_lang)
        or f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    )

    logger.info("Loading CTranslate2 model: %s on %s", model_name, device)
    translator = ctranslate2.Translator(model_path, device=device, inter_threads=4)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        logger.info("Tokenizer not found locally, downloading: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    with _cache_lock:
        _model_cache[cache_key] = (translator, tokenizer)

    return translator, tokenizer


def evict_model(source_lang: str, target_lang: str) -> None:
    """Remove a model from the in-memory cache (called after model deletion)."""
    prefix = f"{source_lang}-{target_lang}"
    with _cache_lock:
        keys_to_remove = [k for k in _model_cache if k.startswith(prefix)]
        for k in keys_to_remove:
            _model_cache.pop(k, None)


# ── Tokenization helpers ──────────────────────────────────────────


def _encode_for_ct2(text: str, tokenizer) -> list[str]:
    """Tokenize text into token strings including the EOS marker that
    MarianMT/CTranslate2 requires at the end of source sequences."""
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(text))


def _decode_from_ct2(token_strings: list[str], tokenizer) -> str:
    """Convert CTranslate2 output token strings back to a text string."""
    ids = tokenizer.convert_tokens_to_ids(token_strings)
    return tokenizer.decode(ids, skip_special_tokens=True)


def _token_len(text: str, tokenizer) -> int:
    """Return the number of tokens that ``_encode_for_ct2`` would produce.

    Uses ``tokenizer.encode`` so the count includes the ``</s>`` marker,
    matching what actually gets sent to CTranslate2.
    """
    return len(tokenizer.encode(text))


# ── Chunking helpers ──────────────────────────────────────────────


def _split_at_word_boundary(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Last-resort splitter: break *text* into segments that each encode
    to at most *max_tokens* tokens. Splits on whitespace boundaries so
    words are never cut in half.
    """
    words = text.split()
    if not words:
        return []

    segments: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(current_words + [word])
        if _token_len(candidate, tokenizer) > max_tokens and current_words:
            segments.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        segments.append(" ".join(current_words))

    return segments


def _split_segment(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split a single segment (typically one sentence) so every piece
    fits within *max_tokens*.

    Strategy:
      1. If it already fits, return as-is.
      2. Try splitting at clause boundaries (``, ; : —``).
      3. Fall back to word-boundary splitting.
    """
    if _token_len(text, tokenizer) <= max_tokens:
        return [text]

    clauses = _CLAUSE_RE.split(text)
    if len(clauses) > 1:
        merged: list[str] = []
        current_parts: list[str] = []
        current_count = 0

        for clause in clauses:
            clause_len = _token_len(clause, tokenizer)
            if clause_len == 0:
                continue
            if current_count + clause_len > max_tokens and current_parts:
                merged.append(" ".join(current_parts))
                current_parts = []
                current_count = 0
            current_parts.append(clause)
            current_count += clause_len

        if current_parts:
            merged.append(" ".join(current_parts))

        result: list[str] = []
        for seg in merged:
            if _token_len(seg, tokenizer) > max_tokens:
                result.extend(_split_at_word_boundary(seg, tokenizer, max_tokens))
            else:
                result.append(seg)
        return result

    return _split_at_word_boundary(text, tokenizer, max_tokens)


def _build_chunks(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split *text* into translation-ready chunks, each guaranteed to
    encode to at most *max_tokens* tokens.

    Splitting order:
      1. Sentence boundaries (``. ! ?``)
      2. Clause boundaries (``, ; : —``) for oversized sentences
      3. Word boundaries as a last resort
    """
    sentences = _SENTENCE_END_RE.split(text)
    if not sentences:
        return []

    segments: list[str] = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        segments.extend(_split_segment(sentence, tokenizer, max_tokens))

    if not segments:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_count = 0

    for seg in segments:
        seg_len = _token_len(seg, tokenizer)
        if current_count + seg_len > max_tokens and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_count = 0
        current_parts.append(seg)
        current_count += seg_len

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _chunk_and_translate(text: str, translator, tokenizer) -> str:
    """Split *text* into chunks that fit within the model's token limit,
    translate each chunk independently, and rejoin the results.
    """
    chunks = _build_chunks(text, tokenizer, _MAX_CHUNK_TOKENS)
    if not chunks:
        return ""

    translated_parts: list[str] = []
    for chunk_text in chunks:
        tokenized = _encode_for_ct2(chunk_text, tokenizer)
        max_decoding_length = min(max(len(tokenized) * 3, 128), _MAX_DECODING_LENGTH)

        result = translator.translate_batch(
            [tokenized],
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_decoding_length=max_decoding_length,
        )

        translated_parts.append(_decode_from_ct2(result[0].hypotheses[0], tokenizer))

    return " ".join(translated_parts)

def extract_codespans(text: str) -> tuple[str, dict[int, str]]:
    spans = {}
    counter = [0]

    def replacer(match):
        span_text = match.group(0)
        c = counter[0]
        placeholder = f"[XXX_{c}_XXX]"
        spans[c] = span_text
        counter[0] += 1
        return placeholder

    # First extract block-level code to avoid matching single backticks inside them
    # re.DOTALL is important for blocks spanning multiple lines
    text = re.sub(r'```.*?```', replacer, text, flags=re.DOTALL)
    
    # Then extract inline code
    text = re.sub(r'`[^`]+`', replacer, text)
    
    return text, spans

def restore_codespans(text: str, spans: dict[int, str]) -> str:
    if not spans:
        return text
    
    # We use a permissive regex because translation models might insert spaces
    # inside our placeholders: [ XXX _ 0 _ XXX ]
    def replacer(match):
        idx_str = match.group(1)
        try:
            idx = int(idx_str)
            if idx in spans:
                return spans[idx]
        except ValueError:
            pass
        return match.group(0) # Return original matched string if not found

    return re.sub(r'\[\s*XXX\s*_\s*(\d+)\s*_\s*XXX\s*\]', replacer, text)

def translate(texts: list[str] | str, target_lang: str, source_lang: str = "en", device: str = "cpu") -> list[str] | str:
    """Translate text(s) from source_lang to target_lang.

    Texts that encode to more than ``_MAX_CHUNK_TOKENS`` tokens are
    automatically split, translated in chunks, and reassembled.
    """
    is_single = isinstance(texts, str)

    text_list: list[str] = [texts] if isinstance(texts, str) else list(texts)  # type: ignore

    if not text_list or not any(text_list):
        return "" if is_single else []

    non_empty_indices = [i for i, t in enumerate(text_list) if t and t.strip()]
    if not non_empty_indices:
        return [""] * len(text_list) if not is_single else ""

    texts_to_translate: list[str] = [text_list[i] for i in non_empty_indices]

    extracted_texts = []
    spans_list = []
    for text in texts_to_translate:
        ext_text, spans = extract_codespans(text)
        extracted_texts.append(ext_text)
        spans_list.append(spans)

    translator, tokenizer = get_translator(source_lang, target_lang, device=device)

    results_non_empty: list[str | None] = [None] * len(extracted_texts)
    short_indices: list[int] = []
    short_texts: list[str] = []

    for i, text in enumerate(extracted_texts):
        if _token_len(text, tokenizer) <= _MAX_CHUNK_TOKENS:
            short_indices.append(i)
            short_texts.append(text)
        else:
            results_non_empty[i] = _chunk_and_translate(text, translator, tokenizer)

    if short_texts:
        tokenized = [_encode_for_ct2(t, tokenizer) for t in short_texts]

        max_input_len = max(len(t) for t in tokenized) if tokenized else 0
        max_decoding_length = min(max(max_input_len * 3, 128), _MAX_DECODING_LENGTH)

        batch_results = translator.translate_batch(
            tokenized,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_decoding_length=max_decoding_length,
        )

        for idx, result in zip(short_indices, batch_results):
            results_non_empty[idx] = _decode_from_ct2(result.hypotheses[0], tokenizer)

    translated_non_empty = [r if r is not None else "" for r in results_non_empty]

    for i in range(len(translated_non_empty)):
        translated_non_empty[i] = restore_codespans(translated_non_empty[i], spans_list[i])

    translated = [""] * len(text_list)
    for i, orig_idx in enumerate(non_empty_indices):
        translated[orig_idx] = translated_non_empty[i]

    if is_single:
        return translated[0]
    return translated

def translate_csv(input_path: str, output_path: str, column: str, target_lang: str, source_lang: str = "en", batch_size: int = 32, device: str = "cpu"):
    """Translate a specific column in a CSV file and write to a new file."""
    try:
        from tqdm import tqdm # type: ignore
    except ImportError:
        def tqdm_fallback(iterable, *args, **kwargs):
            return iterable
        tqdm = tqdm_fallback  # type: ignore
        logger.warning("tqdm not installed. Progress bar will not be shown. (pip install tqdm)")

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Reading CSV: {input_file}")
    
    with open(input_file, mode="r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        
        if fieldnames is None or column not in fieldnames:
            logger.error(f"Column '{column}' not found in CSV. Available columns: {fieldnames}")
            return
            
        fieldnames_list: list[str] = list(fieldnames) # type: ignore
        rows: list[dict] = [row for row in reader] # type: ignore

    if not rows:
        logger.error("CSV file is empty.")
        return

    new_column = f"{column}_{target_lang}"
    if new_column not in fieldnames_list:
        fieldnames_list.append(new_column)

    logger.info(f"Translating {len(rows)} rows from {source_lang} to {target_lang}...")
    
    # Process in batches
    for i in tqdm(range(0, len(rows), batch_size), desc="Translating"): # type: ignore
        batch_rows: list[dict] = rows[i:i + batch_size] # type: ignore
        texts_to_translate = [r[column] for r in batch_rows]
        
        # Translate the batch
        try:
            logger.info(f"Translating batch of {len(texts_to_translate)} items starting at index {i}...")
            translations = translate(texts_to_translate, target_lang=target_lang, source_lang=source_lang, device=device)
            logger.info(f"Batch {i} translation complete.")
            
            # Ensure it's a list even if batch_size was 1
            if not isinstance(translations, list):
                translations = [translations]
                
            for j, translation in enumerate(translations):
                batch_rows[j][new_column] = translation
        except Exception as e:
            logger.error(f"Failed to translate batch starting at row {i}: {e}")
            # Fill with empty string safely
            for row in batch_rows:
                row[new_column] = ""

    logger.info(f"Writing results to: {output_file}")
    with open(output_file, mode="w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames_list)
        writer.writeheader()
        writer.writerows(rows) # type: ignore
        
    logger.info("Translation complete!")

def main():
    parser = argparse.ArgumentParser(description="Standalone CTranslate2 Translator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: text
    text_parser = subparsers.add_parser("text", help="Translate a single text string")
    text_parser.add_argument("text", help="Text to translate")
    text_parser.add_argument("--target", required=True, help="Target language code (e.g. fr, it, pt, ja, ko, de, es)")
    text_parser.add_argument("--source", default="en", help="Source language (default: en)")
    text_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Computation device (cpu, cuda, auto)")

    # Subcommand: csv
    csv_parser = subparsers.add_parser("csv", help="Translate a column in a CSV file")
    csv_parser.add_argument("input", help="Input CSV file path")
    csv_parser.add_argument("output", help="Output CSV file path")
    csv_parser.add_argument("--column", required=True, help="Name of the column to translate")
    csv_parser.add_argument("--target", required=True, help="Target language code (e.g. fr, it, pt, ja, ko, de, es)")
    csv_parser.add_argument("--source", default="en", help="Source language (default: en)")
    csv_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for translation (default: 32)")
    csv_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Computation device (cpu, cuda, auto)")

    args = parser.parse_args()
    
    if args.command == "text":
        result = translate(args.text, target_lang=args.target, source_lang=args.source, device=args.device)
        print("\n--- Translation Result ---")
        print(f"Source ({args.source}): {args.text}")
        print(f"Target ({args.target}): {result}")
        print("--------------------------\n")
    elif args.command == "csv":
        translate_csv(
            input_path=args.input,
            output_path=args.output,
            column=args.column,
            target_lang=args.target,
            source_lang=args.source,
            batch_size=args.batch_size,
            device=args.device
        )

if __name__ == "__main__":
    main()
