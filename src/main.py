import argparse
import csv
import json
import logging
import os
import shutil
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
    pass

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

def _resolve_hf_model_name(source_lang: str, target_lang: str) -> str:
    try:
        from huggingface_hub import model_info # type: ignore
    except ImportError:
        pass
    
    candidates = [
        f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}",
        f"Helsinki-NLP/opus-mt-tc-big-{source_lang}-{target_lang}",
    ]

    for name in candidates:
        try:
            model_info(name)
            logger.info(f"Resolved HuggingFace model: {name}")
            return name
        except Exception:
            continue

    return candidates[0]

def convert_model(source_lang: str, target_lang: str, quantization: str = "int8") -> Path:
    try:
        import ctranslate2 # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # type: ignore
    except ImportError:
        raise ImportError("Please install dependencies: pip install ctranslate2 transformers huggingface_hub torch sentencepiece")

    model_name = _resolve_hf_model_name(source_lang, target_lang)
    output_dir = _model_dir(source_lang, target_lang)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model {model_name} from HuggingFace...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to download model '{model_name}': {e}")

        logger.info(f"Converting model {model_name} -> {output_dir} (quantization={quantization})")
        converter = ctranslate2.converters.TransformersConverter(tmp_dir)
        converter.convert(str(output_dir), quantization=quantization, force=True)

    tokenizer.save_pretrained(str(output_dir))

    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model_name": model_name,
        "quantization": quantization,
        "converted_at": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")

    logger.info(f"Model converted successfully: {output_dir}")
    return output_dir

_model_cache = {}

def get_translator(source_lang: str, target_lang: str, device: str = "cpu"):
    cache_key = f"{source_lang}-{target_lang}-{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not model_exists(source_lang, target_lang):
        logger.info(f"Model for {source_lang}->{target_lang} not found locally. Preparing to download and convert...")
        convert_model(source_lang, target_lang)

    try:
        import ctranslate2 # type: ignore
        from transformers import AutoTokenizer # type: ignore
    except ImportError:
        pass

    model_path = str(_model_dir(source_lang, target_lang))
    model_name = get_stored_model_name(source_lang, target_lang) or f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    logger.info(f"Loading CTranslate2 model: {model_name} on {device}")
    translator = ctranslate2.Translator(model_path, device=device, inter_threads=4)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        logger.info(f"Tokenizer not found locally, downloading: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    _model_cache[cache_key] = (translator, tokenizer)
    return translator, tokenizer

def translate(texts: list[str] | str, target_lang: str, source_lang: str = "en", device: str = "cpu") -> list[str] | str:
    """Translate text(s) from source_lang to target_lang."""
    is_single = isinstance(texts, str)
    
    text_list: list[str] = [texts] if isinstance(texts, str) else list(texts) # type: ignore
        
    if not text_list or not any(text_list):
        return "" if is_single else []

    # Filter out empty or whitespace-only texts
    non_empty_indices = [i for i, t in enumerate(text_list) if t and t.strip()]
    if not non_empty_indices:
        return [""] * len(text_list) if not is_single else ""

    texts_to_translate: list[str] = [text_list[i] for i in non_empty_indices]
    translator, tokenizer = get_translator(source_lang, target_lang, device=device)

    # Ensure tokenization explicitly treats it as a single source sequence per input
    tokenized = []
    for t in texts_to_translate:
        # Some models require source language prefix, typically only multilingual ones
        # For standard MarianMT, this usually isn't necessary, but passing direct lists is safer
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(t, truncation=True, max_length=512))
        tokenized.append(tokens)

    max_input_len = max(len(t) for t in tokenized) if tokenized else 0
    # Keep max decoding proportional but bound it safely
    max_decoding_length = min(max(int(max_input_len * 2), 64), 256)

    # Use default params (beam_size=4 is usually default for ctranslate2)
    results = translator.translate_batch(
        tokenized,
        beam_size=4,
        max_decoding_length=max_decoding_length,
    )

    translated_non_empty = []
    for result in results:
        tokens = result.hypotheses[0]
        text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
        translated_non_empty.append(text)

    # Reconstruct original list order with empty strings substituted back
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
    text_parser.add_argument("--target", required=True, choices=["fr", "it", "pt"], help="Target language (fr, it, pt)")
    text_parser.add_argument("--source", default="en", help="Source language (default: en)")
    text_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Computation device (cpu, cuda, auto)")

    # Subcommand: csv
    csv_parser = subparsers.add_parser("csv", help="Translate a column in a CSV file")
    csv_parser.add_argument("input", help="Input CSV file path")
    csv_parser.add_argument("output", help="Output CSV file path")
    csv_parser.add_argument("--column", required=True, help="Name of the column to translate")
    csv_parser.add_argument("--target", required=True, choices=["fr", "it", "pt"], help="Target language (fr, it, pt)")
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
