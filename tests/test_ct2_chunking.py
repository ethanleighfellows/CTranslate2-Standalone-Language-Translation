"""Tests for CTranslate2 translation chunking (long text support).

Verifies that texts exceeding the 512-token MarianMT input limit are
correctly split at sentence -> clause -> word boundaries.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import (
    _build_chunks,
    _chunk_and_translate,
    _split_at_word_boundary,
    _split_segment,
    _token_len,
    _MAX_CHUNK_TOKENS,
    _SENTENCE_END_RE,
    translate,
)


def _make_tokenizer(tokens_per_word: int = 1):
    """Create a mock tokenizer where each whitespace-separated word maps
    to *tokens_per_word* tokens.  ``encode()`` appends one extra id for
    the ``</s>`` marker, matching real MarianMT behavior.
    """
    mock = MagicMock()

    def _tokenize(t):
        words = t.split()
        return ["_tok"] * (len(words) * tokens_per_word) if words else []

    def _encode(t):
        words = t.split()
        n = len(words) * tokens_per_word
        return list(range(n)) + [0]  # +1 for </s>

    mock.tokenize.side_effect = _tokenize
    mock.encode.side_effect = _encode
    mock.convert_ids_to_tokens.side_effect = lambda ids: [f"t{i}" for i in ids]
    mock.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
    mock.decode.side_effect = lambda ids, **kw: f"translated({len(ids)} tokens)"
    return mock


def _make_translator():
    """Create a mock CTranslate2 Translator."""
    mock = MagicMock()

    def _translate_batch(tokenized, **kwargs):
        results = []
        for tokens in tokenized:
            r = MagicMock()
            r.hypotheses = [["out"]]
            results.append(r)
        return results

    mock.translate_batch.side_effect = _translate_batch
    return mock


# -- _token_len -------------------------------------------------------


class TestTokenLen:
    def test_includes_eos(self):
        tok = _make_tokenizer()
        # "hello world" -> 2 words -> 2 tokens + 1 </s> = 3
        assert _token_len("hello world", tok) == 3

    def test_empty(self):
        tok = _make_tokenizer()
        assert _token_len("", tok) == 1  # just </s>


# -- _split_at_word_boundary -------------------------------------------


class TestSplitAtWordBoundary:
    def test_short_text_single_segment(self):
        tok = _make_tokenizer()
        result = _split_at_word_boundary("a b c", tok, max_tokens=10)
        assert result == ["a b c"]

    def test_splits_when_needed(self):
        tok = _make_tokenizer()
        text = " ".join(f"w{i}" for i in range(10))
        result = _split_at_word_boundary(text, tok, max_tokens=6)
        assert len(result) >= 2
        for seg in result:
            assert _token_len(seg, tok) <= 6

    def test_empty_text(self):
        tok = _make_tokenizer()
        assert _split_at_word_boundary("", tok, max_tokens=10) == []

    def test_single_word_over_limit(self):
        """A single word that exceeds max_tokens still appears in output."""
        tok = _make_tokenizer(tokens_per_word=100)
        result = _split_at_word_boundary("superlongword", tok, max_tokens=10)
        assert result == ["superlongword"]


# -- _split_segment ----------------------------------------------------


class TestSplitSegment:
    def test_fits_returns_unchanged(self):
        tok = _make_tokenizer()
        assert _split_segment("short text", tok, max_tokens=50) == ["short text"]

    def test_clause_level_splitting(self):
        tok = _make_tokenizer()
        text = "first clause, second clause, third clause, fourth clause"
        result = _split_segment(text, tok, max_tokens=5)
        assert len(result) >= 2
        for seg in result:
            assert _token_len(seg, tok) <= 5

    def test_falls_back_to_word_level(self):
        """Text with no clause punctuation is split at word boundaries."""
        tok = _make_tokenizer()
        text = " ".join(f"word{i}" for i in range(20))
        result = _split_segment(text, tok, max_tokens=8)
        assert len(result) >= 3
        for seg in result:
            assert _token_len(seg, tok) <= 8


# -- _build_chunks -----------------------------------------------------


class TestBuildChunks:
    def test_short_text_single_chunk(self):
        tok = _make_tokenizer()
        result = _build_chunks("Hello world.", tok, max_tokens=50)
        assert result == ["Hello world."]

    def test_sentence_splitting(self):
        tok = _make_tokenizer()
        text = "First sentence. Second sentence. Third sentence."
        result = _build_chunks(text, tok, max_tokens=5)
        assert len(result) >= 2

    def test_oversized_sentence_gets_split(self):
        """A single sentence with no periods must still be split."""
        tok = _make_tokenizer()
        text = " ".join(f"w{i}" for i in range(100))
        result = _build_chunks(text, tok, max_tokens=20)
        assert len(result) >= 5
        for chunk in result:
            assert _token_len(chunk, tok) <= 20

    def test_groups_small_sentences(self):
        """Multiple tiny sentences should be grouped into one chunk."""
        tok = _make_tokenizer()
        text = "A. B. C. D. E."
        result = _build_chunks(text, tok, max_tokens=50)
        assert len(result) == 1

    def test_empty_text(self):
        tok = _make_tokenizer()
        assert _build_chunks("", tok, max_tokens=50) == []


# -- _SENTENCE_END_RE --------------------------------------------------


class TestSentenceEndRegex:
    def test_splits_on_period(self):
        parts = _SENTENCE_END_RE.split("Hello world. This is a test.")
        assert parts == ["Hello world.", "This is a test."]

    def test_splits_on_question_mark(self):
        parts = _SENTENCE_END_RE.split("Is this working? Yes it is.")
        assert len(parts) == 2

    def test_splits_on_exclamation(self):
        parts = _SENTENCE_END_RE.split("Wow! That is amazing.")
        assert len(parts) == 2

    def test_no_split_without_whitespace(self):
        parts = _SENTENCE_END_RE.split("e.g.this")
        assert len(parts) == 1


# -- _chunk_and_translate ----------------------------------------------


class TestChunkAndTranslate:
    def test_single_chunk(self):
        translator = _make_translator()
        tokenizer = _make_tokenizer()
        result = _chunk_and_translate("Hello world.", translator, tokenizer)
        assert isinstance(result, str)
        assert len(result) > 0
        assert translator.translate_batch.call_count == 1

    def test_long_text_multiple_chunks(self):
        translator = _make_translator()
        tokenizer = _make_tokenizer()
        sentences = [f"Sentence number {i} here." for i in range(200)]
        text = " ".join(sentences)
        result = _chunk_and_translate(text, translator, tokenizer)
        assert isinstance(result, str)
        assert translator.translate_batch.call_count >= 2

    def test_empty_text(self):
        translator = _make_translator()
        tokenizer = _make_tokenizer()
        assert _chunk_and_translate("", translator, tokenizer) == ""

    def test_single_oversized_sentence(self):
        """A single 600-word sentence must be split and translated
        as multiple chunks — not sent as one oversized input."""
        translator = _make_translator()
        tokenizer = _make_tokenizer()
        text = " ".join(f"word{i}" for i in range(600))
        result = _chunk_and_translate(text, translator, tokenizer)
        assert isinstance(result, str)
        assert len(result) > 0
        assert translator.translate_batch.call_count >= 2


# -- translate (integration with chunking) ------------------------------


class TestTranslateWithChunking:
    @patch("main.get_translator")
    def test_mixed_short_and_long(self, mock_get):
        tokenizer = _make_tokenizer()
        translator = _make_translator()
        mock_get.return_value = (translator, tokenizer)

        short_text = "Hello."
        long_text = " ".join(f"word{i}" for i in range(600))

        results = translate([short_text, long_text], "ja", "en")
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
        assert translator.translate_batch.call_count >= 2

    @patch("main.get_translator")
    def test_short_batch_uses_quality_params(self, mock_get):
        tokenizer = _make_tokenizer()
        translator = _make_translator()
        mock_get.return_value = (translator, tokenizer)

        translate(["hello", "world"], "fr", "en")

        call_kwargs = translator.translate_batch.call_args
        assert call_kwargs.kwargs["repetition_penalty"] == 1.2
        assert call_kwargs.kwargs["no_repeat_ngram_size"] == 3
        assert "max_decoding_length" in call_kwargs.kwargs
