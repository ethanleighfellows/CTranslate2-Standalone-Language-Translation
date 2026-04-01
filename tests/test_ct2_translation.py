"""Tests for the CTranslate2 translation functions."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import (
    translate,
    evict_model,
    _model_cache,
    _cache_lock,
    ModelNotFoundError,
    extract_codespans,
    restore_codespans,
)


class TestTranslate:
    def test_empty_string_returns_empty(self):
        result = translate("", "fr", "en")
        assert result == ""

    def test_empty_list_returns_empty(self):
        result = translate([], "fr", "en")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        result = translate("   ", "fr", "en")
        assert result == ""

    def test_mixed_empty_and_whitespace(self):
        result = translate(["", " ", ""], "fr", "en")
        assert result == ["", "", ""]

    @patch("main.get_translator")
    def test_single_string(self, mock_get):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda t: list(range(len(t.split()))) + [0]
        mock_tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [f"t{i}" for i in ids]
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        mock_tokenizer.decode.side_effect = lambda ids, **kw: "bonjour"

        mock_translator = MagicMock()
        result_obj = MagicMock()
        result_obj.hypotheses = [["bonjour"]]
        mock_translator.translate_batch.return_value = [result_obj]

        mock_get.return_value = (mock_translator, mock_tokenizer)

        result = translate("hello", "fr", "en")
        assert result == "bonjour"

    @patch("main.get_translator")
    def test_batch(self, mock_get):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda t: list(range(len(t.split()))) + [0]
        mock_tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [f"t{i}" for i in ids]
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        mock_tokenizer.decode.side_effect = lambda ids, **kw: "traduit"

        mock_translator = MagicMock()
        result_obj = MagicMock()
        result_obj.hypotheses = [["traduit"]]
        mock_translator.translate_batch.return_value = [result_obj, result_obj]

        mock_get.return_value = (mock_translator, mock_tokenizer)

        result = translate(["hello", "world"], "fr", "en")
        assert result == ["traduit", "traduit"]
        mock_translator.translate_batch.assert_called_once()

    @patch("main.get_translator")
    def test_preserves_empty_slots(self, mock_get):
        """Empty strings in batch are preserved at correct positions."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda t: list(range(len(t.split()))) + [0]
        mock_tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [f"t{i}" for i in ids]
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        mock_tokenizer.decode.side_effect = lambda ids, **kw: "out"

        mock_translator = MagicMock()
        result_obj = MagicMock()
        result_obj.hypotheses = [["out"]]
        mock_translator.translate_batch.return_value = [result_obj]

        mock_get.return_value = (mock_translator, mock_tokenizer)

        result = translate(["", "hello", ""], "fr", "en")
        assert result == ["", "out", ""]


class TestEvictModel:
    def test_evict_existing(self):
        with _cache_lock:
            _model_cache["en-it-cpu"] = ("translator", "tokenizer")
        evict_model("en", "it")
        assert "en-it-cpu" not in _model_cache

    def test_evict_nonexistent(self):
        evict_model("xx", "yy")  # should not raise

    def test_evict_multiple_devices(self):
        with _cache_lock:
            _model_cache["en-fr-cpu"] = ("t1", "tok1")
            _model_cache["en-fr-cuda"] = ("t2", "tok2")
        evict_model("en", "fr")
        assert "en-fr-cpu" not in _model_cache
        assert "en-fr-cuda" not in _model_cache


class TestCodespanProtection:
    def test_extract_inline_code(self):
        text, spans = extract_codespans("Use `print()` here")
        assert "`print()`" not in text
        assert "[XXX_0_XXX]" in text
        assert spans[0] == "`print()`"

    def test_extract_block_code(self):
        text, spans = extract_codespans("Before\n```python\nprint('hi')\n```\nAfter")
        assert "```" not in text
        assert len(spans) == 1

    def test_restore_codespans(self):
        spans = {0: "`code`", 1: "```block```"}
        text = "Use [XXX_0_XXX] and [XXX_1_XXX]"
        restored = restore_codespans(text, spans)
        assert restored == "Use `code` and ```block```"

    def test_restore_with_spacing_variations(self):
        """Translation models sometimes insert spaces in placeholders."""
        spans = {0: "`code`"}
        text = "Use [ XXX _ 0 _ XXX ]"
        restored = restore_codespans(text, spans)
        assert restored == "Use `code`"

    def test_no_spans_returns_unchanged(self):
        assert restore_codespans("hello world", {}) == "hello world"
