"""Tests for Helsinki-NLP model name resolution.

Verifies that _resolve_hf_model_name handles non-standard language codes
(e.g. Helsinki-NLP uses 'jap' instead of 'ja' for Japanese) and falls
back to tc-big variants when the standard model doesn't exist.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import _resolve_hf_model_name


def _fake_model_info(existing_models: set[str]):
    """Return a side_effect function that only succeeds for known models."""
    def _check(name):
        if name in existing_models:
            return MagicMock()
        raise Exception(f"Model {name} not found")
    return _check


class TestResolveHfModelName:

    @patch("main._hf_model_info")
    def test_standard_name_found(self, mock_info):
        """en→fr: standard opus-mt-en-fr exists."""
        mock_info.side_effect = _fake_model_info({
            "Helsinki-NLP/opus-mt-en-fr",
        })
        result = _resolve_hf_model_name("en", "fr")
        assert result == "Helsinki-NLP/opus-mt-en-fr"

    @patch("main._hf_model_info")
    def test_tc_big_fallback(self, mock_info):
        """en→pt: only tc-big variant exists."""
        mock_info.side_effect = _fake_model_info({
            "Helsinki-NLP/opus-mt-tc-big-en-pt",
        })
        result = _resolve_hf_model_name("en", "pt")
        assert result == "Helsinki-NLP/opus-mt-tc-big-en-pt"

    @patch("main._hf_model_info")
    def test_japanese_alias(self, mock_info):
        """en→ja: resolves to opus-mt-en-jap via alias mapping."""
        mock_info.side_effect = _fake_model_info({
            "Helsinki-NLP/opus-mt-en-jap",
        })
        result = _resolve_hf_model_name("en", "ja")
        assert result == "Helsinki-NLP/opus-mt-en-jap"

    @patch("main._hf_model_info")
    def test_japanese_tc_big_fallback(self, mock_info):
        """en→ja: if only tc-big-en-jap exists, finds it."""
        mock_info.side_effect = _fake_model_info({
            "Helsinki-NLP/opus-mt-tc-big-en-jap",
        })
        result = _resolve_hf_model_name("en", "ja")
        assert result == "Helsinki-NLP/opus-mt-tc-big-en-jap"

    @patch("main._hf_model_info")
    def test_korean_tc_big_only(self, mock_info):
        """en→ko: only tc-big variant exists."""
        mock_info.side_effect = _fake_model_info({
            "Helsinki-NLP/opus-mt-tc-big-en-ko",
        })
        result = _resolve_hf_model_name("en", "ko")
        assert result == "Helsinki-NLP/opus-mt-tc-big-en-ko"

    @patch("main._hf_model_info")
    def test_no_model_found_raises(self, mock_info):
        """Raises RuntimeError with all tried candidates when nothing exists."""
        mock_info.side_effect = Exception("not found")
        with pytest.raises(RuntimeError, match="No Helsinki-NLP model found"):
            _resolve_hf_model_name("en", "xx")

    @patch("main._hf_model_info")
    def test_prefers_standard_over_alias(self, mock_info):
        """If both en-ja and en-jap exist, prefer standard ISO code."""
        mock_info.side_effect = _fake_model_info({
            "Helsinki-NLP/opus-mt-en-ja",
            "Helsinki-NLP/opus-mt-en-jap",
        })
        result = _resolve_hf_model_name("en", "ja")
        assert result == "Helsinki-NLP/opus-mt-en-ja"
