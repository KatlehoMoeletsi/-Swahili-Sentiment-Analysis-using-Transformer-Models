"""
tests/test_preprocess.py
------------------------
Unit tests for src/preprocess.py

Run:
    pytest tests/
    # or from project root:
    python -m pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import clean_text, preprocess, LABEL_MAP
import pandas as pd


# ─────────────────────────────────────────────
# clean_text tests
# ─────────────────────────────────────────────

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO World") == "hello world"

    def test_removes_url(self):
        assert "http" not in clean_text("Visit https://example.com today")
        assert "example" not in clean_text("Visit https://example.com today")

    def test_removes_mention(self):
        assert "@user" not in clean_text("Hello @user how are you")

    def test_strips_hashtag_symbol_but_keeps_word(self):
        result = clean_text("Great day #Kenya")
        assert "#" not in result
        assert "kenya" in result

    def test_removes_digits(self):
        result = clean_text("I have 3 cats and 12 dogs")
        assert "3" not in result
        assert "12" not in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! How's it going?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self):
        result = clean_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string_input(self):
        # Should not raise; returns empty string or str representation
        result = clean_text(None)
        assert isinstance(result, str)

    def test_swahili_text_preserved(self):
        text = "Habari za asubuhi"
        result = clean_text(text)
        assert "habari" in result
        assert "asubuhi" in result

    def test_mixed_swahili_english(self):
        text = "Leo ni very good day kweli"
        result = clean_text(text)
        assert "leo" in result
        assert "good" in result
        assert "kweli" in result


# ─────────────────────────────────────────────
# preprocess (DataFrame) tests
# ─────────────────────────────────────────────

class TestPreprocess:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "tweet": [
                "Habari! #Kenya https://t.co/abc @user",
                "Leo ni siku nzuri sana 😊",
                "Ninachukia hali hii sana!",
            ],
            "label": ["positive", "neutral", "negative"],
        })

    def test_adds_clean_text_column(self, sample_df):
        result = preprocess(sample_df)
        assert "clean_text" in result.columns

    def test_adds_label_encoded_column(self, sample_df):
        result = preprocess(sample_df)
        assert "label_encoded" in result.columns

    def test_label_encoding_correct(self, sample_df):
        result = preprocess(sample_df)
        assert result.loc[result["label"] == "positive", "label_encoded"].iloc[0] == 2
        assert result.loc[result["label"] == "neutral",  "label_encoded"].iloc[0] == 1
        assert result.loc[result["label"] == "negative", "label_encoded"].iloc[0] == 0

    def test_no_nulls_in_clean_text(self, sample_df):
        result = preprocess(sample_df)
        assert result["clean_text"].notna().all()

    def test_drops_unknown_labels(self):
        df = pd.DataFrame({
            "tweet": ["good text", "another tweet"],
            "label": ["positive", "unknown_label"],
        })
        result = preprocess(df)
        assert len(result) == 1

    def test_original_df_not_mutated(self, sample_df):
        original_cols = set(sample_df.columns)
        preprocess(sample_df)
        assert set(sample_df.columns) == original_cols


# ─────────────────────────────────────────────
# Label mapping tests
# ─────────────────────────────────────────────

class TestLabelMap:
    def test_all_three_labels_present(self):
        assert "negative" in LABEL_MAP
        assert "neutral"  in LABEL_MAP
        assert "positive" in LABEL_MAP

    def test_correct_integer_values(self):
        assert LABEL_MAP["negative"] == 0
        assert LABEL_MAP["neutral"]  == 1
        assert LABEL_MAP["positive"] == 2
