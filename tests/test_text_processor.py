"""Тесты для ml_core/text_processor.py"""
import pytest
import pandas as pd
from ml_core.text_processor import extract_text_features


class TestExtractTextFeatures:
    """Тесты extract_text_features."""

    def test_creates_length_column(self, sample_text_data):
        result = extract_text_features(sample_text_data, "essay_text")
        assert "essay_text_length" in result.columns

    def test_creates_word_count_column(self, sample_text_data):
        result = extract_text_features(sample_text_data, "essay_text")
        assert "essay_text_word_count" in result.columns

    def test_length_is_correct(self, sample_text_data):
        result = extract_text_features(sample_text_data, "essay_text")
        expected = sample_text_data["essay_text"].str.len()
        assert (result["essay_text_length"] == expected).all()

    def test_word_count_is_correct(self, sample_text_data):
        result = extract_text_features(sample_text_data, "essay_text")
        expected = sample_text_data["essay_text"].str.split().str.len()
        assert (result["essay_text_word_count"] == expected).all()

    def test_complexity_is_ratio(self, sample_text_data):
        result = extract_text_features(sample_text_data, "essay_text")
        length = result["essay_text_length"]
        words = result["essay_text_word_count"]
        # В реализации: complexity = length / (word_count + 1)
        expected_complexity = length / (words + 1)
        assert (result["essay_text_complexity"].round(2) == expected_complexity.round(2)).all()

    def test_missing_column_returns_unchanged(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = extract_text_features(df, "missing_col")
        assert "missing_col" not in result.columns
        assert list(result.columns) == ["other_col"]

    def test_empty_text_handled(self):
        df = pd.DataFrame({"text": ["", "hello", ""]})
        result = extract_text_features(df, "text")
        assert result["text_word_count"].iloc[0] == 0
        assert result["text_word_count"].iloc[1] == 1
