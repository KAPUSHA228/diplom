"""Тесты для ml_core/crosstab.py"""
import pytest
import pandas as pd
import numpy as np
from ml_core.crosstab import (
    create_crosstab,
    simple_crosstab,
    create_multi_crosstab,
)


class TestCreateCrosstab:
    """Тесты create_crosstab."""

    def test_returns_dict_with_table(self, sample_categorical_data):
        result = create_crosstab(
            sample_categorical_data, "gender", "risk_flag"
        )
        assert isinstance(result, dict)
        assert "table" in result

    def test_table_is_dataframe(self, sample_categorical_data):
        result = create_crosstab(
            sample_categorical_data, "gender", "risk_flag"
        )
        assert isinstance(result["table"], pd.DataFrame)

    def test_chi2_test_present(self, sample_categorical_data):
        result = create_crosstab(
            sample_categorical_data, "gender", "risk_flag"
        )
        assert "chi2_test" in result
        assert isinstance(result["chi2_test"], dict)

    def test_heatmap_present(self, sample_categorical_data):
        result = create_crosstab(
            sample_categorical_data, "gender", "risk_flag"
        )
        assert "heatmap" in result

    def test_stacked_bar_present(self, sample_categorical_data):
        result = create_crosstab(
            sample_categorical_data, "gender", "risk_flag"
        )
        assert "stacked_bar" in result

    def test_with_values_and_aggfunc(self, sample_categorical_data):
        result = create_crosstab(
            sample_categorical_data, "gender", "risk_flag",
            values="satisfaction_group", aggfunc="count"
        )
        assert "table" in result


class TestSimpleCrosstab:
    """Тесты simple_crosstab."""

    def test_returns_table_and_chi2(self, sample_categorical_data):
        result = simple_crosstab(
            sample_categorical_data, "gender", "risk_flag"
        )
        assert "table" in result
        assert "chi2_test" in result

    def test_no_heatmap(self, sample_categorical_data):
        result = simple_crosstab(
            sample_categorical_data, "gender", "risk_flag"
        )
        assert "heatmap" not in result


class TestMultiCrosstab:
    """Тесты create_multi_crosstab."""

    def test_returns_dict_per_variable(self, sample_categorical_data):
        variables = ["gender", "year"]
        result = create_multi_crosstab(
            sample_categorical_data, variables, "risk_flag"
        )
        assert isinstance(result, dict)
        for var in variables:
            assert var in result
