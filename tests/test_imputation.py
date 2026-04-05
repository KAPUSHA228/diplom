"""Тесты для ml_core/imputation.py"""
import pytest
import pandas as pd
import numpy as np
from ml_core.imputation import handle_missing_values, detect_outliers


class TestHandleMissingValues:
    """Тесты handle_missing_values."""

    def test_auto_strategy_drops_high_missing_columns(self):
        """Колонки с >30% пропусков удаляются."""
        df = pd.DataFrame({
            "keep": [1.0, 2.0, 3.0, 4.0, 5.0],
            "drop_me": [np.nan, np.nan, np.nan, np.nan, 5.0],  # 80% nan
        })
        result, report = handle_missing_values(df, strategy="auto", threshold=30)
        assert "drop_me" not in result.columns

    def test_auto_strategy_fills_medium_missing(self):
        """Колонки с ~10% пропусков заполняются."""
        df = pd.DataFrame({
            "col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan],  # 10% nan
        })
        result, report = handle_missing_values(df, strategy="auto")
        # Главное — пропуски заполнены (неважно чем)
        assert result["col"].isna().sum() == 0

    def test_fill_median_strategy(self, sample_students_with_nans):
        result, report = handle_missing_values(
            sample_students_with_nans, strategy="fill_median"
        )
        assert result.isna().sum().sum() == 0

    def test_fill_mean_strategy(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        result, _ = handle_missing_values(df, strategy="fill_mean")
        assert result["a"].isna().sum() == 0
        assert abs(result["a"].iloc[2] - 7.0 / 3) < 0.001

    def test_drop_rows_strategy(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [5.0, 6.0, np.nan, 8.0],
        })
        result, report = handle_missing_values(df, strategy="drop_rows")
        assert result.isna().sum().sum() == 0
        assert len(result) == 2  # строки 1 и 2 удалены

    def test_interpolate_strategy(self):
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
        result, _ = handle_missing_values(df, strategy="interpolate")
        assert result["a"].isna().sum() == 0
        assert abs(result["a"].iloc[1] - 2.0) < 0.01
        assert abs(result["a"].iloc[2] - 3.0) < 0.01

    def test_report_contains_actions(self, sample_students_with_nans):
        _, report = handle_missing_values(sample_students_with_nans, strategy="auto")
        assert "actions" in report
        assert isinstance(report["actions"], list)

    def test_report_shape_changes(self, sample_students_with_nans):
        _, report = handle_missing_values(sample_students_with_nans, strategy="auto")
        assert "original_shape" in report
        assert "final_shape" in report

    def test_no_missing_no_change(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result, report = handle_missing_values(df, strategy="auto")
        assert result.shape == df.shape
        assert report["missing_before"] == 0


class TestDetectOutliers:
    """Тесты detect_outliers."""

    def test_iqr_detects_outliers(self):
        df = pd.DataFrame({"a": list(range(100)) + [1000, -1000]})
        result = detect_outliers(df, columns=["a"], method="iqr")
        assert result["a"]["n_outliers"] >= 2

    def test_zscore_detects_outliers(self):
        df = pd.DataFrame({"a": [0] * 50 + [100, -100]})
        result = detect_outliers(df, columns=["a"], method="zscore")
        assert result["a"]["n_outliers"] >= 1

    def test_no_outliers_in_clean_data(self):
        df = pd.DataFrame({"a": np.random.randn(100)})
        result = detect_outliers(df, columns=["a"], method="iqr", threshold=3.0)
        # При широком пороге выбросов быть не должно
        assert result["a"]["n_outliers"] <= 3  # допускаем 1-2

    def test_report_has_expected_keys(self):
        df = pd.DataFrame({"a": [1, 2, 3, 100]})
        result = detect_outliers(df, columns=["a"])
        assert "n_outliers" in result["a"]
        assert "percentage" in result["a"]
        # В реализации lower_bound / upper_bound, а не bounds
        assert "lower_bound" in result["a"] or "upper_bound" in result["a"]

    def test_all_columns_by_default(self):
        df = pd.DataFrame({"a": [1] * 100 + [1000], "b": [2] * 101})
        result = detect_outliers(df, method="iqr")
        assert "a" in result
        assert "b" in result

    def test_empty_dataframe(self):
        result = detect_outliers(pd.DataFrame())
        assert result == {}
