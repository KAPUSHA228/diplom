"""Тестирование кластеризации, корреляций и визуализации."""

import os
import numpy as np
import pandas as pd
from ml_core.analysis import (
    plot_clusters_pca,
    plot_clusters_2d,
    plot_corr_heatmap,
    analyze_cluster_profiles,
)
import pytest
from ml_core.crosstab import create_crosstab, create_multi_crosstab, simple_crosstab


class TestClusterVisualization:
    """Визуализация кластеров."""

    def test_pca_with_single_feature(self):
        """Один признак — fallback без PCA."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "cluster": [0, 0, 1, 1]})
        fig = plot_clusters_pca(df, df["cluster"], features=["a"])
        assert hasattr(fig, "data")

    def test_pca_2d_matplotlib(self, tmp_path):
        """matplotlib-визуализация кластеров."""
        X = np.random.rand(50, 4)
        labels = np.random.randint(0, 3, 50)
        output = str(tmp_path / "clusters.png")
        pca = plot_clusters_2d(X, labels, output_path=output)
        assert os.path.exists(output)
        assert pca.n_components_ == 2

    def test_correlation_heatmap_plotly(self):
        """Plotly heatmap корреляционной матрицы."""
        corr = pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])
        fig = plot_corr_heatmap(corr, title="Test")
        assert hasattr(fig, "data")

    def test_cluster_profiles_computation(self):
        """Профили кластеров."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [4.0, 3.0, 2.0, 1.0],
                "cluster": [0, 0, 1, 1],
            }
        )
        profiles = analyze_cluster_profiles(df, features=["a", "b"])
        assert len(profiles) == 2
        assert "size" in profiles.columns
        assert "size_pct" in profiles.columns


class TestCrosstabValidation:
    """Валидация входных данных кросс-таблицы."""

    def test_raises_when_row_variable_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="не найдена"):
            create_crosstab(df, "nonexistent", "b")

    def test_raises_when_column_variable_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="не найдена"):
            create_crosstab(df, "a", "nonexistent")

    def test_raises_when_numeric_too_many_unique(self):
        df = pd.DataFrame({"id": range(100), "flag": [0, 1] * 50})
        with pytest.raises(ValueError, match="числовой"):
            create_crosstab(df, "id", "flag")


class TestCrosstabAggregation:
    """Агрегация значений вместо подсчёта частот."""

    def test_pivot_table_with_mean(self):
        df = pd.DataFrame(
            {
                "gender": ["M", "F", "M", "F", "M", "F"],
                "group": ["A", "A", "B", "B", "A", "B"],
                "score": [80, 90, 70, 85, 75, 95],
            }
        )
        result = create_crosstab(df, "gender", "group", values="score", aggfunc="mean")
        assert "table" in result
        assert result.get("chi2_test") is None

    def test_normalize_skips_chi2(self):
        df = pd.DataFrame(
            {
                "gender": ["M", "F", "M", "F", "M", "F"],
                "flag": [0, 0, 1, 0, 1, 1],
            }
        )
        result = create_crosstab(df, "gender", "flag", normalize="index")
        assert "table" in result
        assert result.get("chi2_test") is None


class TestMultiCrosstabAnalysis:
    """Мульти-кросстаб: анализ нескольких переменных."""

    def test_skips_missing_variables(self):
        df = pd.DataFrame({"gender": ["M", "F", "M", "F"], "flag": [0, 0, 1, 1]})
        result = create_multi_crosstab(df, ["gender", "nonexistent"], "flag")
        assert "gender" in result
        assert "nonexistent" not in result

    def test_catches_errors_per_variable(self):
        df = pd.DataFrame({"numeric_id": range(100), "flag": [0, 1] * 50})
        result = create_multi_crosstab(df, ["numeric_id"], "flag")
        assert "numeric_id" in result
        assert "error" in result["numeric_id"]


class TestSimpleCrosstab:
    """Упрощённая кросс-таблица — fallback без графиков."""

    def test_returns_error_when_both_variables_missing(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = simple_crosstab(df, "x", "y")
        assert "error" in result
