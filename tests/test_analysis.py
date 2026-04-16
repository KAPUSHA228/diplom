"""Тесты для ml_core/analysis.py"""

import pandas as pd
import numpy as np
from ml_core.analysis import (
    cluster_students,
    analyze_cluster_profiles,
    plot_clusters_pca,
    correlation_analysis,
)


class TestCorrelationAnalysis:
    """Тесты correlation_analysis (теперь возвращает dict)."""

    def test_returns_correlation_matrix(self, sample_students):
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        result = correlation_analysis(sample_students, features, "risk_flag")
        assert isinstance(result, dict)
        matrix = result["full_matrix"]
        assert "risk_flag" in matrix.columns
        assert "risk_flag" in matrix.index

    def test_correlation_values_in_range(self, sample_students):
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        result = correlation_analysis(sample_students, features, "risk_flag")
        matrix = result["full_matrix"]
        assert (matrix.values >= -1.0).all()
        assert (matrix.values <= 1.0).all()

    def test_diagonal_is_one(self, sample_students):
        features = ["avg_grade", "stress_level"]
        result = correlation_analysis(sample_students, features, "risk_flag")
        matrix = result["full_matrix"]
        # Проверяем только числовые колонки (без risk_flag который может быть int)
        for col in features:
            assert abs(matrix.loc[col, col] - 1.0) < 0.001


class TestCorrelationAnalysisEnhanced:
    """Тесты correlation_analysis (расширенный формат)."""

    def test_returns_dict_with_expected_keys(self, sample_students):
        features = ["avg_grade", "stress_level", "satisfaction_score", "motivation_score", "anxiety_score"]
        result = correlation_analysis(sample_students, features, "risk_flag", corr_threshold=0.3)
        assert isinstance(result, dict)
        assert "full_matrix" in result
        assert "strong_matrix" in result
        assert "target_correlations" in result

    def test_strong_matrix_filters_by_threshold(self, sample_students):
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        result = correlation_analysis(sample_students, features, "risk_flag", corr_threshold=0.8)
        # При пороге 0.8 большинство корреляций должно быть отфильтровано
        strong_matrix = result["strong_matrix"]
        # Проверяем, что оставшиеся значения (кроме диагонали) >= 0.8 или NaN
        non_diag = strong_matrix.values[~np.eye(len(strong_matrix), dtype=bool)]
        valid_vals = non_diag[~np.isnan(non_diag)]
        if len(valid_vals) > 0:
            assert (valid_vals >= 0.8).all()


class TestClusterStudents:
    """Тесты cluster_students."""

    def test_returns_three_items(self, sample_students):
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        result = cluster_students(sample_students[features], n_clusters=3)
        assert len(result) == 3

    def test_labels_have_correct_length(self, sample_students):
        features = ["avg_grade", "stress_level"]
        labels, kmeans, scaler = cluster_students(sample_students[features], n_clusters=3)
        assert len(labels) == len(sample_students)

    def test_labels_are_contiguous(self, sample_students):
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        labels, _, _ = cluster_students(sample_students[features], n_clusters=3)
        unique_labels = set(labels)
        assert unique_labels == {0, 1, 2}

    def test_different_n_clusters(self, sample_students):
        features = ["avg_grade", "stress_level"]
        labels, _, _ = cluster_students(sample_students[features], n_clusters=5)
        assert len(set(labels)) == 5


class TestAnalyzeClusterProfiles:
    """Тесты analyze_cluster_profiles."""

    def test_returns_dataframe(self, sample_students):
        features = ["avg_grade", "stress_level"]
        sample_students["cluster"] = [0, 1, 2] * 33 + [0]
        result = analyze_cluster_profiles(sample_students, features)
        assert isinstance(result, pd.DataFrame)

    def test_profile_has_cluster_size(self, sample_students):
        features = ["avg_grade"]
        sample_students["cluster"] = np.random.choice([0, 1, 2], len(sample_students))
        result = analyze_cluster_profiles(sample_students, features)
        # В реализации поле называется 'size', а не 'cluster_size'
        assert "size" in result.columns

    def test_profile_count_matches_clusters(self, sample_students):
        features = ["avg_grade", "stress_level"]
        sample_students["cluster"] = np.random.choice([0, 1, 2], len(sample_students))
        result = analyze_cluster_profiles(sample_students, features)
        assert len(result) == 3


class TestPlotClustersPca:
    """Тесты plot_clusters_pca."""

    def test_returns_plotly_figure(self, sample_students):
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        sample_students["cluster"] = np.random.choice([0, 1, 2], len(sample_students))
        fig = plot_clusters_pca(sample_students, sample_students["cluster"], features)
        # Plotly figure имеет атрибут layout
        assert hasattr(fig, "layout")
