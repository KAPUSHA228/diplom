"""Интеграционные тесты для ml_core/analyzer.py (ResearchAnalyzer)"""
import pytest
import pandas as pd
import numpy as np
from ml_core.analyzer import ResearchAnalyzer


@pytest.fixture
def analyzer():
    return ResearchAnalyzer()


@pytest.fixture
def full_dataset(rng=np.random.RandomState(42)):
    """Полный датасет со всеми признаками."""
    n = 200
    return pd.DataFrame({
        "student_id": range(n),
        "avg_grade": rng.uniform(2.0, 5.0, n),
        "grade_std": rng.uniform(0.1, 1.0, n),
        "min_grade": rng.uniform(2.0, 3.5, n),
        "max_grade": rng.uniform(3.5, 5.0, n),
        "n_courses": rng.randint(3, 10, n),
        "avg_brs": rng.uniform(40, 100, n),
        "attendance_rate": rng.uniform(0.4, 1.0, n),
        "satisfaction_score": rng.uniform(1.0, 5.0, n),
        "engagement_score": rng.uniform(1.0, 5.0, n),
        "workload_perception": rng.uniform(1.0, 5.0, n),
        "stress_level": rng.uniform(0.0, 10.0, n),
        "motivation_score": rng.uniform(0.0, 10.0, n),
        "anxiety_score": rng.uniform(0.0, 10.0, n),
        "n_essays": rng.randint(0, 5, n),
        "avg_essay_grade": rng.uniform(2.0, 5.0, n),
        "risk_flag": (rng.random(n) > 0.7).astype(int),
    })


@pytest.mark.integration
class TestResearchAnalyzer:
    """Интеграционные тесты полного пайплайна."""

    def test_run_full_analysis_returns_result(self, analyzer, full_dataset):
        result = analyzer.run_full_analysis(
            df=full_dataset,
            target_col="risk_flag",
            n_clusters=3,
            risk_threshold=0.5,
            corr_threshold=0.3,
            is_synthetic=True,
            use_smote=True,
        )
        assert result is not None
        assert result.status == "success"

    def test_full_analysis_has_metrics(self, analyzer, full_dataset):
        result = analyzer.run_full_analysis(
            df=full_dataset,
            target_col="risk_flag",
            n_clusters=3,
            risk_threshold=0.5,
            corr_threshold=0.3,
            is_synthetic=True,
            use_smote=True,
        )
        assert result.metrics is not None

    def test_full_analysis_has_cluster_profiles(self, analyzer, full_dataset):
        result = analyzer.run_full_analysis(
            df=full_dataset,
            target_col="risk_flag",
            n_clusters=3,
            risk_threshold=0.5,
            corr_threshold=0.3,
            is_synthetic=True,
            use_smote=True,
        )
        assert result.cluster_profiles is not None

    def test_full_analysis_has_explanations(self, analyzer, full_dataset):
        result = analyzer.run_full_analysis(
            df=full_dataset,
            target_col="risk_flag",
            n_clusters=3,
            risk_threshold=0.5,
            corr_threshold=0.3,
            is_synthetic=True,
            use_smote=True,
        )
        assert result.explanations is not None
        assert len(result.explanations) > 0


class TestResearchAnalyzerHelperMethods:
    """Тесты вспомогательных методов ResearchAnalyzer."""

    def test_create_composite_score(self, analyzer):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        result, name = analyzer.create_composite_score(
            df, feature_weights={"a": 0.7, "b": 0.3}, score_name="test_score"
        )
        assert name == "test_score"
        assert name in result.columns

    def test_select_subset_by_condition(self, analyzer, full_dataset):
        subset = analyzer.select_subset(
            full_dataset,
            condition="avg_grade > 4.0",
        )
        assert (subset["avg_grade"] > 4.0).all()

    def test_select_subset_by_n_samples(self, analyzer, full_dataset):
        subset = analyzer.select_subset(
            full_dataset,
            n_samples=10,
            random_state=42,
        )
        assert len(subset) == 10

    def test_select_subset_by_cluster(self, analyzer, full_dataset):
        full_dataset["cluster"] = np.random.choice([0, 1, 2], len(full_dataset))
        subset = analyzer.select_subset(
            full_dataset,
            by_cluster=0,
        )
        assert (subset["cluster"] == 0).all()
