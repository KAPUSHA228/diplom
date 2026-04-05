"""Тесты для ml_core/schemas.py"""
import pytest
import pandas as pd
from ml_core.schemas import (
    AnalysisRequest,
    CompositeScoreRequest,
    AnalysisResult,
    TrajectoryRequest,
)


class TestAnalysisRequest:
    """Тесты AnalysisRequest."""

    def test_defaults(self):
        req = AnalysisRequest(
            df=[{"a": 1, "b": 2}],
            target_col="risk_flag",
        )
        assert req.target_col == "risk_flag"
        assert req.n_clusters == 3
        assert req.risk_threshold == 0.5
        assert req.corr_threshold == 0.3
        assert req.use_smote is True
        assert req.use_hp_tuning is False
        assert req.use_composite_features is True

    def test_custom_values(self):
        req = AnalysisRequest(
            df=[{"a": 1}],
            target_col="custom_target",
            n_clusters=5,
            risk_threshold=0.7,
            use_smote=False,
        )
        assert req.target_col == "custom_target"
        assert req.n_clusters == 5
        assert req.risk_threshold == 0.7
        assert req.use_smote is False

    def test_n_clusters_validation(self):
        """n_clusters должен быть в диапазоне."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AnalysisRequest(df=[{"a": 1}], n_clusters=1)  # < 2
        with pytest.raises(Exception):
            AnalysisRequest(df=[{"a": 1}], n_clusters=10)  # > 8


class TestCompositeScoreRequest:
    """Тесты CompositeScoreRequest."""

    def test_basic(self):
        req = CompositeScoreRequest(
            df=[{"a": 1, "b": 2}],
            feature_weights={"a": 0.6, "b": 0.4},
        )
        assert req.score_name == "custom_score"
        assert req.normalize is True

    def test_custom_name(self):
        req = CompositeScoreRequest(
            df=[{"a": 1}],
            feature_weights={"a": 1.0},
            score_name="my_score",
            normalize=False,
        )
        assert req.score_name == "my_score"
        assert req.normalize is False


class TestTrajectoryRequest:
    """Тесты TrajectoryRequest."""

    def test_defaults(self):
        req = TrajectoryRequest(
            df=[{"student_id": 1, "semester": 1, "avg_grade": 4.0}],
            student_id=1,
            value_col="avg_grade",
        )
        assert req.time_col == "semester"


class TestAnalysisResult:
    """Тесты AnalysisResult."""

    def test_basic(self):
        result = AnalysisResult(status="success")
        assert result.status == "success"

    def test_error_status(self):
        result = AnalysisResult(
            status="error",
            message="Something went wrong",
        )
        assert result.status == "error"
        assert result.message == "Something went wrong"
