"""Тесты для ml_core/logger.py"""

import os
import pandas as pd
from ml_core.logger import MLLogger


class TestMLLoggerInit:
    """Тесты инициализации MLLogger"""

    def test_creates_log_dir_and_files(self, tmp_path):
        log_dir = tmp_path / "logs"
        # TODO
        # logger = MLLogger(log_dir=str(log_dir))

        assert log_dir.exists()
        assert os.path.exists(str(log_dir / "ml_events.log"))
        assert os.path.exists(str(log_dir / "model_metrics.csv"))

    def test_metrics_csv_has_expected_columns(self, tmp_path):
        log_dir = tmp_path / "logs"
        MLLogger(log_dir=str(log_dir))

        df = pd.read_csv(str(log_dir / "model_metrics.csv"))
        assert "timestamp" in df.columns
        assert "model_name" in df.columns
        assert "f1_score" in df.columns


class TestMLLoggerLogEvent:
    """Тесты log_event"""

    def test_writes_to_log_file(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = MLLogger(log_dir=str(log_dir))
        logger.log_event("test_event", {"key": "value", "count": 42})

        log_file = str(log_dir / "ml_events.log")
        with open(log_file, "r") as f:
            content = f.read()
        assert "test_event" in content
        assert "key" in content


class TestMLLoggerLogModelMetrics:
    """Тесты log_model_metrics"""

    def test_saves_single_metrics(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = MLLogger(log_dir=str(log_dir))

        metrics = {
            "test": {"f1": 0.85, "roc_auc": 0.90, "precision": 0.82, "recall": 0.88},
            "data_size": 1000,
            "features": ["feat_a", "feat_b"],
        }
        logger.log_model_metrics("RF", metrics)

        df = pd.read_csv(str(log_dir / "model_metrics.csv"))
        assert len(df) == 1
        assert df.iloc[0]["model_name"] == "RF"
        assert df.iloc[0]["f1_score"] == 0.85

    def test_appends_multiple_metrics(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = MLLogger(log_dir=str(log_dir))

        m1 = {"test": {"f1": 0.80, "roc_auc": 0.85, "precision": 0.78, "recall": 0.82}}
        m2 = {"test": {"f1": 0.90, "roc_auc": 0.95, "precision": 0.88, "recall": 0.92}}

        logger.log_model_metrics("LR", m1)
        logger.log_model_metrics("XGB", m2)

        df = pd.read_csv(str(log_dir / "model_metrics.csv"))
        assert len(df) == 2
        assert list(df["model_name"]) == ["LR", "XGB"]


class TestMLLoggerLogFeatures:
    """Тесты log_features"""

    def test_logs_feature_list(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = MLLogger(log_dir=str(log_dir))
        logger.log_features(["age", "gpa", "attendance"])

        with open(str(log_dir / "ml_events.log"), "r") as f:
            content = f.read()
        assert "features_used" in content
        assert "age" in content


class TestMLLoggerLogPrediction:
    """Тесты log_prediction"""

    def test_logs_prediction_details(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = MLLogger(log_dir=str(log_dir))
        logger.log_prediction("student_001", prediction=1, probability=0.75)

        with open(str(log_dir / "ml_events.log"), "r") as f:
            content = f.read()
        assert "prediction" in content
        assert "student_001" in content
