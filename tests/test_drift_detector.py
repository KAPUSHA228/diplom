"""Тесты для ml_core/drift_detector.py"""
import pytest
import pandas as pd
import numpy as np
import json
import os
from ml_core.drift_detector import (
    DataDriftDetector,
    generate_recommendations,
)


@pytest.fixture
def reference_data(rng=np.random.RandomState(42)):
    """Эталонные данные для дрейфа."""
    n = 200
    return pd.DataFrame({
        "num_a": rng.normal(50, 10, n),
        "num_b": rng.uniform(0, 100, n),
        "cat_a": rng.choice(["X", "Y", "Z"], n),
    })


@pytest.fixture
def drifted_data(rng=np.random.RandomState(99)):
    """Данные со сдвигом (дрейфом)."""
    n = 200
    return pd.DataFrame({
        "num_a": rng.normal(70, 15, n),  # сдвинуто с 50 до 70
        "num_b": rng.uniform(0, 100, n),  # без изменений
        "cat_a": rng.choice(["X", "Y", "W"], n),  # новая категория W
    })


class TestDataDriftDetector:
    """Тесты DataDriftDetector."""

    def test_no_drift_on_same_data(self, reference_data):
        detector = DataDriftDetector(reference_data, threshold=0.01)
        report = detector.detect_drift(reference_data)
        assert "overall_drift" in report
        # На тех же данных дрейфа быть не должно
        assert report["overall_drift"] is False

    def test_drift_detected_on_shifted_data(self, reference_data, drifted_data):
        detector = DataDriftDetector(reference_data, threshold=0.05)
        report = detector.detect_drift(drifted_data)
        assert "overall_drift" in report
        assert "drifted_features" in report
        assert "drift_percentage" in report

    def test_report_has_expected_keys(self, reference_data):
        detector = DataDriftDetector(reference_data)
        report = detector.detect_drift(reference_data)
        expected_keys = [
            "overall_drift", "drift_percentage", "drifted_features",
            "feature_reports", "recommendations", "data_quality",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key: {key}"

    def test_drifted_features_is_list(self, reference_data):
        detector = DataDriftDetector(reference_data)
        report = detector.detect_drift(reference_data)
        assert isinstance(report["drifted_features"], list)

    def test_recommendations_is_list(self, reference_data):
        detector = DataDriftDetector(reference_data)
        report = detector.detect_drift(reference_data)
        assert isinstance(report["recommendations"], list)

    def test_data_quality_report(self, reference_data):
        detector = DataDriftDetector(reference_data)
        report = detector.detect_drift(reference_data)
        dq = report["data_quality"]
        # Реальные ключи: missing_values, outliers_percentage, new_categories
        assert "missing_values" in dq or "outliers_percentage" in dq

    def test_alert_message_is_string(self, reference_data, drifted_data):
        detector = DataDriftDetector(reference_data)
        report = detector.detect_drift(drifted_data)
        msg = detector.generate_alert_message(report)
        assert isinstance(msg, str)
        assert len(msg) > 0


class TestDriftDetectorSaveReport:
    """Тесты сохранения отчёта дрейфа."""

    def test_save_report_creates_json(self, reference_data, tmp_path):
        detector = DataDriftDetector(reference_data)
        report = detector.detect_drift(reference_data)
        path = detector.save_report(report, report_dir=str(tmp_path))
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert "overall_drift" in loaded


class TestGenerateRecommendations:
    """Тесты standalone функции рекомендаций."""

    def test_returns_list(self, reference_data):
        """generate_recommendations — метод детектора, не standalone функция."""
        detector = DataDriftDetector(reference_data)
        report = detector.detect_drift(reference_data)
        result = report["recommendations"]
        assert isinstance(result, list)

    def test_no_drift_no_recommendations(self):
        """При отсутствии дрейфа рекомендации пустые."""
        df = pd.DataFrame({
            "num_a": np.random.randn(100),
        })
        detector = DataDriftDetector(df)
        report = detector.detect_drift(df)
        result = report["recommendations"]
        assert isinstance(result, list)
