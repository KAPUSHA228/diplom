"""Тесты для детекции дрейфа данных (ml_core/drift_detector.py)"""

import pandas as pd
import numpy as np
from ml_core.drift_detector import DataDriftDetector, generate_recommendations


class TestDataDrift:
    """Проверка работы детектора дрейфа: инициализация, обнаружение, отчеты."""

    def setup_method(self):
        # Референсные данные
        np.random.seed(42)
        self.ref_data = pd.DataFrame(
            {
                "num_col": np.random.normal(0, 1, 100),
                "cat_col": np.random.choice(["A", "B", "C"], 100),
                "int_col": np.random.randint(1, 10, 100),
            }
        )

    def test_detector_initialization(self):
        """Простая инициализация (покрывает __init__)."""
        detector = DataDriftDetector(self.ref_data)
        assert len(detector.reference_data) == 100
        # Исправлено: в коде используется атрибут numerical_features
        assert "num_col" in detector.numerical_features
        assert "cat_col" in detector.categorical_features

    def test_detect_drift_no_drift(self):
        """Ситуация без дрейфа (данные те же)."""
        detector = DataDriftDetector(self.ref_data)
        report = detector.detect_drift(self.ref_data)

        assert "overall_drift" in report
        assert "drifted_features" in report
        assert isinstance(report["drifted_features"], list)

    def test_detect_drift_with_shift(self):
        """Ситуация с сильным дрейфом (сдвиг распределения)."""
        drift_data = pd.DataFrame(
            {
                "num_col": np.random.normal(10, 5, 100),  # Сильный сдвиг
                "cat_col": np.random.choice(["A", "B", "C"], 100),
                "int_col": np.random.randint(1, 10, 100),
            }
        )

        detector = DataDriftDetector(self.ref_data)
        report = detector.detect_drift(drift_data)

        # "num_col" должен быть определен как дрейфующий
        assert "num_col" in report["drifted_features"]
        assert report["overall_drift"] is True

    def test_detect_drift_cat_shift(self):
        """Ситуация с дрейфом категориальной колонки."""
        drift_data = pd.DataFrame(
            {
                "num_col": np.random.normal(0, 1, 100),
                "cat_col": np.random.choice(["D", "E"], 100),  # Новые категории
                "int_col": np.random.randint(1, 10, 100),
            }
        )

        detector = DataDriftDetector(self.ref_data)
        report = detector.detect_drift(drift_data)

        assert "cat_col" in report["drifted_features"]

    def test_generate_recommendations(self):
        """Покрытие метода генерации рекомендаций."""
        # Исправлено: функция находится на уровне модуля, а не в классе
        # TODO
        # detector = DataDriftDetector(self.ref_data)

        report = {"overall_drift": True, "drift_percentage": 50.0, "drifted_features": ["num_col", "cat_col"]}

        recs = generate_recommendations(report)
        assert isinstance(recs, list)
        assert len(recs) > 0
        assert any("переобуч" in r.lower() or "модель" in r.lower() for r in recs)

    def test_generate_recommendations_low_drift(self):
        """Рекомендации при низком дрейфе."""
        # Исправлено: вызов функции модуля
        # TODO
        # detector = DataDriftDetector(self.ref_data)

        report = {"overall_drift": False, "drift_percentage": 5.0, "drifted_features": []}

        recs = generate_recommendations(report)
        assert isinstance(recs, list)
        # Должно быть сообщение о том, что всё ок (ищем "стабильн" или "использовать")
        assert any("стабильн" in r.lower() or "использовать" in r.lower() for r in recs)
