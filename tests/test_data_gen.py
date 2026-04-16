"""Интеграционные тесты генерации данных и загрузчика (ml_core/data.py, ml_core/loader.py)."""

import numpy as np
from ml_core.data import generate_synthetic_data_by_category
from ml_core.loader import detect_sheet_group


class TestDataGeneration:
    def test_generate_grades_data(self):
        """Генерация данных категории 'grades'."""
        result = generate_synthetic_data_by_category("grades", n_students=50)
        df = result["data"]

        assert len(df) == 50
        # Проверяем наличие ключевых колонок
        assert "avg_grade" in df.columns
        assert "risk_flag" in df.columns
        # Проверяем типы данных
        assert df["risk_flag"].dtype in [np.int64, np.int32, np.int8]

    def test_generate_psychology_data(self):
        """Генерация данных категории 'psychology'."""
        result = generate_synthetic_data_by_category("psychology", n_students=20)
        df = result["data"]

        assert len(df) == 20
        assert "stress_level" in df.columns


class TestLoaderLogic:
    def test_detect_sheet_group_numeric(self):
        """Определение категории листа (Numeric)."""
        columns = ["Любознательность", "Воображение", "Сумма"]
        group = detect_sheet_group(columns)
        assert group == "numeric"

    def test_detect_sheet_group_skip(self):
        """Определение категории листа (Skip/Mednik)."""
        columns = ["случайная;", "вечерняя;", "обратно;"]
        group = detect_sheet_group(columns)
        assert group == "skip"

    def test_detect_sheet_group_single_choice(self):
        """Определение категории листа (Single Choice)."""
        columns = ["Мне нравится работать в команде", "организаторские способности", "дисциплинированный"]
        group = detect_sheet_group(columns)
        assert group == "single_choice"
