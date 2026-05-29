"""Интеграционные тесты генерации данных и загрузчика (ml_core/data.py, ml_core/loader.py)."""

import numpy as np
from ml_core.data import generate_synthetic_data_by_category
from ml_core.loader import detect_sheet_group, get_sheet_category


class TestDataGeneration:
    def test_generate_grades_data(self):
        result = generate_synthetic_data_by_category("grades", n_students=50)
        df = result["data"]

        assert len(df) == 50
        assert "avg_grade" in df.columns
        assert "risk_flag" in df.columns
        assert df["risk_flag"].dtype in [np.int64, np.int32, np.int8]

    def test_generate_psychology_data(self):
        result = generate_synthetic_data_by_category("psychology", n_students=20)
        df = result["data"]

        assert len(df) == 20
        assert "stress_level" in df.columns


class TestLoaderLogic:
    def test_detect_sheet_group_numeric(self):
        assert detect_sheet_group("Вильямс") == "numeric"
        assert get_sheet_category("Шварц") == "numeric"

    def test_detect_sheet_group_skip(self):
        assert detect_sheet_group("Медник") == "skip"

    def test_detect_sheet_group_single_choice(self):
        assert detect_sheet_group("Соц1") == "single_choice"

    def test_detect_sheet_group_unknown(self):
        assert detect_sheet_group("Random Sheet") == "unknown"
