"""Тестирование генерации и сохранения синтетических данных."""

import os
import numpy as np
import pandas as pd
from ml_core.data import (
    save_synthetic_data,
    save_data_for_monitoring,
    load_data,
    load_from_csv,
    generate_temporal_data,
    generate_synthetic_data_by_category,
    prepare_data_for_training,
)


class TestSyntheticDataGeneration:
    """Тестирование генерации и сохранения синтетических данных."""

    def test_save_synthetic_data_to_custom_path(self, tmp_path):
        """Сохранение синтетических данных в произвольный путь."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = save_synthetic_data(df, directory=str(tmp_path), filename="custom.csv")
        assert os.path.exists(path)

    def test_save_synthetic_data_auto_filename(self):
        """Автоматическое имя файла."""
        df = pd.DataFrame({"a": [1, 2]})
        path = save_synthetic_data(df)
        assert os.path.exists(path)
        assert path.endswith(".csv")

    def test_save_monitoring_data_excludes_target(self, tmp_path):
        """Мониторинговые данные не включают целевую переменную."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
        path = save_data_for_monitoring(df, feature_cols=["a", "b"], semester="test", directory=str(tmp_path))
        assert os.path.exists(path)
        loaded = pd.read_csv(path)
        assert "target" not in loaded.columns

    def test_load_grades_synthetic_data(self):
        """Загрузка синтетических данных об успеваемости."""
        result = load_data("grades", n_students=50)
        assert isinstance(result, dict)
        assert "data" in result
        assert len(result["data"]) == 50

    def test_load_two_sets_for_drift_detection(self):
        """Генерация двух наборов данных для обнаружения дрейфа."""
        result = load_data("grades", n_students=50, generate_two_sets=True)
        assert "reference" in result
        assert "new" in result
        assert len(result["reference"]) == 50
        assert len(result["new"]) == 50

    def test_generate_all_categories(self):
        """Генерация всех категорий синтетических данных."""
        for category in ["grades", "psychology", "creativity", "values", "personality", "activities", "career"]:
            result = generate_synthetic_data_by_category(category, n_students=50)
            assert isinstance(result, dict)
            assert "data" in result
            assert len(result["data"]) == 50, f"Category {category}: expected 50 rows"

    def test_generate_temporal_data_multiple_semesters(self):
        """Генерация временных данных с несколькими семестрами."""
        result = generate_temporal_data(n_students=50, n_semesters=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 50 * 3

    def test_load_from_csv_file(self, tmp_path):
        """Загрузка данных из CSV-файла."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        filepath = str(tmp_path / "test.csv")
        df.to_csv(filepath, index=False)
        loaded = load_from_csv(filepath)
        assert len(loaded) == 3

    def test_prepare_data_for_training_split(self):
        """Разделение данных на train/test."""
        df = pd.DataFrame(
            {
                "a": np.random.rand(100),
                "b": np.random.rand(100),
                "c": np.random.rand(100),
                "risk_flag": (np.random.rand(100) > 0.5).astype(int),
            }
        )
        X_train, X_test, y_train, y_test = prepare_data_for_training(
            df, feature_cols=["a", "b", "c"], target_col="risk_flag"
        )
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == 100
