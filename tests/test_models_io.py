"""Тесты сохранения/загрузки моделей и работы тренера (ml_core/models.py)."""

import os
import tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml_core.models import ModelTrainer


class TestModelIO:
    def test_save_and_load_model(self):
        """Проверка цикла сохранения и загрузки модели."""
        trainer = ModelTrainer()

        # Обучаем простую модель
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression().fit(X, y)

        # Создаем временную директорию
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.models_dir = tmpdir

            # Сохраняем
            meta = {"accuracy": 1.0}
            path, metadata = trainer.save_model(model, "LR_test", meta, ["f1", "f2"])

            assert os.path.exists(path)
            assert os.path.exists(path.replace(".pkl", "_meta.json"))

            # Загружаем по полному пути
            loaded_model = trainer.load_model(model_path=path)
            assert loaded_model is not None

            # Предсказываем (проверяем, что модель живая)
            preds = loaded_model.predict([[1.5, 2.5]])
            assert preds[0] == 0

    def test_load_latest_model_by_name(self):
        """Загрузка последней модели по имени."""
        trainer = ModelTrainer()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model = LogisticRegression().fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.models_dir = tmpdir
            trainer.save_model(model, "LR_latest", {"acc": 0.99}, ["f1", "f2"])

            # Пытаемся загрузить по имени
            loaded = trainer.load_model(model_name="LR_latest")
            assert loaded is not None
