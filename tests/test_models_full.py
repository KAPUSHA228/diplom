"""Тесты для модуля обучения моделей (ml_core/models.py)"""

import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_core.models import ModelTrainer


class TestModelPipeline:
    """Проверка работы тренера: CV, обучение, сохранение/загрузка."""

    def setup_method(self):
        self.trainer = ModelTrainer()
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Имена признаков для логирования
        self.feature_names = [f"f{i}" for i in range(5)]

    def test_cross_validate_f1(self):
        """CV с метрикой F1."""
        df_X = pd.DataFrame(self.X_train, columns=self.feature_names)
        cv_res = self.trainer.cross_validate(df_X, self.y_train, cv_folds=3, scoring="f1")
        assert "LR" in cv_res
        assert "RF" in cv_res
        assert "XGB" in cv_res
        assert len(cv_res["LR"]["scores"]) == 3

    def test_cross_validate_roc_auc(self):
        """CV с метрикой ROC-AUC."""
        df_X = pd.DataFrame(self.X_train, columns=self.feature_names)
        cv_res = self.trainer.cross_validate(df_X, self.y_train, cv_folds=3, scoring="roc_auc")
        assert "LR" in cv_res
        assert cv_res["LR"]["mean"] > 0

    def test_train_best_model_with_test(self):
        """Обучение с возвратом метрик теста (покрывает ветку if X_test is not None)."""
        df_X = pd.DataFrame(self.X_train, columns=self.feature_names)
        df_X_test = pd.DataFrame(self.X_test, columns=self.feature_names)

        model, name, metrics = self.trainer.train_best_model(df_X, self.y_train, df_X_test, self.y_test, scoring="f1")

        assert model is not None
        assert "cv_results" in metrics
        assert "test" in metrics
        assert "f1" in metrics["test"]
        assert "roc_auc" in metrics["test"]

    def test_train_best_model_without_test(self):
        """Обучение без теста (ветка else)."""
        df_X = pd.DataFrame(self.X_train, columns=self.feature_names)

        model, name, metrics = self.trainer.train_best_model(df_X, self.y_train, scoring="f1")

        assert model is not None
        assert "cv_results" in metrics
        # Ключа "test" быть не должно
        assert "test" not in metrics

    def test_save_and_load_model(self):
        """Сохранение и загрузка модели (покрывает joblib и json)."""
        model, name, _ = self.trainer.train_best_model(
            pd.DataFrame(self.X_train, columns=self.feature_names), self.y_train
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer.models_dir = tmpdir
            meta = {"f1": 0.99}

            # Сохранение
            path, saved_meta = self.trainer.save_model(model, "LR_test", meta, self.feature_names)

            assert os.path.exists(path)
            assert os.path.exists(path.replace(".pkl", "_meta.json"))

            # Загрузка
            loaded = self.trainer.load_model(model_path=path)
            assert loaded is not None

            preds = loaded.predict(self.X_test)
            assert len(preds) == len(self.X_test)

    def test_load_latest_model_by_name(self):
        """Поиск последней модели по имени."""
        model, name, _ = self.trainer.train_best_model(
            pd.DataFrame(self.X_train, columns=self.feature_names), self.y_train
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer.models_dir = tmpdir
            self.trainer.save_model(model, "RF_latest", {"f1": 0.95}, self.feature_names)

            loaded = self.trainer.load_model(model_name="RF_latest")
            assert loaded is not None
