"""Тестирование обучения и оценки моделей ML."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml_core.models import ModelTrainer


class TestModelTrainingAndEvaluation:
    """Тестирование обучения и оценки моделей."""

    def test_load_model_by_path(self, tmp_path):
        """Загрузка модели по полному пути."""
        trainer = ModelTrainer(models_dir=str(tmp_path))
        model = trainer.models["LR"]
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])
        model.fit(X, y)
        path, meta = trainer.save_model(model, "LR", {"test": {"f1": 0.8}}, ["a"])
        loaded = trainer.load_model(model_path=path)
        assert loaded is not None

    def test_get_best_model_no_saved_models(self, tmp_path):
        """Нет сохранённых моделей → None."""
        trainer = ModelTrainer(models_dir=str(tmp_path))
        model, name, meta = trainer.get_best_model()
        assert model is None
        assert name is None

    def test_train_and_evaluate_all_models(self):
        """Обучение и оценка всех трёх моделей."""
        rng = np.random.RandomState(42)
        X_train = pd.DataFrame({"a": rng.rand(80), "b": rng.rand(80)})
        X_test = pd.DataFrame({"a": rng.rand(20), "b": rng.rand(20)})
        y_train = (rng.rand(80) > 0.5).astype(int)
        y_test = (rng.rand(20) > 0.5).astype(int)
        trainer = ModelTrainer()
        results, best = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        assert "LR" in results
        assert "RF" in results
        assert "XGB" in results
        assert best is not None

    def test_cross_validate_custom_models(self):
        """Кросс-валидация произвольного набора моделей."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"a": rng.rand(100), "b": rng.rand(100)})
        y = (rng.rand(100) > 0.5).astype(int)
        models = {"LR": LogisticRegression(max_iter=200)}
        trainer = ModelTrainer()
        cv_results = trainer.cross_validate_models(X, y, models)
        assert "LR" in cv_results
        assert "mean" in cv_results["LR"]
