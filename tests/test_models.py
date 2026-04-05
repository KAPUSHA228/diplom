"""Тесты для ml_core/models.py"""
import pytest
import pandas as pd
import numpy as np
import os
import shutil
from ml_core.models import ModelTrainer


@pytest.fixture
def trainer(tmp_path):
    """ModelTrainer с временной директорией."""
    return ModelTrainer(models_dir=str(tmp_path / "models"))


@pytest.fixture
def train_data(rng=np.random.RandomState(42)):
    """Простые train данные."""
    n = 200
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    y = (X["f1"] + X["f2"] > 0).astype(int)
    return X, y


class TestModelTrainerCrossValidate:
    """Тесты cross_validate."""

    def test_returns_dict_with_model_names(self, trainer, train_data):
        X, y = train_data
        result = trainer.cross_validate(X, y, cv_folds=3)
        assert "LR" in result
        assert "RF" in result
        assert "XGB" in result

    def test_returns_mean_and_std(self, trainer, train_data):
        X, y = train_data
        result = trainer.cross_validate(X, y, cv_folds=3)
        for name in ["LR", "RF", "XGB"]:
            assert "mean" in result[name]
            assert "std" in result[name]

    def test_mean_is_between_0_and_1(self, trainer, train_data):
        X, y = train_data
        result = trainer.cross_validate(X, y, cv_folds=3)
        for name in ["LR", "RF", "XGB"]:
            assert 0 <= result[name]["mean"] <= 1


class TestModelTrainerTrainBest:
    """Тесты train_best_model."""

    def test_returns_model_name_and_metrics(self, trainer, train_data):
        X, y = train_data
        X_test = X.iloc[:40]
        y_test = y.iloc[:40]
        X_train = X.iloc[40:]
        y_train = y.iloc[40:]
        model, name, metrics = trainer.train_best_model(
            X_train, y_train, X_test, y_test
        )
        assert name in ["LR", "RF", "XGB"]
        assert "f1" in metrics.get("test", {})
        assert "roc_auc" in metrics.get("test", {})


class TestModelTrainerTuneXGBoost:
    """Тесты tune_xgboost."""

    def test_returns_estimator_params_score(self, trainer, train_data):
        X, y = train_data
        best_model, best_params, best_score = trainer.tune_xgboost(
            X, y, n_iter=3, cv_folds=2
        )
        assert best_params is not None
        assert 0 <= best_score <= 1


class TestModelTrainerSaveLoad:
    """Тесты save_model / load_model."""

    def test_save_creates_files(self, trainer, train_data):
        X, y = train_data
        model, name, metrics = trainer.train_best_model(X, y)
        path, meta = trainer.save_model(model, name, metrics, list(X.columns))
        assert os.path.exists(path)
        assert path.endswith(".pkl")

    def test_load_returns_model(self, trainer, train_data):
        X, y = train_data
        model, name, metrics = trainer.train_best_model(X, y)
        path, meta = trainer.save_model(model, name, metrics, list(X.columns))
        loaded = trainer.load_model(model_path=path)
        assert loaded is not None
        # Проверка что модель работает
        preds = loaded.predict(X.iloc[:5])
        assert len(preds) == 5
