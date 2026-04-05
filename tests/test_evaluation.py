"""Тесты для ml_core/evaluation.py"""
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ml_core.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    generate_shap_explanations,
    generate_text_explanation,
    generate_detailed_explanation,
    generate_batch_explanations,
)


@pytest.fixture
def trained_model(rng=np.random.RandomState(42)):
    """Обученная модель RandomForest для тестов SHAP (совместима с shap без masker)."""
    n = 100
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    y = (X["f1"] + 0.5 * X["f2"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y, list(X.columns)


class TestCalculateMetrics:
    """Тесты calculate_metrics."""

    def test_returns_expected_keys(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 1, 0, 1]
        y_proba = [0.1, 0.6, 0.9, 0.4, 0.8]
        result = calculate_metrics(y_true, y_pred, y_proba)
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        assert "roc_auc" in result

    def test_metrics_in_valid_range(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 1, 0, 1]
        y_proba = [0.1, 0.6, 0.9, 0.4, 0.8]
        result = calculate_metrics(y_true, y_pred, y_proba)
        for key in ["f1", "precision", "recall", "roc_auc"]:
            assert 0 <= result[key] <= 1

    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.8, 0.9]
        result = calculate_metrics(y_true, y_pred, y_proba)
        assert result["f1"] == 1.0
        assert result["roc_auc"] == 1.0

    def test_without_proba(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        result = calculate_metrics(y_true, y_pred)
        assert "f1" in result
        assert "roc_auc" not in result


class TestPlotFunctions:
    """Тесты функций визуализации."""

    def test_confusion_matrix_returns_figure(self):
        y_true = [0, 0, 1, 1, 1, 0]
        y_pred = [0, 1, 1, 0, 1, 0]
        fig = plot_confusion_matrix(y_true, y_pred, "TestModel")
        assert hasattr(fig, "layout")

    def test_feature_importance_returns_figure(self, trained_model):
        model, X, y, names = trained_model
        fig = plot_feature_importance(model, names, top_n=3)
        assert hasattr(fig, "layout")


class TestShapExplanations:
    """Тесты SHAP-объяснений."""

    def test_generate_shap_returns_list(self, trained_model):
        model, X, y, names = trained_model
        explanations = generate_shap_explanations(
            model, X, names, threshold=0.5, top_n=3
        )
        assert isinstance(explanations, list)

    def test_explanation_has_required_keys(self, trained_model):
        model, X, y, names = trained_model
        explanations = generate_shap_explanations(
            model, X, names, threshold=0.5, top_n=3
        )
        if len(explanations) > 0:
            exp = explanations[0]
            assert "student_index" in exp
            assert "risk_probability" in exp
            assert "risk_level" in exp
            assert "explanation" in exp

    def test_batch_explanations_respects_top_n(self, trained_model):
        model, X, y, names = trained_model
        explanations = generate_batch_explanations(
            model, X, names, threshold=0.5, top_n=3
        )
        assert len(explanations) <= 3

    def test_detailed_explanation_is_string(self, trained_model):
        model, X, y, names = trained_model
        result = generate_detailed_explanation(
            model, X, student_idx=0, feature_names=names, threshold=0.5
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_explanation_is_string(self, trained_model):
        model, X, y, names = trained_model
        explanations = generate_shap_explanations(model, X, names, top_n=1)
        if explanations:
            text = generate_text_explanation(explanations[0])
            assert isinstance(text, str)
