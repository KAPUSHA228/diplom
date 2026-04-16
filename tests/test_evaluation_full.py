"""Тесты для модуля оценки моделей (ml_core/evaluation.py)"""

import pandas as pd
import numpy as np
import plotly
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml_core.evaluation import (
    generate_shap_explanations,
    generate_text_explanation,
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
)


class TestEvaluationMetrics:
    """Проверка генерации графиков, SHAP и текстовых отчетов."""

    def setup_method(self):
        np.random.seed(42)
        self.X = pd.DataFrame({"f1": np.random.rand(100), "f2": np.random.rand(100), "f3": np.random.rand(100)})
        self.y = np.random.randint(0, 2, 100)
        self.feature_names = ["f1", "f2", "f3"]

    def _train_model(self, ModelClass):
        m = ModelClass(random_state=42)
        m.fit(self.X, self.y)
        return m

    def test_shap_explanations_logistic_regression(self):
        """Покрытие LinearExplainer."""
        model = self._train_model(LogisticRegression)
        explanations = generate_shap_explanations(model, self.X, self.feature_names, top_n=3)
        assert isinstance(explanations, list)
        assert len(explanations) > 0
        assert "risk_probability" in explanations[0]

    def test_shap_explanations_random_forest(self):
        """Покрытие TreeExplainer для RF."""
        model = self._train_model(RandomForestClassifier)
        explanations = generate_shap_explanations(model, self.X, self.feature_names, top_n=3)
        assert isinstance(explanations, list)
        assert len(explanations) > 0

    def test_shap_explanations_xgboost(self):
        """Покрытие TreeExplainer для XGB."""
        model = self._train_model(XGBClassifier)
        explanations = generate_shap_explanations(model, self.X, self.feature_names, top_n=3)
        assert isinstance(explanations, list)
        assert len(explanations) > 0

    def test_generate_text_explanation_high_risk(self):
        """Генерация текста для высокого риска."""
        mock_exp = {
            "risk_probability": 0.85,
            "risk_level": "high",
            "top_features": [
                {"feature": "stress_level", "shap_value": 0.4, "effect": "повышает риск"},
                {"feature": "avg_grade", "shap_value": -0.2, "effect": "снижает риск"},
            ],
        }
        text = generate_text_explanation(mock_exp)
        assert "85.0%" in text
        assert "stress_level" in text
        # Проверяем стрелки
        assert "↑" in text
        assert "↓" in text

    def test_calculate_metrics(self):
        """Прямой вызов функции метрик."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2])

        metrics = calculate_metrics(y_true, y_pred, y_proba)
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_plot_confusion_matrix(self):
        fig = plot_confusion_matrix(np.array([0, 1]), np.array([0, 1]), "LR")
        assert isinstance(fig, plotly.graph_objs.Figure)

    def test_plot_roc_curves(self):
        model = self._train_model(LogisticRegression)
        fig = plot_roc_curves({"LR": model}, self.X, self.y)
        assert isinstance(fig, plotly.graph_objs.Figure)

    def test_plot_feature_importance_rf(self):
        model = self._train_model(RandomForestClassifier)
        fig = plot_feature_importance(model, self.feature_names)
        assert isinstance(fig, plotly.graph_objs.Figure)

    def test_plot_feature_importance_lr(self):
        model = self._train_model(LogisticRegression)
        fig = plot_feature_importance(model, self.feature_names)
        assert isinstance(fig, plotly.graph_objs.Figure)
