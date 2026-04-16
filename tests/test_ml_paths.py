"""Дополнительные тесты для покрытия ветвей анализа, оценки и моделей."""

import pandas as pd
import numpy as np
import plotly
from sklearn.ensemble import RandomForestClassifier

from ml_core.analysis import cluster_students, correlation_analysis, plot_clusters_pca
from ml_core.evaluation import generate_shap_explanations, generate_text_explanation
from ml_core.models import ModelTrainer
from ml_core.features import build_composite_score


class TestAnalysisBranches:
    """Покрывает ml_core/analysis.py (строки 37-67, 146-199)"""

    def test_correlation_high_threshold(self, sample_students):
        """Проверка корреляции с жестким порогом (строка фильтрации)."""
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        result = correlation_analysis(sample_students, features, "risk_flag", corr_threshold=0.9)
        assert isinstance(result, dict)
        # При высоком пороге strong_matrix должна быть почти пустой
        strong_vals = result["strong_matrix"].dropna(how="all").dropna(axis=1, how="all")
        # Диагональ всегда 1.0, остальные должны быть отфильтрованы или NaN
        non_diag = strong_vals.values[~np.eye(len(strong_vals), dtype=bool)]
        assert len(non_diag[~np.isnan(non_diag)]) == 0 or all(abs(v) >= 0.9 for v in non_diag[~np.isnan(non_diag)])

    def test_cluster_students_returns_correct_shape(self, sample_students):
        """Проверка формы и типов возврата кластеризации."""
        features = ["avg_grade", "stress_level"]
        labels, kmeans, scaler = cluster_students(sample_students[features], n_clusters=4)
        assert len(labels) == len(sample_students)
        assert len(set(labels)) == 4
        assert scaler is not None

    def test_plot_clusters_pca_returns_figure(self, sample_students):
        """Проверка создания PCA графика (покрывает 185-199)."""
        features = ["avg_grade", "stress_level", "satisfaction_score"]
        sample_students["cluster"] = np.random.choice([0, 1, 2], len(sample_students))
        fig = plot_clusters_pca(sample_students, sample_students["cluster"], features)
        assert isinstance(fig, plotly.graph_objs.Figure)
        assert fig.layout.title.text is not None


class TestEvaluationBranches:
    """Покрывает ml_core/evaluation.py (строки 252-267, 382-425)"""

    def test_generate_text_explanation(self):
        """Проверка генерации текстового отчета SHAP."""
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
        assert "avg_grade" in text
        # Функция выводит стрелки ↑ и ↓
        assert "↑" in text
        assert "↓" in text

    def test_generate_shap_explanations_with_rf(self):
        """Интеграционный тест SHAP для RandomForest (покрывает 382+)."""
        X = pd.DataFrame({"f1": np.random.rand(100), "f2": np.random.rand(100)})
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

        explanations = generate_shap_explanations(model, X, feature_names=["f1", "f2"], top_n=3)
        assert isinstance(explanations, list)
        assert len(explanations) == min(len(X), 15)  # Ограничение в 15 студентов
        assert "risk_probability" in explanations[0]


class TestModelTrainerBranches:
    """Покрывает ml_core/models.py (строки 157, 172-232)"""

    def test_cross_validate_returns_dict(self, sample_students):
        """Проверка возврата метрик CV."""
        trainer = ModelTrainer()
        features = ["avg_grade", "stress_level"]
        X = sample_students[features].fillna(0)
        y = sample_students["risk_flag"]

        cv_results = trainer.cross_validate(X, y, cv_folds=3, scoring="f1")
        assert "LR" in cv_results
        assert "mean" in cv_results["LR"]
        assert "scores" in cv_results["LR"]
        assert len(cv_results["LR"]["scores"]) == 3

    def test_train_best_model_with_custom_scoring(self, sample_students):
        """Обучение с кастомной метрикой (покрывает ветки scoring и test)."""
        trainer = ModelTrainer()
        features = ["avg_grade", "stress_level"]
        X = sample_students[features].fillna(0)
        y = sample_students["risk_flag"]

        # Чтобы покрыть ветку "test", нужно передать тестовую выборку
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model, name, metrics = trainer.train_best_model(X_train, y_train, X_test, y_test, scoring="roc_auc")
        assert model is not None
        assert "cv_results" in metrics
        assert "test" in metrics


class TestFeaturesEdgeCases:
    """Покрывает ml_core/features.py (строки 161, 302-341)"""

    def test_composite_score_no_normalize(self):
        """Композитный скор без нормализации (строгая математика)."""
        df = pd.DataFrame({"a": [10, 20], "b": [2, 4]})
        weights = {"a": 2.0, "b": 1.0}
        df_res, name = build_composite_score(df, weights, "score", normalize=False)
        # 10*2 + 2*1 = 22
        assert df_res.iloc[0]["score"] == 22.0
        assert df_res.iloc[1]["score"] == 44.0

    def test_composite_score_with_missing_col(self):
        """Игнорирование несуществующих колонок в весах."""
        df = pd.DataFrame({"a": [10], "b": [2]})
        weights = {"a": 1.0, "non_existent": 5.0}
        df_res, name = build_composite_score(df, weights, "score")
        assert "score" in df_res.columns
