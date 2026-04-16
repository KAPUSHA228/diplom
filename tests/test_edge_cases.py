"""
Edge-case тесты: граничные и экстремальные условия,
при которых код чаще всего ломается.
"""

import pytest
import pandas as pd
import numpy as np

from ml_core.features import (
    add_composite_features,
    get_base_features,
    preprocess_data,
)
from ml_core.analysis import (
    correlation_analysis,
    cluster_students,
)
from ml_core.imputation import handle_missing_values, detect_outliers
from ml_core.evaluation import calculate_metrics
from ml_core.drift_detector import DataDriftDetector


class TestEmptyAndTinyData:
    """Тесты на пустых и минимальных данных."""

    def test_one_student(self):
        """Один студент — не падает."""
        df = pd.DataFrame(
            {
                "avg_grade": [4.0],
                "stress_level": [5.0],
                "risk_flag": [0],
            }
        )
        features = get_base_features(df)
        X, y = preprocess_data(df, features or ["avg_grade", "stress_level"], "risk_flag", use_smote=False)
        assert len(X) == 1

    def test_two_students_one_each_class(self):
        """Два студента, по одному в каждом классе."""
        df = pd.DataFrame(
            {
                "avg_grade": [2.0, 5.0],
                "risk_flag": [0, 1],
            }
        )
        X, y = preprocess_data(df, ["avg_grade"], "risk_flag", use_smote=False)
        assert len(X) == 2
        assert set(y) == {0, 1}

    def test_all_same_class(self):
        """Все студенты одного класса — SMOTE отключается."""
        df = pd.DataFrame(
            {
                "avg_grade": [3.0, 4.0, 5.0, 3.5],
                "risk_flag": [0, 0, 0, 0],
            }
        )
        X, y = preprocess_data(df, ["avg_grade"], "risk_flag", use_smote=True)
        assert len(X) == 4  # SMOTE должен отключиться

    def test_all_nan_column(self):
        """Вся колонка NaN — handle_missing_values справляется."""
        df = pd.DataFrame(
            {
                "a": [np.nan] * 10,
                "b": list(range(10)),
            }
        )
        result, report = handle_missing_values(df, strategy="auto")
        # Колонка 'a' должна быть удалена или заполнена
        assert result.isna().sum().sum() == 0


class TestExtremeValues:
    """Тесты на экстремальных значениях."""

    def test_very_large_values(self):
        """Очень большие числа — корреляция не падает."""
        n = 50
        df = pd.DataFrame(
            {
                "a": np.random.randn(n) * 1e9,
                "b": np.random.randn(n) * 1e9,
                "risk_flag": np.random.choice([0, 1], n),
            }
        )
        corr = correlation_analysis(df, ["a", "b"], "risk_flag")
        matrix = corr["full_matrix"]
        assert (matrix.values >= -1.0).all()
        assert (matrix.values <= 1.0).all()

    def test_negative_values(self):
        """Отрицательные значения — кластеризация не падает."""
        df = pd.DataFrame(
            {
                "a": np.random.randn(50) * 1000,
                "b": np.random.randn(50) * 1000,
            }
        )
        labels, _, _ = cluster_students(df, n_clusters=3)
        assert len(labels) == 50
        assert set(labels) == {0, 1, 2}

    def test_constant_column(self):
        """Константная колонка — detect_outliers обрабатывает."""
        df = pd.DataFrame(
            {
                "constant": [5.0] * 50,
                "varying": list(range(50)),
            }
        )
        result = detect_outliers(df, columns=["constant", "varying"])
        assert "constant" in result

    def test_single_unique_value_in_target(self):
        """Target = одна константа — calculate_metrics не падает."""
        y_true = [0] * 20
        y_pred = [0] * 20
        y_proba = [0.1] * 20
        result = calculate_metrics(y_true, y_pred, y_proba)
        assert "f1" in result


class TestAllMissingData:
    """Тесты на полностью пропущенных данных."""

    def test_all_nan_dataframe(self):
        """Весь DataFrame из NaN — handle_missing_values."""
        df = pd.DataFrame(
            {
                "a": [np.nan] * 30,
                "b": [np.nan] * 30,
                "c": [np.nan] * 30,
            }
        )
        result, report = handle_missing_values(df, strategy="auto")
        # Либо удалены, либо заполнены
        assert isinstance(result, pd.DataFrame)

    def test_99_percent_missing(self):
        """99% данных — NaN."""
        n = 100
        df = pd.DataFrame(
            {
                "a": [np.nan] * n,
                "b": list(range(n)),
                "risk_flag": [0] * n,
            }
        )
        df.loc[0, "a"] = 1.0  # единственное значение
        result, report = handle_missing_values(df, strategy="auto")
        assert result.isna().sum().sum() == 0


class TestDegenerateCases:
    """Вырожденные случаи."""

    def test_cluster_more_than_students(self):
        """n_clusters > n_students — KMeans ValueError (ограничение sklearn)."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="n_samples"):
            cluster_students(df, n_clusters=5)

    def test_identical_rows(self):
        """Все строки одинаковые — кластеризация."""
        df = pd.DataFrame(
            {
                "a": [5.0] * 30,
                "b": [5.0] * 30,
            }
        )
        labels, _, _ = cluster_students(df, n_clusters=3)
        assert len(labels) == 30

    def test_single_feature_for_clustering(self):
        """Один признак для кластеризации."""
        df = pd.DataFrame(
            {
                "a": list(range(30)),
            }
        )
        labels, _, _ = cluster_students(df, n_clusters=3)
        assert len(labels) == 30
        assert len(set(labels)) == 3

    def test_drift_identical_data(self):
        """Дрейф на идентичных данных — нет дрейфа."""
        ref = pd.DataFrame({"a": np.random.randn(100)})
        detector = DataDriftDetector(ref, threshold=0.05)
        report = detector.detect_drift(ref.copy())
        assert report["drift_percentage"] == 0.0


class TestRegressionBugs:
    """Регрессионные тесты — проверям что старые баги не вернулись."""

    def test_no_duplicate_plot_feature_importance(self):
        """plot_feature_importance возвращает Plotly фигуру, а не None."""
        from ml_core.evaluation import plot_feature_importance
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = pd.DataFrame({f"f{i}": rng.random(50) for i in range(5)})
        y = rng.choice([0, 1], 50)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        fig = plot_feature_importance(model, list(X.columns))
        assert fig is not None
        assert hasattr(fig, "layout")  # Plotly figure

    def test_forecast_grades_dataframe_version(self):
        """forecast_grades принимает (df, student_id, ...) и возвращает dict."""
        from ml_core.timeseries import forecast_grades

        records = []
        for sem in range(1, 6):
            records.append(
                {
                    "student_id": 0,
                    "semester": sem,
                    "avg_grade": 4.0 - 0.1 * sem,
                }
            )
        df = pd.DataFrame(records)
        result = forecast_grades(df, student_id=0, value_col="avg_grade", future_semesters=2)
        assert "future_semesters" in result
        assert "predictions" in result
        assert len(result["predictions"]) == 2

    def test_composite_features_no_infinite(self):
        """Композитные признаки не содержат inf."""
        df = pd.DataFrame(
            {
                "avg_grade": [0.0, 0.0],  # может вызвать деление на 0
                "grade_std": [0.0, 0.0],
            }
        )
        result = add_composite_features(df)
        for col in result.columns:
            if col in ["avg_grade", "grade_std"]:
                continue
            assert not result[col].apply(np.isinf).any(), f"inf in {col}"

    def test_smote_does_crash_on_impossible(self):
        """SMOTE не падает когда класс слишком мал."""
        df = pd.DataFrame(
            {
                "f1": np.random.randn(15),
                "f2": np.random.randn(15),
                "risk_flag": [0] * 13 + [1] * 2,  # 2 примера минорного класса
            }
        )
        X, y = preprocess_data(df, ["f1", "f2"], "risk_flag", use_smote=True)
        # Не должно упасть — SMOTE автоматически отключится
        assert len(y) >= 15
