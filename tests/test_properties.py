"""
Property-based тесты: проверяют ИНВАРИАНТЫ — свойства, которые должны
выполняться для ЛЮБЫХ допустимых входных данных.
"""
import pytest
import pandas as pd
import numpy as np

from ml_core.features import (
    add_composite_features,
    get_base_features,
    preprocess_data,
    select_features_for_model,
    create_feature_combinations,
)
from ml_core.analysis import (
    correlation_analysis,
    cluster_students,
    analyze_cluster_profiles,
)
from ml_core.imputation import handle_missing_values, detect_outliers
from ml_core.evaluation import calculate_metrics
from ml_core.models import ModelTrainer
from ml_core.crosstab import create_crosstab, simple_crosstab


class TestFeatureProperties:
    """Свойства модуля features."""

    def test_composite_never_raises(self, rng):
        """add_composite_features НЕ должен падать ни на каком DataFrame."""
        for _ in range(10):
            n = rng.randint(1, 50)
            df = pd.DataFrame({
                f"col_{i}": rng.random(n) * 100
                for i in range(rng.randint(1, 10))
            })
            result = add_composite_features(df)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == n

    def test_base_features_are_numeric(self, rng):
        """get_base_features возвращает ТОЛЬКО существующие числовые колонки."""
        for _ in range(10):
            n = rng.randint(5, 30)
            df = pd.DataFrame({
                "num_a": rng.random(n),
                "num_b": rng.random(n),
                "str_col": ["x"] * n,
                "risk_flag": rng.choice([0, 1], n),
            })
            features = get_base_features(df)
            for f in features:
                assert f in df.columns
                assert df[f].dtype.kind in "biufc"

    def test_preprocess_never_nan(self, rng):
        """preprocess_data НЕ возвращает NaN в X."""
        for _ in range(10):
            n = rng.randint(20, 100)
            df = pd.DataFrame({
                "f1": rng.random(n),
                "f2": rng.random(n),
                "risk_flag": rng.choice([0, 1], n),
            })
            # Добавляем случайные NaN
            mask = rng.random(df.shape) < 0.1
            df.iloc[mask[:, 0], 0] = np.nan
            X, y = preprocess_data(df, ["f1", "f2"], "risk_flag", use_smote=False)
            assert not X.isna().any().any()

    def test_select_features_reduces_dimension(self):
        """select_features_for_model НЕ увеличивает число признаков."""
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            **{f"f{i}": rng.random(n) for i in range(10)},
            "risk_flag": rng.choice([0, 1], n),
        })
        X = df[[f"f{i}" for i in range(10)]]
        y = df["risk_flag"]
        X_sel, names = select_features_for_model(X, y, top_n=7, final_n=5)
        assert X_sel.shape[1] <= X.shape[1]
        assert len(names) <= X.shape[1]


class TestAnalysisProperties:
    """Свойства модуля analysis."""

    def test_correlation_symmetric(self, rng):
        """Корреляционная матрица СИММЕТРИЧНА."""
        for _ in range(10):
            n = rng.randint(20, 100)
            k = rng.randint(3, 8)
            df = pd.DataFrame({
                **{f"f{i}": rng.random(n) for i in range(k)},
                "target": rng.choice([0, 1], n),
            })
            cols = [f"f{i}" for i in range(k)]
            corr = correlation_analysis(df, cols, "target")
            # corr[i,j] == corr[j,i]
            diff = corr.values - corr.values.T
            assert np.abs(diff).max() < 1e-10

    def test_correlation_bounded(self, rng):
        """Все значения корреляции ∈ [-1, 1]."""
        n = 50
        df = pd.DataFrame({
            "a": rng.random(n),
            "b": rng.random(n),
            "target": rng.choice([0, 1], n),
        })
        corr = correlation_analysis(df, ["a", "b"], "target")
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()

    def test_cluster_labels_are_contiguous_integers(self, rng):
        """Метки кластеров — целые числа от 0 до n_clusters-1."""
        for n_clusters in [2, 3, 4, 5]:
            n = rng.randint(20, 100)
            df = pd.DataFrame({
                "f1": rng.random(n),
                "f2": rng.random(n),
                "f3": rng.random(n),
            })
            labels, _, _ = cluster_students(df, n_clusters=n_clusters)
            unique = set(labels)
            expected = set(range(n_clusters))
            assert unique == expected

    def test_cluster_sizes_sum_to_total(self, rng):
        """Сумма размеров кластеров = общее число студентов."""
        n = rng.randint(50, 200)
        df = pd.DataFrame({
            "f1": rng.random(n),
            "f2": rng.random(n),
        })
        labels, _, _ = cluster_students(df, n_clusters=3)
        df["cluster"] = labels
        profiles = analyze_cluster_profiles(df, ["f1", "f2"])
        assert profiles["size"].sum() == n


class TestImputationProperties:
    """Свойства модуля imputation."""

    def test_handle_missing_removes_all_nan(self, rng):
        """После handle_missing_values НЕ остаётся NaN."""
        for _ in range(10):
            n = rng.randint(10, 100)
            df = pd.DataFrame({
                "a": rng.random(n),
                "b": rng.random(n),
            })
            # Случайные NaN
            for col in ["a", "b"]:
                mask = rng.random(n) < 0.3
                df.loc[mask, col] = np.nan
            result, _ = handle_missing_values(df, strategy="auto")
            assert result.isna().sum().sum() == 0

    def test_outlier_percentage_valid(self, rng):
        """Процент выбросов ∈ [0, 100]."""
        for _ in range(10):
            n = rng.randint(10, 200)
            df = pd.DataFrame({
                "a": rng.random(n) * rng.choice([1, 10, 100]),
            })
            # Иногда добавляем экстремальные значения
            if rng.random() < 0.5:
                df.loc[0, "a"] = 1e6
            result = detect_outliers(df, columns=["a"])
            pct = result["a"]["percentage"]
            assert 0 <= pct <= 100


class TestMetricsProperties:
    """Свойства модуля evaluation."""

    def test_metrics_always_in_unit_interval(self, rng):
        """Все метрики ∈ [0, 1] для любых y_true/y_pred."""
        for _ in range(20):
            n = rng.randint(5, 100)
            y_true = rng.choice([0, 1], n)
            y_pred = rng.choice([0, 1], n)
            y_proba = rng.random(n)
            result = calculate_metrics(y_true, y_pred, y_proba)
            for key in ["f1", "precision", "recall", "roc_auc"]:
                if key in result:
                    assert 0 <= result[key] <= 1, f"{key} = {result[key]}"

    def test_perfect_prediction_yields_ones(self):
        """Идеальные предсказания → все метрики = 1.0."""
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        y_proba = [0.1, 0.2, 0.8, 0.9, 0.95]
        result = calculate_metrics(y_true, y_pred, y_proba)
        assert result["f1"] == 1.0
        assert result["roc_auc"] == 1.0


class TestCrosstabProperties:
    """Свойства модуля crosstab."""

    def test_crosstab_row_sums_equal_total(self, rng):
        """Сумма строк кросс-таблицы = общее число наблюдений."""
        n = 200
        df = pd.DataFrame({
            "a": rng.choice(["X", "Y"], n),
            "b": rng.choice([0, 1], n),
        })
        result = create_crosstab(df, "a", "b")
        table = result["table"]
        # Для crosstab без normalize сумма всех ячеек = n
        assert abs(table.values.sum() - n) < 1

    def test_simple_crosstab_no_error(self, rng):
        """simple_crosstab НЕ падает на любых категориальных данных."""
        for _ in range(10):
            n = rng.randint(10, 100)
            df = pd.DataFrame({
                "a": rng.choice(list("ABC"), n),
                "b": rng.choice([0, 1, 2], n),
            })
            result = simple_crosstab(df, "a", "b")
            assert "table" in result
            assert "chi2_test" in result
