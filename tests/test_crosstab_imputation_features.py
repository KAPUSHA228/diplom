"""Тесты кросс-таблиц, импутации, экспериментов и генерации признаков."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
from ml_core.imputation import handle_missing_values, detect_outliers
from ml_core.experiment_tracker import ExperimentTracker
from ml_core.features import (
    add_composite_features,
    select_features_for_model,
    create_feature_combinations,
    build_composite_score,
)

matplotlib.use("Agg")

# ========== Кросс-таблицы ==========

from ml_core.crosstab import create_crosstab, create_multi_crosstab, simple_crosstab


class TestCrosstabValidation:
    """Валидация входных данных кросс-таблицы.

    Кросс-таблица должна отклонять некорректные входные данные:
    - отсутствующие колонки
    - числовые колонки с большим числом уникальных значений
    """

    def test_raises_when_row_variable_missing(self):
        """Row variable отсутствует в DataFrame → ValueError."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="не найдена"):
            create_crosstab(df, "nonexistent", "b")

    def test_raises_when_column_variable_missing(self):
        """Column variable отсутствует в DataFrame → ValueError."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="не найдена"):
            create_crosstab(df, "a", "nonexistent")

    def test_raises_when_numeric_too_many_unique(self):
        """Числовая колонка с >20 уникальными → ValueError.

        Кросс-таблица предназначена для категориальных данных.
        Числовые с большим числом уникальных нужно предварительно
        группировать (pd.cut / pd.qcut).
        """
        df = pd.DataFrame(
            {
                "id": range(100),
                "flag": [0, 1] * 50,
            }
        )
        with pytest.raises(ValueError, match="числовой"):
            create_crosstab(df, "id", "flag")


class TestCrosstabAggregation:
    """Агрегация значений вместо подсчёта частот.

    Помимо стандартного подсчёта частот (pd.crosstab),
    поддерживается агрегация произвольных значений (pd.pivot_table).
    """

    def test_pivot_table_with_mean(self):
        """Агрегация по среднему значению (values + aggfunc='mean')."""
        df = pd.DataFrame(
            {
                "gender": ["M", "F", "M", "F", "M", "F"],
                "group": ["A", "A", "B", "B", "A", "B"],
                "score": [80, 90, 70, 85, 75, 95],
            }
        )
        result = create_crosstab(df, "gender", "group", values="score", aggfunc="mean")
        assert "table" in result
        # С values хи-квадрат не применим
        assert result.get("chi2_test") is None

    def test_normalize_skips_chi2(self):
        """Нормализованная таблица — хи-квадрат не считается.

        Хи-квадрат работает только с абсолютными частотами.
        """
        df = pd.DataFrame(
            {
                "gender": ["M", "F", "M", "F", "M", "F"],
                "flag": [0, 0, 1, 0, 1, 1],
            }
        )
        result = create_crosstab(df, "gender", "flag", normalize="index")
        assert "table" in result
        assert result.get("chi2_test") is None


class TestMultiCrosstab:
    """Мульти-кросстаб: анализ нескольких переменных за один вызов."""

    def test_skips_missing_variables(self):
        """Отсутствующие переменные молча пропускаются."""
        df = pd.DataFrame(
            {
                "gender": ["M", "F", "M", "F"],
                "flag": [0, 0, 1, 1],
            }
        )
        result = create_multi_crosstab(df, ["gender", "nonexistent", "also_missing"], "flag")
        assert "gender" in result
        assert "nonexistent" not in result
        assert "also_missing" not in result

    def test_catches_errors_per_variable(self):
        """Ошибка в отдельной переменной не ломает весь анализ."""
        df = pd.DataFrame(
            {
                "numeric_id": range(100),
                "flag": [0, 1] * 50,
            }
        )
        result = create_multi_crosstab(df, ["numeric_id"], "flag")
        assert "numeric_id" in result
        assert "error" in result["numeric_id"]


class TestSimpleCrosstab:
    """Упрощённая кросс-таблица — только таблица + хи-квадрат, без графиков."""

    def test_returns_error_when_variables_missing(self):
        """Обе переменные отсутствуют → error."""
        df = pd.DataFrame({"a": [1, 2]})
        result = simple_crosstab(df, "x", "y")
        assert "error" in result


# ========== Импутация и обработка выбросов ==========


class TestImputationStrategies:
    """Тестирование стратегий обработки пропусков."""

    def test_auto_strategy_drops_high_missing_columns(self):
        """Auto-стратегия: >30% пропусков → удаление колонки."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, np.nan, np.nan, np.nan],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result, report = handle_missing_values(df, strategy="auto")
        assert "a" not in result.columns or report["missing_after"] < report["missing_before"]

    def test_no_missing_values_returns_unchanged(self):
        """Нет пропусков → возвращаем как есть."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result, report = handle_missing_values(df, strategy="auto")
        assert report["missing_before"] == 0
        assert report["missing_after"] == 0


class TestOutlierDetection:
    """Тестирование обнаружения выбросов."""

    def test_zscore_detects_extreme_outlier(self):
        """Z-score метод находит экстремальный выброс."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
        outliers = detect_outliers(df, method="zscore")
        assert "a" in outliers
        assert outliers["a"]["n_outliers"] >= 1

    def test_iqr_no_outliers_in_uniform_data(self):
        """IQR метод: равномерные данные без выбросов."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        outliers = detect_outliers(df, method="iqr")
        assert outliers["a"]["n_outliers"] == 0


# ========== Управление экспериментами ==========


class TestExperimentLifecycle:
    """Полный цикл управления экспериментами: сохранение → список → загрузка → удаление."""

    def test_list_experiments(self, tmp_path):
        """Список экспериментов с ограничением."""
        tracker = ExperimentTracker(storage_dir=str(tmp_path))
        for i in range(3):
            tracker.save_experiment(f"exp_{i}", {"metrics": {"f1": 0.5 + i * 0.1}})
        result = tracker.list_experiments()
        assert len(result) == 3

    def test_save_and_load_experiment(self, tmp_path):
        """Сохранение и загрузка эксперимента."""
        tracker = ExperimentTracker(storage_dir=str(tmp_path))
        exp_id = tracker.save_experiment("test", {"metrics": {"f1": 0.9}})
        loaded = tracker.load_experiment(exp_id)
        assert loaded is not None

    def test_delete_experiment(self, tmp_path):
        """Удаление эксперимента."""
        tracker = ExperimentTracker(storage_dir=str(tmp_path))
        exp_id = tracker.save_experiment("to_del", {"metrics": {}})
        result = tracker.delete_experiment(exp_id)
        assert result is True

    def test_delete_nonexistent_experiment(self, tmp_path):
        """Удаление несуществующего эксперимента."""
        tracker = ExperimentTracker(storage_dir=str(tmp_path))
        result = tracker.delete_experiment("nonexistent")
        assert result is False


# ========== Генерация признаков ==========


class TestCompositeFeatures:
    """Тестирование композитных признаков.

    Автоматически создаваемые признаки:
    - trend_grades, grade_stability (успеваемость)
    - cognitive_load (когнитивная нагрузка)
    - overall_satisfaction, psychological_wellbeing (удовлетворённость)
    - academic_activity (академическая активность)
    """

    def test_no_matching_columns_skips_composites(self):
        """Нет колонок для композитов → данные без изменений."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = add_composite_features(df)
        assert "trend_grades" not in result.columns


class TestFeatureCombinations:
    """Комбинации признаков.

    Автоматическое создание новых признаков из существующих:
    - num + num: sum, diff, ratio, product
    - num + text: numeric × text_length
    - text + text: конкатенация
    """

    def test_numeric_combinations(self):
        """Комбинации числовых признаков."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = create_feature_combinations(df)
        new_cols = set(result.columns) - set(df.columns)
        assert len(new_cols) > 0

    def test_composite_score_returns_tuple(self):
        """build_composite_score возвращает (df, score_name)."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = build_composite_score(df, feature_weights={"a": 1.0}, score_name="test")
        assert isinstance(result, tuple)
        result_df, name = result
        assert name == "test"
        assert "test" in result_df.columns


class TestFeatureSelection:
    """Отбор признаков для модели."""

    def test_select_features_default(self):
        """Отбор признаков — метод по умолчанию."""
        rng = np.random.RandomState(42)
        n = 80
        df = pd.DataFrame(
            {
                "a": rng.rand(n),
                "b": rng.rand(n),
                "c": rng.rand(n),
                "target": (rng.rand(n) > 0.5).astype(int),
            }
        )
        X = df[["a", "b", "c"]]
        y = df["target"]
        features = select_features_for_model(X, y, top_n=2)
        assert len(features) >= 1
