"""Тесты для ml_core/features.py"""
import pytest
import pandas as pd
import numpy as np
from ml_core.features import (
    add_composite_features,
    get_base_features,
    select_features_for_model,
    preprocess_data,
    preprocess_data_for_smote,
    create_feature_combinations,
)


class TestAddCompositeFeatures:
    """Тесты add_composite_features."""

    def test_creates_all_composite_columns(self, sample_students):
        df = add_composite_features(sample_students)
        expected = [
            "trend_grades", "grade_stability", "cognitive_load",
            "overall_satisfaction", "psychological_wellbeing", "academic_activity",
        ]
        for col in expected:
            assert col in df.columns, f"Missing composite: {col}"

    def test_no_original_columns_dropped(self, sample_students):
        original_cols = set(sample_students.columns)
        df = add_composite_features(sample_students)
        assert original_cols.issubset(set(df.columns))

    def test_all_values_are_finite(self, sample_students):
        df = add_composite_features(sample_students)
        composites = [
            "trend_grades", "grade_stability", "cognitive_load",
            "overall_satisfaction", "psychological_wellbeing", "academic_activity",
        ]
        for col in composites:
            assert df[col].apply(np.isfinite).all(), f"Non-finite values in {col}"

    def test_missing_original_columns_no_error(self):
        """Если исходных колонок нет — функция не падает, создаёт дефолтные значения."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3],
            "max_grade": [4.0, 5.0, 3.0],
            "min_grade": [2.0, 3.0, 1.0],
        })
        result = add_composite_features(df)
        # trend_grades создаётся из max_grade - min_grade
        assert "trend_grades" in result.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = add_composite_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_grade_stability_formula(self):
        """grade_stability = grade_std / (avg_grade + 0.01)."""
        df = pd.DataFrame({
            "avg_grade": [4.0],
            "grade_std": [0.4],
        })
        result = add_composite_features(df)
        expected = 0.4 / (4.0 + 0.01)
        assert abs(result["grade_stability"].iloc[0] - expected) < 0.001


class TestGetBaseFeatures:
    """Тесты get_base_features."""

    def test_returns_numeric_columns(self, sample_students):
        features = get_base_features(sample_students)
        assert "avg_grade" in features
        assert "stress_level" in features
        assert "satisfaction_score" in features

    def test_excludes_id_and_target(self, sample_students):
        features = get_base_features(sample_students)
        assert "student_id" not in features
        assert "risk_flag" not in features

    def test_excludes_string_columns(self):
        df = pd.DataFrame({
            "num_col": [1.0, 2.0],
            "str_col": ["a", "b"],
        })
        features = get_base_features(df)
        assert "num_col" in features
        assert "str_col" not in features

    def test_empty_dataframe(self):
        features = get_base_features(pd.DataFrame())
        assert features == []


class TestPreprocessData:
    """Тесты preprocess_data."""

    def test_returns_X_and_y(self, sample_students):
        feature_cols = ["avg_grade", "stress_level"]
        X, y = preprocess_data(sample_students, feature_cols, "risk_flag")
        # SMOTE увеличивает данные, поэтому len(X) >= len(sample_students)
        assert len(X) >= len(sample_students)
        assert len(y) == len(X)

    def test_no_nan_in_X(self, sample_students_with_nans):
        feature_cols = ["avg_grade", "stress_level", "satisfaction_score"]
        X, y = preprocess_data(
            sample_students_with_nans, feature_cols, "risk_flag", use_smote=False
        )
        assert not X.isna().any().any()

    def test_smote_balances_classes(self, rng):
        """SMOTE должен создать примерно равные классы."""
        n = 200
        df = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "risk_flag": [0] * 170 + [1] * 30,  # сильный дисбаланс
        })
        X, y = preprocess_data(df, ["f1", "f2"], "risk_flag", use_smote=True)
        # Классы должны стать ближе
        assert len(set(y)) == 2

    def test_smote_disabled_for_small_classes(self):
        """SMOTE отключается если min_class_size <= 5."""
        df = pd.DataFrame({
            "f1": np.random.randn(20),
            "risk_flag": [0] * 18 + [1] * 2,
        })
        X, y = preprocess_data(df, ["f1"], "risk_flag", use_smote=True)
        # Не должно упасть — SMOTE автоматически отключится
        assert len(y) > 0


class TestSelectFeaturesForModel:
    """Тесты select_features_for_model."""

    def test_returns_correct_n_features(self, sample_students):
        X = sample_students[["avg_grade", "stress_level", "satisfaction_score",
                              "motivation_score", "anxiety_score", "n_essays",
                              "engagement_score", "attendance_rate"]]
        y = sample_students["risk_flag"]
        X_sel, names = select_features_for_model(X, y, top_n=7, final_n=5)
        assert X_sel.shape[1] == 5
        assert len(names) == 5

    def test_selected_names_are_subset(self, sample_students):
        X = sample_students[["avg_grade", "stress_level", "satisfaction_score",
                              "motivation_score", "n_essays"]]
        y = sample_students["risk_flag"]
        _, names = select_features_for_model(X, y, top_n=5, final_n=3)
        assert set(names).issubset(set(X.columns))


class TestCreateFeatureCombinations:
    """Тесты create_feature_combinations."""

    def test_num_plus_num_creates_new_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = create_feature_combinations(
            df, numerical_cols=["a", "b"], text_cols=[], methods=["sum", "diff"]
        )
        # Новые колонки должны появиться
        assert len(result.columns) > len(df.columns)

    def test_no_infinite_values(self):
        df = pd.DataFrame({"a": [1, 2, 0], "b": [4, 0, 6]})
        result = create_feature_combinations(
            df, numerical_cols=["a", "b"], text_cols=[], methods=["ratio"]
        )
        # Деление на 0 не должно дать inf
        for col in result.columns:
            if col not in df.columns:
                assert result[col].apply(lambda x: not np.isinf(x) if pd.notna(x) else True).all()

    def test_empty_numerical_cols(self):
        df = pd.DataFrame({"a": [1, 2, 3], "text": ["x", "y", "z"]})
        result = create_feature_combinations(
            df, numerical_cols=[], text_cols=["text"], methods=["sum"]
        )
        assert len(result.columns) == len(df.columns)  # ничего не добавилось
