"""Тесты для ml_core/timeseries.py"""
import pytest
import pandas as pd
import numpy as np
from ml_core.timeseries import (
    analyze_student_trajectory,
    detect_negative_dynamics,
    forecast_grades,
    create_temporal_features,
)


class TestAnalyzeStudentTrajectory:
    """Тесты analyze_student_trajectory."""

    def test_returns_dict_with_expected_keys(self, sample_temporal_data):
        result = analyze_student_trajectory(
            sample_temporal_data,
            student_id=0,
            time_col="semester",
            value_col="avg_grade",
            min_semesters=2,
        )
        assert "trend" in result
        assert "status" in result
        assert "figure" in result

    def test_status_is_one_of_expected(self, sample_temporal_data):
        result = analyze_student_trajectory(
            sample_temporal_data,
            student_id=0,
            time_col="semester",
            value_col="avg_grade",
        )
        assert result["status"] in ["improving", "declining", "stable"]

    def test_trend_is_numeric(self, sample_temporal_data):
        result = analyze_student_trajectory(
            sample_temporal_data, student_id=0,
            time_col="semester", value_col="avg_grade",
        )
        assert isinstance(result["trend"], (int, float))

    def test_insufficient_semesters(self, sample_temporal_data):
        result = analyze_student_trajectory(
            sample_temporal_data,
            student_id=0,
            time_col="semester",
            value_col="avg_grade",
            min_semesters=100,  # заведомо больше доступных
        )
        assert "error" in result or result["status"] == "insufficient_data"


class TestDetectNegativeDynamics:
    """Тесты detect_negative_dynamics."""

    def test_returns_expected_keys(self, sample_temporal_data):
        result = detect_negative_dynamics(
            sample_temporal_data,
            student_id_col="student_id",
            time_col="semester",
            value_col="avg_grade",
        )
        assert "n_students_analyzed" in result
        assert "at_risk_students" in result
        assert "risk_percentage" in result

    def test_risk_percentage_in_range(self, sample_temporal_data):
        result = detect_negative_dynamics(
            sample_temporal_data,
            student_id_col="student_id",
            time_col="semester",
            value_col="avg_grade",
        )
        assert 0 <= result["risk_percentage"] <= 100

    def test_at_risk_is_dataframe(self, sample_temporal_data):
        result = detect_negative_dynamics(
            sample_temporal_data,
            student_id_col="student_id",
            time_col="semester",
            value_col="avg_grade",
        )
        assert isinstance(result["at_risk_students"], pd.DataFrame)


class TestForecastGrades:
    """Тесты forecast_grades (DataFrame-версия)."""

    def test_returns_future_semesters_and_predictions(self, sample_temporal_data):
        result = forecast_grades(
            sample_temporal_data,
            student_id=0,
            time_col="semester",
            value_col="avg_grade",
            future_semesters=2,
        )
        assert "future_semesters" in result
        assert "predictions" in result
        assert len(result["future_semesters"]) == 2
        assert len(result["predictions"]) == 2

    def test_predictions_are_numeric(self, sample_temporal_data):
        result = forecast_grades(
            sample_temporal_data, student_id=0,
            time_col="semester", value_col="avg_grade",
        )
        for pred in result["predictions"]:
            assert isinstance(pred, (int, float))


class TestCreateTemporalFeatures:
    """Тесты create_temporal_features."""

    def test_creates_lag_and_diff_columns(self, sample_temporal_data):
        result = create_temporal_features(
            sample_temporal_data,
            time_col="semester",
            student_id_col="student_id",
        )
        # Должны появиться lag/diff колонки
        col_names = " ".join(result.columns).lower()
        has_lag = "lag" in col_names
        has_diff = "diff" in col_names
        assert has_lag or has_diff

    def test_same_number_of_rows(self, sample_temporal_data):
        result = create_temporal_features(
            sample_temporal_data,
            time_col="semester",
            student_id_col="student_id",
        )
        assert len(result) == len(sample_temporal_data)
