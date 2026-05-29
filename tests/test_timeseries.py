"""Тесты для ml_core/timeseries.py"""

import pandas as pd
import pytest
from ml_core.timeseries import TimeSeriesAnalyzer, forecast_student


@pytest.fixture
def temporal_df():
    records = []
    for sid in range(5):
        for sem in range(1, 5):
            records.append(
                {
                    "student_id": sid,
                    "semester": sem,
                    "avg_grade": 3.5 + 0.1 * sem + sid * 0.05,
                }
            )
    return pd.DataFrame(records)


class TestAnalyzeStudentTrajectory:
    def test_returns_dict_with_expected_keys(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.analyze_student(
            student_id=0,
            value_col="avg_grade",
            time_col="semester",
            min_points=2,
        )
        assert "trend" in result
        assert "status" in result
        assert "figure" in result

    def test_status_is_one_of_expected(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.analyze_student(
            student_id=0,
            value_col="avg_grade",
            time_col="semester",
            min_points=2,
        )
        assert result["status"] in ["improving", "declining", "stable", "error"]

    def test_trend_is_numeric(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.analyze_student(
            student_id=0,
            value_col="avg_grade",
            time_col="semester",
            min_points=2,
        )
        if "trend" in result:
            assert isinstance(result["trend"], (int, float))

    def test_insufficient_semesters(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.analyze_student(
            student_id=0,
            value_col="avg_grade",
            time_col="semester",
            min_points=100,
        )
        assert result.get("status") == "error" or "error" in result


class TestDetectNegativeDynamics:
    def test_returns_expected_keys(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.detect_negative_dynamics(
            value_col="avg_grade",
            time_col="semester",
            min_points=2,
        )
        assert "n_students_analyzed" in result
        assert "at_risk_students" in result
        assert "risk_percentage" in result

    def test_risk_percentage_in_range(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.detect_negative_dynamics(
            value_col="avg_grade",
            time_col="semester",
            min_points=2,
        )
        assert 0 <= result["risk_percentage"] <= 100

    def test_at_risk_is_list(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.detect_negative_dynamics(
            value_col="avg_grade",
            time_col="semester",
            min_points=2,
        )
        assert isinstance(result["at_risk_students"], list)


class TestForecastGrades:
    def test_returns_future_periods_and_predictions(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.forecast_student(
            student_id=0,
            value_col="avg_grade",
            time_col="semester",
            periods=2,
        )
        assert "future_periods" in result
        assert "predictions" in result
        assert len(result["predictions"]) == 2

    def test_predictions_are_numeric(self, temporal_df):
        analyzer = TimeSeriesAnalyzer(temporal_df)
        result = analyzer.forecast_student(
            student_id=0,
            value_col="avg_grade",
            time_col="semester",
        )
        for pred in result["predictions"]:
            assert isinstance(pred, (int, float))

    def test_forecast_student_helper(self, temporal_df):
        result = forecast_student(
            temporal_df,
            student_id=0,
            value_col="avg_grade",
            time_col="semester",
            periods=2,
        )
        assert "predictions" in result
        assert len(result["predictions"]) == 2

    def test_forecast_insufficient_data(self):
        df = pd.DataFrame(
            {
                "student_id": [1, 1],
                "semester": [1, 2],
                "avg_grade": [3.0, 3.5],
            }
        )
        analyzer = TimeSeriesAnalyzer(df)
        result = analyzer.forecast_student(
            student_id=1,
            value_col="avg_grade",
            time_col="semester",
        )
        assert "error" in result


class TestCreateTemporalFeatures:
    def test_analyzer_requires_student_id_column(self):
        df = pd.DataFrame({"semester": [1, 2], "avg_grade": [3.0, 3.5]})
        with pytest.raises(ValueError, match="обязательные"):
            TimeSeriesAnalyzer(df)

    def test_detect_negative_dynamics_missing_time_column(self):
        df = pd.DataFrame(
            {
                "student_id": [1, 1, 1],
                "avg_grade": [3.0, 3.5, 4.0],
            }
        )
        analyzer = TimeSeriesAnalyzer(df)
        with pytest.raises(KeyError):
            analyzer.detect_negative_dynamics(
                value_col="avg_grade",
                time_col="nonexistent",
            )
