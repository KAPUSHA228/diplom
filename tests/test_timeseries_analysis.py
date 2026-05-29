"""Тестирование анализа временных рядов и прогнозирования."""

import pandas as pd
import pytest
from ml_core.timeseries import TimeSeriesAnalyzer, forecast_student


class TestTimeSeriesForecasting:
    def test_forecast_with_insufficient_history(self):
        df = pd.DataFrame(
            {
                "student_id": [1, 1],
                "semester": [1, 2],
                "avg_grade": [3.0, 3.5],
            }
        )
        result = forecast_student(df, student_id=1, value_col="avg_grade", periods=3)
        assert "error" in result

    def test_detect_negative_dynamics_missing_column(self):
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

    def test_cohort_like_multi_student_analysis(self):
        records = []
        for year in [2020, 2021]:
            for sem in range(1, 4):
                for sid in range(3):
                    records.append(
                        {
                            "student_id": f"{year}_{sid}",
                            "semester": sem,
                            "avg_grade": 3.5 + sem * 0.1,
                        }
                    )
        df = pd.DataFrame(records)
        analyzer = TimeSeriesAnalyzer(df)
        result = analyzer.detect_negative_dynamics(
            value_col="avg_grade",
            time_col="semester",
            min_points=2,
        )
        assert result["n_students_analyzed"] > 0
