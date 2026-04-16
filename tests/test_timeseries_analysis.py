"""Тестирование анализа временных рядов и прогнозирования."""

import numpy as np
import pandas as pd
from ml_core.timeseries import (
    forecast_grades_chupep,
    detect_negative_dynamics,
    analyze_cohort_trajectory,
)


class TestTimeSeriesForecasting:
    """Прогнозирование оценок студентов."""

    def test_forecast_with_empty_history(self):
        """Пустая история → нулевые прогнозы."""
        result = forecast_grades_chupep([], periods=3)
        assert len(result) == 3
        assert all(v == 0 for v in result)

    def test_detect_negative_dynamics_missing_column(self):
        """Отсутствует колонка времени → ошибка."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = detect_negative_dynamics(df, time_col="nonexistent")
        assert "error" in result or result.get("n_students_analyzed", 0) == 0

    def test_cohort_trajectory_comparison(self):
        """Сравнение траекторий когорт."""
        rng = np.random.RandomState(42)
        records = []
        for year in [2020, 2021]:
            for sem in range(1, 4):
                for _ in range(5):
                    records.append(
                        {
                            "year": year,
                            "semester": sem,
                            "avg_grade": 3.5 + rng.normal(0, 0.3),
                        }
                    )
        df = pd.DataFrame(records)
        result = analyze_cohort_trajectory(df, cohort_col="year", time_col="semester")
        assert "cohort_data" in result
        assert "figure" in result
