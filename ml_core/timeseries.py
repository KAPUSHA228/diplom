import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, Any


class TimeSeriesAnalyzer:
    """
    Основной класс для анализа временных рядов студентов.
    """

    def __init__(self, df: pd.DataFrame, student_id_col: str = "student_id"):
        self.df = df.copy()
        self.student_id_col = student_id_col
        self._validate_data()

    def _validate_data(self):
        required = [self.student_id_col]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    def analyze_student(
        self, student_id: str, value_col: str, time_col: str = "semester", min_points: int = 3
    ) -> Dict[str, Any]:
        """
        Полный анализ траектории одного студента.
        """
        student_df = self.df[self.df[self.student_id_col] == student_id].sort_values(time_col)

        if len(student_df) < min_points:
            return {"status": "error", "message": f"Недостаточно данных ({len(student_df)} < {min_points})"}

        x = np.arange(len(student_df)).reshape(-1, 1)
        y = student_df[value_col].values

        # Линейная регрессия
        model = LinearRegression().fit(x, y)
        trend = model.coef_[0]
        r2 = r2_score(y, model.predict(x))

        # Классификация тренда
        if trend > 0.08:
            status = "improving"
        elif trend < -0.08:
            status = "declining"
        else:
            status = "stable"

        # Визуализация
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=student_df[time_col],
                y=y,
                mode="lines+markers",
                name="Фактические значения",
                line=dict(color="#1f77b4"),
            )
        )

        # Линия тренда
        trend_line = model.predict(x)
        fig.add_trace(
            go.Scatter(
                x=student_df[time_col],
                y=trend_line,
                mode="lines",
                name=f"Тренд (β₁={trend:.3f}, R²={r2:.3f})",
                line=dict(dash="dash", color="red"),
            )
        )

        fig.update_layout(
            title=f"Траектория студента {student_id} — {value_col}",
            xaxis_title=time_col,
            yaxis_title=value_col,
            height=500,
            template="plotly_white",
        )

        return {
            "student_id": student_id,
            "n_points": len(student_df),
            "trend": float(trend),
            "r2": float(r2),
            "status": status,
            "first_value": float(y[0]),
            "last_value": float(y[-1]),
            "figure": fig,
            "values": y.tolist(),
            "times": student_df[time_col].tolist(),
        }

    def detect_negative_dynamics(
        self, value_col: str, time_col: str = "semester", threshold: float = -0.08, min_points: int = 3
    ) -> Dict[str, Any]:
        """
        Выявление студентов с значимой негативной динамикой.
        """
        results = []

        for student_id in self.df[self.student_id_col].unique():
            student_df = self.df[self.df[self.student_id_col] == student_id].sort_values(time_col)

            if len(student_df) < min_points:
                continue

            x = np.arange(len(student_df)).reshape(-1, 1)
            y = student_df[value_col].values

            model = LinearRegression().fit(x, y)
            trend = model.coef_[0]

            results.append(
                {
                    self.student_id_col: student_id,
                    "trend": float(trend),
                    "at_risk": trend < threshold,
                    "n_points": len(student_df),
                    "first_value": float(y[0]),
                    "last_value": float(y[-1]),
                    "r2": float(r2_score(y, model.predict(x))),
                }
            )

        results_df = pd.DataFrame(results)

        at_risk_df = results_df[results_df["at_risk"]] if not results_df.empty else pd.DataFrame()

        return {
            "n_students_analyzed": len(results_df),
            "at_risk_count": len(at_risk_df),
            "risk_percentage": round(len(at_risk_df) / len(results_df) * 100, 2) if len(results_df) > 0 else 0,
            "threshold": threshold,
            "at_risk_students": at_risk_df.to_dict("records"),
            "all_students": results_df.to_dict("records"),
        }

    def forecast_student(
        self, student_id: str, value_col: str, time_col: str = "semester", periods: int = 2
    ) -> Dict[str, Any]:
        """
        Прогнозирование на будущие периоды с помощью линейной регрессии.
        """
        student_df = self.df[self.df[self.student_id_col] == student_id].sort_values(time_col)

        if len(student_df) < 3:
            return {"error": "Недостаточно данных для прогноза (минимум 3 точки)"}

        x = np.arange(len(student_df)).reshape(-1, 1)
        y = student_df[value_col].values

        model = LinearRegression().fit(x, y)

        future_x = np.arange(len(student_df), len(student_df) + periods).reshape(-1, 1)
        predictions = model.predict(future_x)

        # Ограничиваем прогноз разумными пределами (например, оценки 2-5)
        predictions = np.clip(predictions, 2.0, 5.0)

        return {
            "student_id": student_id,
            "future_periods": list(
                range(int(student_df[time_col].max()) + 1, int(student_df[time_col].max()) + periods + 1)
            ),
            "predictions": predictions.tolist(),
            "trend": float(model.coef_[0]),
        }


# ==================== Удобные функции для API ====================


def get_trajectory(df: pd.DataFrame, student_id: str, value_col: str, time_col: str = "semester"):
    analyzer = TimeSeriesAnalyzer(df)
    return analyzer.analyze_student(student_id, value_col, time_col)


def find_negative_dynamics(df: pd.DataFrame, value_col: str, time_col: str = "semester", threshold: float = -0.08):
    analyzer = TimeSeriesAnalyzer(df)
    return analyzer.detect_negative_dynamics(value_col, time_col, threshold)


def forecast_student(df: pd.DataFrame, student_id: str, value_col: str, time_col: str = "semester", periods: int = 2):
    analyzer = TimeSeriesAnalyzer(df)
    return analyzer.forecast_student(student_id, value_col, time_col, periods)
