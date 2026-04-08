"""
Анализ временных рядов для отслеживания динамики студентов
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


def analyze_student_trajectory(df, student_id, time_col="semester", value_col="avg_grade", min_semesters=2):
    """
    Анализирует траекторию одного студента: линейный тренд, статус, Plotly-график.

    Args:
        df: DataFrame с данными по семестрам
        student_id: идентификатор студента
        time_col: колонка времени (по умолчанию 'semester')
        value_col: колонка показателя (по умолчанию 'avg_grade')
        min_semesters: минимум семестров для анализа

    Returns:
        dict: {'trend', 'status', 'first_value', 'last_value', 'figure'} или {'error'}
    """
    df = df.copy()
    student_df = df[df["student_id"] == student_id].sort_values(time_col)

    if len(student_df) < min_semesters:
        return {"error": f"Недостаточно данных (нужно минимум {min_semesters} семестров)"}

    values = student_df[value_col].values
    times = student_df[time_col].values

    # Простой линейный тренд (коэффициент)
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    trend = coeffs[0]

    # Классификация динамики
    if trend > 0.1:
        status = "improving"
    elif trend < -0.1:
        status = "declining"
    else:
        status = "stable"

    # Визуализация
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=values, mode="lines+markers", name="Значения", line=dict(color="blue")))

    # Линия тренда
    trend_line = coeffs[0] * x + coeffs[1]
    fig.add_trace(
        go.Scatter(
            x=times, y=trend_line, mode="lines", name=f"Тренд (коэф: {trend:.3f})", line=dict(dash="dash", color="red")
        )
    )

    fig.update_layout(
        title=f"Траектория студента {student_id}: {value_col}", xaxis_title=time_col, yaxis_title=value_col
    )

    return {
        "student_id": student_id,
        "n_points": len(student_df),
        "trend": trend,
        "status": status,
        "values": values.tolist(),
        "times": times.tolist(),
        "figure": fig,
    }


def analyze_cohort_trajectory(df, cohort_col="year", time_col="semester", value_col="avg_grade"):
    """
    Сравнивает средние траектории когорт (по году, группе и т.д.).

    Args:
        df: DataFrame с данными
        cohort_col: колонка когорты (по умолчанию 'year')
        time_col: колонка времени
        value_col: колонка показателя

    Returns:
        dict: {'cohort_trends': dict, 'figure': Plotly Figure}
    """
    df = df.copy()
    cohorts = df.groupby(cohort_col).mean(numeric_only=True).reset_index()

    fig = px.line(
        cohorts, x=time_col, y=value_col, color=cohort_col, title=f"Сравнение когорт по {value_col}", markers=True
    )

    return {"cohort_data": cohorts, "figure": fig}


def detect_negative_dynamics(
    df, student_id_col="student_id", time_col="semester", value_col="avg_grade", threshold=-0.05
):
    """
    Находит студентов с отрицательной динамикой показателя.

    Args:
        df: DataFrame с данными по семестрам
        student_id_col: имя колонки ID студента
        time_col: колонка времени
        value_col: колонка показателя
        threshold: порог тренда для классификации как "negative"

    Returns:
        dict: {'n_students_analyzed', 'at_risk_students', 'risk_percentage'}
    """
    df = df.copy()
    # Проверяем наличие необходимых колонок
    if time_col not in df.columns:
        return {
            "error": f"Колонка '{time_col}' не найдена в данных",
            "all_students": pd.DataFrame(),
            "at_risk_students": pd.DataFrame(),
            "risk_percentage": 0,
        }

    if value_col not in df.columns:
        return {
            "error": f"Колонка '{value_col}' не найдена в данных",
            "all_students": pd.DataFrame(),
            "at_risk_students": pd.DataFrame(),
            "risk_percentage": 0,
        }

    results = []

    for student_id in df[student_id_col].unique():
        student_df = df[df[student_id_col] == student_id].sort_values(time_col)

        if len(student_df) >= 2:
            x = np.arange(len(student_df))
            values = student_df[value_col].values
            coeffs = np.polyfit(x, values, 1)
            trend = coeffs[0]

            results.append(
                {
                    student_id_col: student_id,
                    "trend": trend,
                    "at_risk": trend < threshold,
                    "n_observations": len(student_df),
                    "first_value": values[0],
                    "last_value": values[-1],
                }
            )

    if not results:
        return {
            "all_students": pd.DataFrame(),
            "at_risk_students": pd.DataFrame(),
            "risk_percentage": 0,
            "n_students_analyzed": 0,
        }

    results_df = pd.DataFrame(results)

    # Создаем колонку at_risk_students для фильтрации
    at_risk = results_df[results_df["at_risk"]] if "at_risk" in results_df.columns else pd.DataFrame()

    return {
        "all_students": results_df,
        "at_risk_students": at_risk,
        "risk_percentage": len(at_risk) / len(results_df) * 100 if len(results_df) > 0 else 0,
        "n_students_analyzed": len(results_df),
        "threshold": threshold,
    }


def forecast_grades_chupep(student_history, periods=2, method="linear"):
    """
    Прогноз оценок на будущие периоды (list-based версия).

    Args:
        student_history: list оценок по семестрам [3.5, 4.0, 3.8, ...]
        periods: число будущих периодов для прогноза
        method: метод ('linear' или 'last_value')

    Returns:
        list[float]: прогнозируемые оценки
    """
    if len(student_history) < 2:
        return [student_history[-1]] * periods if student_history else [0] * periods

    if method == "linear":
        x = np.arange(len(student_history))
        y = np.array(student_history)
        coeffs = np.polyfit(x, y, 1)

        forecasts = []
        for i in range(1, periods + 1):
            forecast = coeffs[0] * (len(student_history) + i) + coeffs[1]
            forecasts.append(max(2, min(5, forecast)))  # оценки от 2 до 5
        return forecasts

    elif method == "last":
        return [student_history[-1]] * periods

    return [student_history[-1]] * periods


def create_temporal_features(df, time_col="semester", student_id_col="student_id"):
    """
    Создаёт временные признаки: lag1, diff, pct_change для всех числовых колонок.

    Args:
        df: DataFrame с данными по семестрам
        time_col: колонка времени
        student_id_col: колонка ID студента

    Returns:
        pd.DataFrame: DataFrame с добавленными временными признаками
    """
    df = df.copy()
    result_df = df.copy()

    # Сортируем по студенту и времени
    result_df = result_df.sort_values([student_id_col, time_col])

    # Лаговые признаки (предыдущие значения)
    result_df[f"{time_col}_lag1"] = result_df.groupby(student_id_col)[time_col].shift(1)

    # Изменение между временными точками
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != student_id_col and col != time_col:
            result_df[f"{col}_change"] = result_df.groupby(student_id_col)[col].diff()
            result_df[f"{col}_pct_change"] = result_df.groupby(student_id_col)[col].pct_change()

    return result_df


def forecast_grades(df, student_id, time_col="semester", value_col="avg_grade", future_semesters=2):
    """
    Прогноз оценок студента на будущие семестры (LinearRegression).

    Args:
        df: DataFrame с данными по семестрам
        student_id: идентификатор студента
        time_col: название колонки времени (семестр)
        value_col: название колонки с оцениваемым показателем
        future_semesters: на сколько семестров вперёд прогнозировать

    Returns:
        dict: future_semesters (список) и predictions (список)
    """
    df = df.copy()
    student_data = df[df["student_id"] == student_id].sort_values(time_col)
    X = student_data[time_col].values.reshape(-1, 1)
    y = student_data[value_col].values

    model = LinearRegression().fit(X, y)
    future_x = np.array([X.max() + i + 1 for i in range(future_semesters)]).reshape(-1, 1)
    pred = model.predict(future_x)

    return {
        "future_semesters": list(range(int(X.max()) + 1, int(X.max()) + future_semesters + 1)),
        "predictions": pred.tolist(),
    }
