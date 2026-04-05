"""
Общие фикстуры для всех тестов ml_core.
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Отключаем GUI-бэкенд для headless-тестов


@pytest.fixture
def rng():
    """Генератор случайных чисел с фиксированным seed."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_students(rng):
    """
    Минимальный датасет: 100 студентов со всеми основными признаками.
    Используется в большинстве unit-тестов.
    """
    n = 100
    return pd.DataFrame({
        "student_id": range(n),
        "avg_grade": rng.uniform(2.0, 5.0, n),
        "grade_std": rng.uniform(0.1, 1.0, n),
        "min_grade": rng.uniform(2.0, 3.5, n),
        "max_grade": rng.uniform(3.5, 5.0, n),
        "n_courses": rng.randint(3, 10, n),
        "avg_brs": rng.uniform(40, 100, n),
        "attendance_rate": rng.uniform(0.4, 1.0, n),
        "satisfaction_score": rng.uniform(1.0, 5.0, n),
        "engagement_score": rng.uniform(1.0, 5.0, n),
        "workload_perception": rng.uniform(1.0, 5.0, n),
        "stress_level": rng.uniform(0.0, 10.0, n),
        "motivation_score": rng.uniform(0.0, 10.0, n),
        "anxiety_score": rng.uniform(0.0, 10.0, n),
        "n_essays": rng.randint(0, 5, n),
        "avg_essay_grade": rng.uniform(2.0, 5.0, n),
        "risk_flag": (rng.random(n) > 0.7).astype(int),
    })


@pytest.fixture
def sample_students_with_nans(rng):
    """Датасет с пропусками — для тестов импутации."""
    n = 100
    df = pd.DataFrame({
        "student_id": range(n),
        "avg_grade": rng.uniform(2.0, 5.0, n),
        "stress_level": rng.uniform(0.0, 10.0, n),
        "satisfaction_score": rng.uniform(1.0, 5.0, n),
        "risk_flag": (rng.random(n) > 0.7).astype(int),
    })
    # Добавляем пропуски: 10% avg_grade, 5% stress, 20% satisfaction
    mask_grade = rng.random(n) < 0.10
    mask_stress = rng.random(n) < 0.05
    mask_satisf = rng.random(n) < 0.20
    df.loc[mask_grade, "avg_grade"] = np.nan
    df.loc[mask_stress, "stress_level"] = np.nan
    df.loc[mask_satisf, "satisfaction_score"] = np.nan
    return df


@pytest.fixture
def sample_temporal_data(rng):
    """Данные с временной динамикой (несколько семестров)."""
    records = []
    for sid in range(20):
        for sem in range(1, 5):
            records.append({
                "student_id": sid,
                "semester": sem,
                "avg_grade": 3.5 + rng.normal(0, 0.3) - 0.05 * sem,
                "n_courses": rng.randint(3, 7),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_categorical_data(rng):
    """Данные для кросс-таблиц (категориальные переменные)."""
    n = 200
    return pd.DataFrame({
        "gender": rng.choice(["M", "F"], n),
        "year": rng.choice([1, 2, 3, 4], n),
        "risk_flag": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "cluster": rng.choice([0, 1, 2], n),
        "satisfaction_group": rng.choice(
            ["Низкий", "Средний", "Высокий"], n
        ),
    })


@pytest.fixture
def sample_text_data():
    """Данные с текстовым столбцом."""
    return pd.DataFrame({
        "student_id": [1, 2, 3],
        "essay_text": [
            "This is a short essay about machine learning.",
            "A much longer and more detailed essay discussing "
            "the applications of deep learning in natural language "
            "processing and computer vision with extensive references.",
            "Brief.",
        ],
    })
