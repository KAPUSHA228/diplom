"""
Модуль для работы с данными
ВАША ЗОНА ОТВЕТСТВЕННОСТИ
"""
import pandas as pd
import numpy as np
import streamlit as st
import os
import datetime
from sklearn.model_selection import train_test_split

try:
    import seaborn as sns
except ImportError:
    sns = None


def save_synthetic_data(df, filename=None, directory='analysis_data/synthetic'):
    """
    Сохраняет синтетические данные в CSV файл

    Args:
        df: DataFrame с данными
        filename: имя файла (если None, генерируется автоматически)
        directory: директория для сохранения

    Returns:
        str: путь к сохраненному файлу
    """
    # Создаем директорию если её нет
    os.makedirs(directory, exist_ok=True)

    # Генерируем имя файла если не указано
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_students_{timestamp}.csv"

    # Полный путь к файлу
    filepath = os.path.join(directory, filename)

    # Сохраняем
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"✅ Данные сохранены: {filepath}")

    return filepath


def save_data_for_monitoring(df, feature_cols, semester="spring_2026", directory='analysis_data/monitoring'):
    """
    Сохраняет данные специально для мониторинга дрейфа

    Args:
        df: DataFrame с данными
        feature_cols: список признаков для сохранения
        semester: название семестра
        directory: директория для сохранения
    """
    os.makedirs(directory, exist_ok=True)

    # Сохраняем только признаки (без target)
    monitoring_data = df[feature_cols].copy()
    filepath = os.path.join(directory, f"monitoring_{semester}.csv")
    monitoring_data.to_csv(filepath, index=False, encoding='utf-8')

    print(f"✅ Данные для мониторинга сохранены: {filepath}")
    return filepath


# ---- Загрузка данных согласно ТЗ ----
@st.cache_data(ttl=3600)
def load_data(path=None, grades_path=None, questionnaires_path=None,
              psych_tests_path=None, essays_path=None, generate_synthetic=True):
    """
    Загружает данные согласно ТЗ из 4 источников:
    1. Данные об успеваемости студентов
    2. Данные по итогам анкетирования студентов
    3. Данные по итогам психологического тестирования студентов
    4. Эссе как учебная работа
    """
    from pathlib import Path

    if generate_synthetic or all(p is None for p in [grades_path, questionnaires_path, psych_tests_path, essays_path]):
        return _generate_synthetic_data_tz()

    # Загрузка данных об успеваемости
    if grades_path and Path(grades_path).exists():
        grades_df = pd.read_csv(grades_path)
    else:
        grades_df = _generate_grades_data()

    # Загрузка данных анкетирования
    if questionnaires_path and Path(questionnaires_path).exists():
        questionnaires_df = pd.read_csv(questionnaires_path)
    else:
        questionnaires_df = _generate_questionnaires_data()

    # Загрузка данных психологического тестирования
    if psych_tests_path and Path(psych_tests_path).exists():
        psych_tests_df = pd.read_csv(psych_tests_path)
    else:
        psych_tests_df = _generate_psych_tests_data()

    # Загрузка данных эссе
    if essays_path and Path(essays_path).exists():
        essays_df = pd.read_csv(essays_path)
    else:
        essays_df = _generate_essays_data()

    # Агрегация данных по студентам
    return _aggregate_student_features(grades_df, questionnaires_df, psych_tests_df, essays_df)


def _generate_synthetic_data_tz():
    """Генерирует синтетические данные, соответствующие структуре ТЗ."""
    np.random.seed(42)
    n_students = 1200

    # Генерируем данные об успеваемости
    grades_data = []
    for student_id in range(n_students):
        n_courses = np.random.randint(5, 10)
        for course in range(n_courses):
            grades_data.append({
                'student_id': student_id,
                'course_name': f'Course_{course}',
                'grade': np.clip(np.random.normal(3.5, 0.8), 2, 5),
                'semester': np.random.choice([1, 2]),
                'brs_score': np.random.uniform(0, 100)
            })
    grades_df = pd.DataFrame(grades_data)

    # Генерируем данные анкетирования
    questionnaires_df = pd.DataFrame({
        'student_id': range(n_students),
        'satisfaction_score': np.random.uniform(1, 5, n_students),
        'engagement_score': np.random.uniform(1, 5, n_students),
        'workload_perception': np.random.uniform(1, 5, n_students)
    })

    # Генерируем данные психологического тестирования
    psych_tests_df = pd.DataFrame({
        'student_id': range(n_students),
        'stress_level': np.random.uniform(0, 10, n_students),
        'motivation_score': np.random.uniform(0, 10, n_students),
        'anxiety_score': np.random.uniform(0, 10, n_students)
    })

    # Генерируем данные эссе
    essays_df = pd.DataFrame({
        'student_id': range(n_students),
        'n_essays': np.random.randint(1, 5, n_students),
        'avg_essay_grade': np.clip(np.random.normal(3.5, 0.7, n_students), 2, 5)
    })

    # Агрегируем
    student_features = _aggregate_student_features(
        grades_df, questionnaires_df, psych_tests_df, essays_df
    )
    student_features['semester'] = np.random.choice([1, 2], size=len(student_features), p=[0.5, 0.5])

    # Добавляем целевую переменную (риск)
    risk_prob = (
            (student_features['avg_grade'] < 3.0).astype(int) * 0.4 +
            (student_features['stress_level'] > 7).astype(int) * 0.3 +
            (student_features['satisfaction_score'] < 2.5).astype(int) * 0.3
    )
    student_features['risk_flag'] = (risk_prob > 0.5).astype(int)

    return student_features


def _generate_grades_data():
    """Генерирует синтетические данные об успеваемости."""
    np.random.seed(42)
    n_students = 1200
    grades_data = []
    for student_id in range(n_students):
        n_courses = np.random.randint(5, 10)
        for course in range(n_courses):
            grades_data.append({
                'student_id': student_id,
                'course_name': f'Course_{course}',
                'grade': np.clip(np.random.normal(3.5, 0.8), 2, 5),
                'semester': np.random.choice([1, 2]),
                'brs_score': np.random.uniform(0, 100)
            })
    return pd.DataFrame(grades_data)


def _generate_questionnaires_data():
    """Генерирует синтетические данные анкетирования."""
    np.random.seed(42)
    n_students = 1200
    return pd.DataFrame({
        'student_id': range(n_students),
        'satisfaction_score': np.random.uniform(1, 5, n_students),
        'engagement_score': np.random.uniform(1, 5, n_students),
        'workload_perception': np.random.uniform(1, 5, n_students)
    })


def _generate_psych_tests_data():
    """Генерирует синтетические данные психологического тестирования."""
    np.random.seed(42)
    n_students = 1200
    return pd.DataFrame({
        'student_id': range(n_students),
        'stress_level': np.random.uniform(0, 10, n_students),
        'motivation_score': np.random.uniform(0, 10, n_students),
        'anxiety_score': np.random.uniform(0, 10, n_students)
    })


def _generate_essays_data():
    """Генерирует синтетические данные эссе."""
    np.random.seed(42)
    n_students = 1200
    return pd.DataFrame({
        'student_id': range(n_students),
        'n_essays': np.random.randint(1, 5, n_students),
        'avg_essay_grade': np.clip(np.random.normal(3.5, 0.7, n_students), 2, 5)
    })


def _aggregate_student_features(grades_df, questionnaires_df, psych_tests_df, essays_df):
    """
    Агрегирует данные из 4 источников в единый датасет по студентам.
    Соответствует пункту ТЗ 6.1 и 6.2 - группировка и агрегирование.
    """
    # Агрегация данных об успеваемости
    grades_agg = grades_df.groupby('student_id').agg({
        'grade': ['mean', 'std', 'min', 'max', 'count'],
        'brs_score': 'mean'
    }).reset_index()
    grades_agg.columns = ['student_id', 'avg_grade', 'grade_std', 'min_grade', 'max_grade', 'n_courses', 'avg_brs']

    # Тренд успеваемости
    if 'semester' in grades_df.columns:
        grades_by_sem = grades_df.groupby(['student_id', 'semester'])['grade'].mean().reset_index()
        grades_pivot = grades_by_sem.pivot(index='student_id', columns='semester', values='grade').reset_index()
        if 1 in grades_pivot.columns and 2 in grades_pivot.columns:
            grades_pivot['trend_grades'] = grades_pivot[2] - grades_pivot[1]
            grades_agg = grades_agg.merge(grades_pivot[['student_id', 'trend_grades']], on='student_id', how='left')
        else:
            grades_agg['trend_grades'] = 0
    else:
        grades_agg['trend_grades'] = 0

    # Объединение всех данных
    result = grades_agg.merge(questionnaires_df, on='student_id', how='left')
    result = result.merge(psych_tests_df, on='student_id', how='left')
    result = result.merge(essays_df, on='student_id', how='left')

    # Заполнение пропусков
    result = result.fillna(result.median(numeric_only=True))

    return result


def load_from_csv(file):
    """
    Загрузка данных из CSV файла
    """
    return pd.read_csv(file)


def generate_temporal_data(n_students=500, n_semesters=3):
    """
    Генерирует данные с временной динамикой (несколько семестров на студента).
    """
    np.random.seed(42)

    records = []

    for student_id in range(n_students):
        # Базовые характеристики студента (меняются медленно)
        base_stress = np.random.uniform(2, 8)
        base_grades = np.random.uniform(2.5, 4.8)
        base_motivation = np.random.uniform(3, 8)

        for semester in range(1, n_semesters + 1):
            # Динамика: стресс может расти/падать, оценки могут меняться
            # Добавляем случайный тренд
            stress_trend = np.random.uniform(-0.5, 0.5) * (semester - 1)
            grades_trend = np.random.uniform(-0.3, 0.3) * (semester - 1)

            stress = np.clip(base_stress + stress_trend + np.random.normal(0, 0.5), 1, 10)
            grades = np.clip(base_grades + grades_trend + np.random.normal(0, 0.3), 2, 5)
            motivation = np.clip(base_motivation + np.random.normal(0, 0.5), 1, 10)

            # Риск: зависит от динамики
            if semester > 1:
                # Если оценки падают, риск выше
                risk_prob = max(0, (5 - grades) / 5 + (stress / 15))
            else:
                risk_prob = 0.3

            risk_flag = 1 if np.random.random() < risk_prob else 0

            records.append({
                'student_id': student_id,
                'semester': semester,
                'avg_grade': grades,
                'stress_level': stress,
                'motivation_score': motivation,
                'risk_flag': risk_flag
            })

    df = pd.DataFrame(records)
    return df


def prepare_data_for_training(df, feature_cols, target_col='risk_flag', test_size=0.2):
    """
    Подготовка данных для обучения
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Обработка пропусков
    X = X.fillna(X.median())

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
