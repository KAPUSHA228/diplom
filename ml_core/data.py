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
import seaborn as sns


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
def load_data(category: str = 'grades', n_students: int = 500, generate_two_sets: bool = False):
    """
    Загружает синтетические данные для выбранной категории.

    Parameters:
    -----------
    category : str
        'grades', 'psychology', 'creativity', 'values',
        'personality', 'activities', 'career'
    n_students : int
        Количество студентов
    generate_two_sets : bool
        Если True, возвращает (reference_data, new_data) для дрейфа

    Returns:
    --------
    dict с данными
    """
    return generate_synthetic_data_by_category(category, n_students, generate_two_sets)


def _generate_synthetic_data_tz():
    """
    Генерирует синтетические данные, соответствующие структуре из ТЗ.
    Создаёт единую таблицу со всеми признаками.
    """
    np.random.seed(42)
    n_students = 1200

    # ========== 1. Социодемография ==========
    demographics = pd.DataFrame({
        'student_id': range(n_students),
        'age': np.random.randint(17, 25, n_students),
        'gender': np.random.choice(['М', 'Ж'], n_students, p=[0.45, 0.55]),
        'course': np.random.choice([1, 2, 3, 4], n_students, p=[0.3, 0.3, 0.25, 0.15]),
        'university': 'Примерный университет',
        'field_of_study': np.random.choice(
            ['Социология', 'Психология', 'Экономика', 'Политология', 'Социальная работа'],
            n_students
        )
    })

    # ========== 2. Успеваемость (из ТЗ) ==========
    grades = pd.DataFrame({
        'student_id': range(n_students),
        'avg_grade': np.clip(np.random.normal(3.8, 0.6, n_students), 2, 5),
        'grade_std': np.random.uniform(0.2, 0.8, n_students),
        'min_grade': np.random.choice([2, 3, 4, 5], n_students, p=[0.05, 0.25, 0.5, 0.2]),
        'max_grade': np.random.choice([4, 5], n_students, p=[0.3, 0.7]),
        'n_courses': np.random.randint(5, 12, n_students),
        'avg_brs': np.random.uniform(60, 95, n_students),
        'semester': np.random.choice([1, 2, 3, 4, 5, 6], n_students)
    })

    # ========== 3. Анкетирование (из ТЗ) ==========
    questionnaires = pd.DataFrame({
        'student_id': range(n_students),
        'satisfaction_score': np.random.uniform(1, 5, n_students),
        'engagement_score': np.random.uniform(1, 5, n_students),
        'workload_perception': np.random.uniform(1, 5, n_students)
    })

    # ========== 4. Психологическое тестирование (из ТЗ) ==========
    psych_tests = pd.DataFrame({
        'student_id': range(n_students),
        'stress_level': np.random.uniform(1, 10, n_students),
        'motivation_score': np.random.uniform(1, 10, n_students),
        'anxiety_score': np.random.uniform(1, 10, n_students)
    })

    # ========== 5. Эссе (из ТЗ) ==========
    essays = pd.DataFrame({
        'student_id': range(n_students),
        'n_essays': np.random.randint(1, 6, n_students),
        'avg_essay_grade': np.clip(np.random.normal(3.5, 0.8, n_students), 2, 5)
    })

    # ========== 6. Тест креативности Вильямса ==========
    creativity = pd.DataFrame({
        'student_id': range(n_students),
        'curiosity': np.random.randint(10, 40, n_students),  # Любознательность
        'imagination': np.random.randint(10, 40, n_students),  # Воображение
        'complexity': np.random.randint(10, 40, n_students),  # Сложность
        'risk_taking': np.random.randint(10, 40, n_students),  # Склонность к риску
        'creativity_total': np.random.randint(40, 160, n_students)  # Сумма
    })

    # ========== 7. Ценности Шварца ==========
    schwartz = pd.DataFrame({
        'student_id': range(n_students),
        'security': np.random.uniform(1, 7, n_students),  # Безопасность
        'conformity': np.random.uniform(1, 7, n_students),  # Конформность
        'tradition': np.random.uniform(1, 7, n_students),  # Традиция
        'self_direction': np.random.uniform(1, 7, n_students),  # Самостоятельность
        'stimulation': np.random.uniform(1, 7, n_students),  # Риск–новизна
        'hedonism': np.random.uniform(1, 7, n_students),  # Гедонизм
        'achievement': np.random.uniform(1, 7, n_students),  # Достижение
        'power': np.random.uniform(1, 7, n_students),  # Власть–богатство
        'benevolence': np.random.uniform(1, 7, n_students),  # Благожелательность
        'universalism': np.random.uniform(1, 7, n_students)  # Универсализм
    })

    # ========== 8. Личностные опросы (соц1) ==========
    personality = pd.DataFrame({
        'student_id': range(n_students),
        'teamwork': np.random.choice([1, 2, 3, 4, 5], n_students, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        'leadership': np.random.choice([1, 2, 3, 4, 5], n_students),
        'adaptability': np.random.choice([1, 2, 3, 4, 5], n_students),
        'optimism': np.random.choice([1, 2, 3, 4, 5], n_students)
    })

    # ========== 9. Активность (соц4) ==========
    activities = pd.DataFrame({
        'student_id': range(n_students),
        'research_projects': np.random.binomial(1, 0.3, n_students),
        'sports': np.random.binomial(1, 0.4, n_students),
        'volunteering': np.random.binomial(1, 0.25, n_students),
        'creative_events': np.random.binomial(1, 0.35, n_students),
        'student_council': np.random.binomial(1, 0.15, n_students)
    })
    activities['activity_score'] = activities[
        ['research_projects', 'sports', 'volunteering', 'creative_events', 'student_council']].sum(axis=1)

    # ========== 10. Объединяем всё ==========
    result = demographics
    for df in [grades, questionnaires, psych_tests, essays, creativity, schwartz, personality, activities]:
        result = result.merge(df, on='student_id', how='left')

    # ========== 11. Целевая переменная (риск) ==========
    risk_prob = (
            (result['avg_grade'] < 3.0).astype(int) * 0.35 +
            (result['stress_level'] > 7).astype(int) * 0.25 +
            (result['satisfaction_score'] < 2.5).astype(int) * 0.2 +
            (result['activity_score'] < 1).astype(int) * 0.2
    )
    result['risk_flag'] = (risk_prob > 0.5).astype(int)

    # ========== 12. Добавляем композитные признаки ==========
    from ml_core.features import add_composite_features
    result = add_composite_features(result)

    return result


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


def generate_synthetic_data_by_category(category: str, n_students: int = 500,
                                        generate_two_sets: bool = False) -> dict:
    """
    Генерирует синтетические данные для конкретной категории.

    Parameters:
    -----------
    category : str
        'grades', 'psychology', 'creativity', 'values', 'personality',
        'activities', 'career'
    n_students : int
        Количество студентов
    generate_two_sets : bool
        Если True, возвращает (reference_data, new_data) для дрейфа

    Returns:
    --------
    dict с ключами:
        - 'data': DataFrame с данными
        - 'target': название целевой переменной
        - 'target_values': значения целевой переменной (если generate_two_sets=False)
        - 'reference': DataFrame эталонных данных (если generate_two_sets=True)
        - 'new': DataFrame новых данных (если generate_two_sets=True)
    """
    np.random.seed(42)

    if category == 'grades':
        return _generate_grades_category(n_students, generate_two_sets)
    elif category == 'psychology':
        return _generate_psychology_category(n_students, generate_two_sets)
    elif category == 'creativity':
        return _generate_creativity_category(n_students, generate_two_sets)
    elif category == 'values':
        return _generate_values_category(n_students, generate_two_sets)
    elif category == 'personality':
        return _generate_personality_category(n_students, generate_two_sets)
    elif category == 'activities':
        return _generate_activities_category(n_students, generate_two_sets)
    elif category == 'career':
        return _generate_career_category(n_students, generate_two_sets)
    else:
        # Fallback
        return _generate_grades_category(n_students, generate_two_sets)


def _generate_grades_category(n_students: int, generate_two_sets: bool = False) -> dict:
    """
    Категория: Успеваемость
    Целевая переменная: risk_flag (риск отчисления)
    """
    np.random.seed(42)

    # Базовые признаки
    data = pd.DataFrame({
        'student_id': range(n_students),
        'avg_grade': np.clip(np.random.normal(3.8, 0.6, n_students), 2, 5),
        'grade_std': np.random.uniform(0.2, 0.8, n_students),
        'min_grade': np.random.choice([2, 3, 4, 5], n_students, p=[0.05, 0.25, 0.5, 0.2]),
        'max_grade': np.random.choice([4, 5], n_students, p=[0.3, 0.7]),
        'n_courses': np.random.randint(5, 12, n_students),
        'avg_brs': np.random.uniform(60, 95, n_students),
        'attendance_rate': np.random.uniform(0.6, 1.0, n_students),
        'semester': np.random.choice([1, 2, 3, 4, 5, 6], n_students)
    })

    # Целевая переменная: риск отчисления
    risk_score = (
            (data['avg_grade'] < 3.0).astype(int) * 0.4 +
            (data['attendance_rate'] < 0.7).astype(int) * 0.3 +
            (data['grade_std'] > 0.6).astype(int) * 0.3
    )
    data['risk_flag'] = (risk_score > 0.5).astype(int)

    result = {
        'data': data,
        'target': 'risk_flag',
        'target_values': data['risk_flag']
    }

    if generate_two_sets:
        # Генерируем новые данные с небольшим смещением (дрейф)
        np.random.seed(43)
        new_data = data.copy()
        # Имитируем дрейф: средняя оценка снижается, посещаемость падает
        new_data['avg_grade'] = np.clip(new_data['avg_grade'] - np.random.normal(0.2, 0.1, n_students), 2, 5)
        new_data['attendance_rate'] = np.clip(new_data['attendance_rate'] - np.random.normal(0.1, 0.05, n_students),
                                              0.4, 1.0)
        new_data['risk_flag'] = (
                                        (new_data['avg_grade'] < 3.0).astype(int) * 0.4 +
                                        (new_data['attendance_rate'] < 0.7).astype(int) * 0.3 +
                                        (new_data['grade_std'] > 0.6).astype(int) * 0.3
                                ) > 0.5

        result['reference'] = data
        result['new'] = new_data

    return result


def _generate_psychology_category(n_students: int, generate_two_sets: bool = False) -> dict:
    """
    Категория: Психологическое состояние
    Целевая переменная: burnout_risk (риск выгорания)
    """
    np.random.seed(42)

    data = pd.DataFrame({
        'student_id': range(n_students),
        'stress_level': np.random.uniform(1, 10, n_students),
        'motivation_score': np.random.uniform(1, 10, n_students),
        'anxiety_score': np.random.uniform(1, 10, n_students),
        'sleep_quality': np.random.uniform(1, 5, n_students),
        'social_support': np.random.uniform(1, 5, n_students)
    })

    # Риск выгорания: высокий стресс + низкая мотивация + плохой сон
    burnout_score = (
            (data['stress_level'] > 7).astype(int) * 0.4 +
            (data['motivation_score'] < 4).astype(int) * 0.3 +
            (data['sleep_quality'] < 2).astype(int) * 0.3
    )
    data['burnout_risk'] = (burnout_score > 0.5).astype(int)

    result = {
        'data': data,
        'target': 'burnout_risk',
        'target_values': data['burnout_risk']
    }

    if generate_two_sets:
        np.random.seed(43)
        new_data = data.copy()
        # Имитируем дрейф: стресс растёт, мотивация падает
        new_data['stress_level'] = np.clip(new_data['stress_level'] + np.random.normal(0.5, 0.2, n_students), 1, 10)
        new_data['motivation_score'] = np.clip(new_data['motivation_score'] - np.random.normal(0.3, 0.2, n_students), 1,
                                               10)
        new_data['burnout_risk'] = (
                                           (new_data['stress_level'] > 7).astype(int) * 0.4 +
                                           (new_data['motivation_score'] < 4).astype(int) * 0.3 +
                                           (new_data['sleep_quality'] < 2).astype(int) * 0.3
                                   ) > 0.5

        result['reference'] = data
        result['new'] = new_data

    return result


def _generate_creativity_category(n_students: int, generate_two_sets: bool = False) -> dict:
    """
    Категория: Креативность (тест Вильямса)
    Целевая переменная: high_creativity (высокая креативность)
    """
    np.random.seed(42)

    data = pd.DataFrame({
        'student_id': range(n_students),
        'curiosity': np.random.randint(10, 40, n_students),  # Любознательность
        'imagination': np.random.randint(10, 40, n_students),  # Воображение
        'complexity': np.random.randint(10, 40, n_students),  # Сложность
        'risk_taking': np.random.randint(10, 40, n_students),  # Склонность к риску
        'creativity_total': np.random.randint(40, 160, n_students)
    })

    # Высокая креативность: сумма > 100
    data['high_creativity'] = (data['creativity_total'] > 100).astype(int)

    result = {
        'data': data,
        'target': 'high_creativity',
        'target_values': data['high_creativity']
    }

    if generate_two_sets:
        np.random.seed(43)
        new_data = data.copy()
        # Имитируем дрейф: общая креативность снижается
        new_data['creativity_total'] = np.clip(new_data['creativity_total'] - np.random.normal(15, 10, n_students), 40,
                                               160)
        new_data['high_creativity'] = (new_data['creativity_total'] > 100).astype(int)

        result['reference'] = data
        result['new'] = new_data

    return result


def _generate_values_category(n_students: int, generate_two_sets: bool = False) -> dict:
    """
    Категория: Ценности Шварца
    Целевая переменная: value_profile (тип профиля: 0-консервативный, 1-открытый)
    """
    np.random.seed(42)

    data = pd.DataFrame({
        'student_id': range(n_students),
        'security': np.random.uniform(1, 7, n_students),  # Безопасность
        'conformity': np.random.uniform(1, 7, n_students),  # Конформность
        'tradition': np.random.uniform(1, 7, n_students),  # Традиция
        'self_direction': np.random.uniform(1, 7, n_students),  # Самостоятельность
        'stimulation': np.random.uniform(1, 7, n_students),  # Риск–новизна
        'hedonism': np.random.uniform(1, 7, n_students),  # Гедонизм
        'achievement': np.random.uniform(1, 7, n_students),  # Достижение
        'power': np.random.uniform(1, 7, n_students),  # Власть–богатство
        'benevolence': np.random.uniform(1, 7, n_students),  # Благожелательность
        'universalism': np.random.uniform(1, 7, n_students)  # Универсализм
    })

    # Тип профиля: 0 - консервативный (высокие security, conformity, tradition)
    #              1 - открытый (высокие self_direction, stimulation)
    conservative_score = (data['security'] + data['conformity'] + data['tradition']) / 3
    open_score = (data['self_direction'] + data['stimulation']) / 2
    data['value_profile'] = (open_score > conservative_score).astype(int)

    result = {
        'data': data,
        'target': 'value_profile',
        'target_values': data['value_profile']
    }

    if generate_two_sets:
        np.random.seed(43)
        new_data = data.copy()
        # Имитируем дрейф: смещение в сторону консервативных ценностей
        new_data['security'] = np.clip(new_data['security'] + np.random.normal(0.3, 0.2, n_students), 1, 7)
        new_data['conformity'] = np.clip(new_data['conformity'] + np.random.normal(0.3, 0.2, n_students), 1, 7)
        new_data['self_direction'] = np.clip(new_data['self_direction'] - np.random.normal(0.2, 0.2, n_students), 1, 7)

        conservative_score_new = (new_data['security'] + new_data['conformity'] + new_data['tradition']) / 3
        open_score_new = (new_data['self_direction'] + new_data['stimulation']) / 2
        new_data['value_profile'] = (open_score_new > conservative_score_new).astype(int)

        result['reference'] = data
        result['new'] = new_data

    return result


def _generate_personality_category(n_students: int, generate_two_sets: bool = False) -> dict:
    """
    Категория: Личностные опросы
    Целевая переменная: leadership_potential (потенциал лидерства)
    """
    np.random.seed(42)

    data = pd.DataFrame({
        'student_id': range(n_students),
        'teamwork': np.random.choice([1, 2, 3, 4, 5], n_students, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        'leadership': np.random.choice([1, 2, 3, 4, 5], n_students),
        'adaptability': np.random.choice([1, 2, 3, 4, 5], n_students),
        'optimism': np.random.choice([1, 2, 3, 4, 5], n_students),
        'discipline': np.random.choice([1, 2, 3, 4, 5], n_students)
    })

    # Потенциал лидерства: высокие leadership + teamwork + optimism
    leadership_score = (data['leadership'] + data['teamwork'] + data['optimism']) / 3
    data['leadership_potential'] = (leadership_score > 3.5).astype(int)

    result = {
        'data': data,
        'target': 'leadership_potential',
        'target_values': data['leadership_potential']
    }

    if generate_two_sets:
        np.random.seed(43)
        new_data = data.copy()
        # Имитируем дрейф: снижение лидерских качеств
        new_data['leadership'] = np.clip(new_data['leadership'] - np.random.normal(0.3, 0.2, n_students), 1, 5)
        leadership_score_new = (new_data['leadership'] + new_data['teamwork'] + new_data['optimism']) / 3
        new_data['leadership_potential'] = (leadership_score_new > 3.5).astype(int)

        result['reference'] = data
        result['new'] = new_data

    return result


def _generate_activities_category(n_students: int, generate_two_sets: bool = False) -> dict:
    """
    Категория: Активность (участие в мероприятиях)
    Целевая переменная: active_participation (активный участник)
    """
    np.random.seed(42)

    data = pd.DataFrame({
        'student_id': range(n_students),
        'research_projects': np.random.binomial(1, 0.3, n_students),
        'sports': np.random.binomial(1, 0.4, n_students),
        'volunteering': np.random.binomial(1, 0.25, n_students),
        'creative_events': np.random.binomial(1, 0.35, n_students),
        'student_council': np.random.binomial(1, 0.15, n_students),
        'conferences': np.random.binomial(1, 0.2, n_students)
    })

    data['activity_score'] = data[['research_projects', 'sports', 'volunteering',
                                   'creative_events', 'student_council', 'conferences']].sum(axis=1)
    data['active_participation'] = (data['activity_score'] >= 3).astype(int)

    result = {
        'data': data,
        'target': 'active_participation',
        'target_values': data['active_participation']
    }

    if generate_two_sets:
        np.random.seed(43)
        new_data = data.copy()
        # Имитируем дрейф: снижение активности
        for col in ['research_projects', 'sports', 'volunteering', 'creative_events', 'student_council', 'conferences']:
            new_data[col] = np.random.binomial(1, max(0.1, data[col].mean() - 0.1), n_students)

        new_data['activity_score'] = new_data[['research_projects', 'sports', 'volunteering',
                                               'creative_events', 'student_council', 'conferences']].sum(axis=1)
        new_data['active_participation'] = (new_data['activity_score'] >= 3).astype(int)

        result['reference'] = data
        result['new'] = new_data

    return result


def _generate_career_category(n_students: int, generate_two_sets: bool = False) -> dict:
    """
    Категория: Карьерные намерения
    Целевая переменная: career_clarity (определённость с карьерой)
    """
    np.random.seed(42)

    data = pd.DataFrame({
        'student_id': range(n_students),
        'work_by_specialty': np.random.choice(['Да', 'Нет', 'Еще не решил'], n_students, p=[0.5, 0.2, 0.3]),
        'desired_field': np.random.choice(['IT', 'Образование', 'Бизнес', 'Наука', 'Госслужба'], n_students),
        'has_internship': np.random.binomial(1, 0.4, n_students),
        'works_now': np.random.choice(['Да, по специальности', 'Да, не по специальности', 'Нет'], n_students,
                                      p=[0.2, 0.3, 0.5])
    })

    # Определённость с карьерой: если работа по специальности ИЛИ твёрдое решение работать по специальности
    data['career_clarity'] = (
            ((data['work_by_specialty'] == 'Да') & (data['has_internship'] == 1)) |
            (data['works_now'] == 'Да, по специальности')
    ).astype(int)

    result = {
        'data': data,
        'target': 'career_clarity',
        'target_values': data['career_clarity']
    }

    if generate_two_sets:
        np.random.seed(43)
        new_data = data.copy()
        # Имитируем дрейф: снижение определённости с карьерой
        new_data['work_by_specialty'] = np.random.choice(['Да', 'Нет', 'Еще не решил'], n_students, p=[0.3, 0.3, 0.4])
        new_data['has_internship'] = np.random.binomial(1, 0.3, n_students)
        new_data['career_clarity'] = (
                ((new_data['work_by_specialty'] == 'Да') & (new_data['has_internship'] == 1)) |
                (new_data['works_now'] == 'Да, по специальности')
        ).astype(int)

        result['reference'] = data
        result['new'] = new_data

    return result
