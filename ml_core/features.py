"""
Модуль с функциями для создания признаков (feature engineering)
ВАША ЗОНА ОТВЕТСТВЕННОСТИ
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif, RFE
from itertools import combinations
from typing import List, Dict, Union, Optional
from ml_core.error_handler import logger


def build_composite_score(df, feature_weights: dict,
                          score_name="custom_score",
                          normalize: bool = True,
                          assign_id: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Конструктор композитной оценки.

    feature_weights = {'avg_grade': 0.4, 'stress_level': -0.3, 'activity_score': 0.3, ...}
    Отрицательный вес = обратное влияние.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    for feature, weight in feature_weights.items():
        if feature in df.columns:
            col = df[feature].fillna(df[feature].median())
            if normalize and df[feature].dtype in ['int64', 'float64']:
                col = (col - col.min()) / (col.max() - col.min() + 1e-8)
            score += col * weight

    df[score_name] = score
    if assign_id:
        df[f"{score_name}_id"] = pd.qcut(
            df[score_name],
            q=100,
            labels=False,
            duplicates='drop'
        ) + 1  # от 1 до 100
    return df, score_name


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает композитные признаки ТОЛЬКО если есть необходимые исходные данные.
    Если данных нет — колонка не создаётся или заполняется NaN.
    """
    df = df.copy()

    # 1. Тренд успеваемости
    if {'max_grade', 'min_grade'}.issubset(df.columns):
        df['trend_grades'] = df['max_grade'] - df['min_grade']
    # else: колонка не создаётся — это лучше, чем заполнять 0

    # 2. Стабильность успеваемости
    if {'grade_std', 'avg_grade'}.issubset(df.columns):
        df['grade_stability'] = df['grade_std'] / (df['avg_grade'] + 0.01)

    # 3. Когнитивная нагрузка
    cognitive_components = []
    if 'workload_perception' in df.columns:
        cognitive_components.append(df['workload_perception'] * 2)
    if 'stress_level' in df.columns:
        cognitive_components.append(df['stress_level'])
    if 'n_essays' in df.columns:
        cognitive_components.append(df['n_essays'] * 0.5)

    if cognitive_components:
        df['cognitive_load'] = sum(cognitive_components) / len(cognitive_components)
    # else: не создаём колонку

    # 4. Индекс удовлетворенности
    satisfaction_components = []
    if 'satisfaction_score' in df.columns:
        satisfaction_components.append(df['satisfaction_score'])
    if 'engagement_score' in df.columns:
        satisfaction_components.append(df['engagement_score'])

    if satisfaction_components:
        df['overall_satisfaction'] = sum(satisfaction_components) / len(satisfaction_components)

    # 5. Психологическое благополучие
    psych_components = []
    if 'motivation_score' in df.columns:
        psych_components.append(df['motivation_score'])
    if 'anxiety_score' in df.columns:
        psych_components.append(10 - df['anxiety_score'])
    if 'stress_level' in df.columns:
        psych_components.append(10 - df['stress_level'])

    if psych_components:
        df['psychological_wellbeing'] = sum(psych_components) / len(psych_components)

    # 6. Академическая активность
    activity_components = []
    if 'avg_grade' in df.columns:
        activity_components.append(df['avg_grade'] * 2)
    if 'n_courses' in df.columns:
        activity_components.append(df['n_courses'] * 1.0)
    if 'n_essays' in df.columns:
        activity_components.append(df['n_essays'] * 2.0)

    if activity_components:
        df['academic_activity'] = sum(activity_components) / len(activity_components)

    return df


def get_base_features(df: pd.DataFrame, is_synthetic: bool = False) -> list:
    """
    Определение базовых признаков для анализа.
    Для синтетических данных берёт все числовые колонки, кроме служебных.
    """
    df = df.copy()

    exclude = {
        'student_id', 'user', 'user_id', 'VK_id', 'vk ID',
        'Фамилия', 'Имя', 'ВУЗ', 'дата', 'date',
        '_source_sheet', '_sheet_type', 'cluster',
        'risk_flag', 'burnout_risk', 'high_creativity',
        'value_profile', 'leadership_potential',
        'active_participation', 'career_clarity'
    }

    if is_synthetic:
        # Для синтетики берём ВСЕ числовые колонки, кроме исключений
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col not in exclude]

        # Если после исключения ничего не осталось — берём все числовые кроме student_id
        if not features:
            features = [col for col in numeric_cols if col != 'student_id']

        return features
    else:
        # Для реальных данных — все числовые, кроме служебных
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude]


def select_features_for_model(x, y, top_n=7, final_n=5):
    """
    Отбор признаков для модели
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier

    # Метод 1: ANOVA F-value
    selector_f = SelectKBest(score_func=f_classif, k=min(top_n, x.shape[1]))
    selector_f.fit(x, y)
    f_scores = pd.Series(selector_f.scores_, index=x.columns)

    # Метод 2: Важность из Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x, y)
    rf_importance = pd.Series(rf.feature_importances_, index=x.columns)

    # Комбинируем метрики
    combined_scores = (f_scores / f_scores.max() + rf_importance / rf_importance.max()) / 2
    top_features = combined_scores.nlargest(final_n).index.tolist()

    return x[top_features], top_features


# ---- Предобработка и борьба с дисбалансом ----
def preprocess_data(df: pd.DataFrame, feature_cols: list, target_col: str, use_smote: bool = True):
    df = df.copy()

    x = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df[target_col]

    if not use_smote:
        return x, y

    min_class_size = y.value_counts().min()
    if min_class_size <= 5:
        logger.warning(f"SMOTE отключён: в меньшем классе всего {min_class_size} объектов")
        return x, y

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_size - 1))
    x_res, y_res = smote.fit_resample(x, y)
    return x_res, y_res


# ---- Отбор признаков ----
def select_features(X, y, top_n=10, final_n=5):
    mi = mutual_info_classif(X, y)
    top_features = X.columns[np.argsort(-mi)[:top_n]]
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=final_n)
    X_sel = rfe.fit_transform(X[top_features], y)
    selected_cols = np.array(top_features)[rfe.support_]
    return pd.DataFrame(X_sel, columns=selected_cols), selected_cols


def preprocess_data_for_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """SMOTE применяется ТОЛЬКО к обучающей выборке"""
    min_class_size = y_train.value_counts().min()
    if min_class_size <= 5:
        logger.warning(f"SMOTE отключён: в меньшем классе {min_class_size} объектов")
        return X_train, y_train

    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_size - 1))
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def create_feature_combinations(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    max_pairs: int = 15,
    methods: List[str] = None
) -> pd.DataFrame:
    """
    Автоматически создаёт новые признаки путём объединения существующих.
    Поддерживает: num+num, num+text, text+text.
    """
    if methods is None:
        methods = ['sum', 'diff', 'ratio', 'product', 'concat']

    df = df.copy()
    new_features = {}

    # 1. Числовые + Числовые
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col1, col2 in combinations(numerical_cols, 2):
        if len(new_features) >= max_pairs:
            break
        if col1 == col2:
            continue

        if 'sum' in methods:
            new_features[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
        if 'diff' in methods:
            new_features[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
        if 'ratio' in methods and (df[col2] != 0).all():
            new_features[f"{col1}_div_{col2}"] = df[col1] / df[col2]
        if 'product' in methods:
            new_features[f"{col1}_mul_{col2}"] = df[col1] * df[col2]

    # 2. Числовые + Текстовые
    if text_cols is None:
        text_cols = [col for col in df.columns if df[col].dtype == 'object']

    for num_col in numerical_cols[:3]:           # ограничиваем, чтобы не плодить слишком много
        for txt_col in text_cols:
            if len(new_features) >= max_pairs:
                break
            # Пример: длина текста * числовой признак
            new_features[f"{num_col}_by_{txt_col}_len"] = df[num_col] * df[txt_col].astype(str).str.len()

    # 3. Текстовые + Текстовые
    for col1, col2 in combinations(text_cols, 2):
        if len(new_features) >= max_pairs:
            break
        new_features[f"{col1}_and_{col2}"] = (
            df[col1].astype(str) + " | " + df[col2].astype(str)
        )

    # Добавляем новые признаки
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df
