"""
Модуль с функциями для создания признаков (feature engineering)
ВАША ЗОНА ОТВЕТСТВЕННОСТИ
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif, RFE

try:
    import seaborn as sns
except ImportError:
    sns = None


# ---- Композитные признаки согласно ТЗ ----
def add_composite_features(df):
    """
    Создает композитные признаки на основе данных из ТЗ.
    Соответствует пункту ТЗ 6.2 - группировка и агрегирование показателей.
    """
    # 1. Тренд успеваемости (если еще не создан)
    if 'trend_grades' not in df.columns:
        if 'max_grade' in df.columns and 'min_grade' in df.columns:
            df['trend_grades'] = df['max_grade'] - df['min_grade']
        else:
            df['trend_grades'] = 0

    # 2. Стабильность успеваемости (коэффициент вариации)
    if 'grade_std' in df.columns and 'avg_grade' in df.columns:
        df['grade_stability'] = df['grade_std'] / (df['avg_grade'] + 0.01)
    else:
        df['grade_stability'] = 0

    # 3. Когнитивная нагрузка (комбинация из ТЗ)
    cognitive_components = []
    if 'workload_perception' in df.columns:
        cognitive_components.append(df['workload_perception'] * 2)
    if 'stress_level' in df.columns:
        cognitive_components.append(df['stress_level'])
    if 'n_essays' in df.columns:
        cognitive_components.append(df['n_essays'] * 0.5)

    if cognitive_components:
        df['cognitive_load'] = sum(cognitive_components) / len(cognitive_components)
    else:
        df['cognitive_load'] = 0

    # 4. Индекс удовлетворенности (из анкетирования)
    satisfaction_components = []
    if 'satisfaction_score' in df.columns:
        satisfaction_components.append(df['satisfaction_score'])
    if 'engagement_score' in df.columns:
        satisfaction_components.append(df['engagement_score'])

    if satisfaction_components:
        df['overall_satisfaction'] = sum(satisfaction_components) / len(satisfaction_components)
    else:
        df['overall_satisfaction'] = 3.0

    # 5. Психологическое благополучие (из психологического тестирования)
    psych_components = []
    if 'motivation_score' in df.columns:
        psych_components.append(df['motivation_score'])
    if 'anxiety_score' in df.columns:
        psych_components.append(10 - df['anxiety_score'])
    if 'stress_level' in df.columns:
        psych_components.append(10 - df['stress_level'])

    if psych_components:
        df['psychological_wellbeing'] = sum(psych_components) / len(psych_components)
    else:
        df['psychological_wellbeing'] = 5.0

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
    else:
        df['academic_activity'] = 5.0

    return df


def get_base_features(df: pd.DataFrame) -> list:
    """
    Определение базовых признаков из ТЗ
    """
    base_features = []

    # Признаки из успеваемости
    for col in ['avg_grade', 'grade_std', 'min_grade', 'max_grade', 'n_courses', 'avg_brs']:
        if col in df.columns:
            base_features.append(col)

    # Признаки из анкетирования
    for col in ['satisfaction_score', 'engagement_score', 'workload_perception']:
        if col in df.columns:
            base_features.append(col)

    # Признаки из психологического тестирования
    for col in ['stress_level', 'motivation_score', 'anxiety_score']:
        if col in df.columns:
            base_features.append(col)

    # Признаки из эссе
    for col in ['n_essays', 'avg_essay_grade']:
        if col in df.columns:
            base_features.append(col)

    return base_features


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
def preprocess_data(df, feature_cols, target_col):
    x = df[feature_cols]
    y = df[target_col]
    # Fill NA if present
    x = x.fillna(x.median(numeric_only=True))
    # SMOTE
    smote = SMOTE(random_state=42)
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
