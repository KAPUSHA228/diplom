"""
Модуль для оценки моделей и объяснения предсказаний
ВАША ЗОНА ОТВЕТСТВЕННОСТИ
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    confusion_matrix,
)
import shap
import matplotlib.pyplot as plt

import seaborn as sns

from ml_core.error_handler import logger


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Расчет всех метрик
    """
    metrics = {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

    return metrics


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Построение ROC-кривых для нескольких моделей
    """
    fig = go.Figure()

    # Диагональная линия (случайная модель)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines', name='Random',
        line=dict(dash='dash', color='gray')
    ))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines', name=f'{name} (AUC={auc:.3f})'
            ))

    fig.update_layout(
        title='ROC-кривые моделей',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600, height=500
    )

    return fig


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Построение confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="True"),
        title=f'Confusion Matrix: {model_name}',
        color_continuous_scale='Blues'
    )

    return fig


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Построение важности признаков
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    # Сортируем
    indices = np.argsort(importances)[-top_n:]

    fig = px.bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        title='Важность признаков',
        labels={'x': 'Importance', 'y': ''}
    )

    return fig


def generate_shap_explanations(model, X: pd.DataFrame, feature_names: list, threshold: float = 0.5, top_n: int = 5):
    """Единый обработчик SHAP с корректной поддержкой всех типов моделей"""
    try:
        # Единый способ создания explainer
        if isinstance(model, (XGBClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X) if hasattr(model, 'coef_') else shap.Explainer(model)

        shap_values = explainer.shap_values(X)

        # Приведение к единому формату (для класса 1)
        if isinstance(shap_values, list) and len(shap_values) == 2:   # TreeExplainer для бинарной классификации
            shap_values = shap_values[1]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]

        explanations = []
        for i in range(min(len(X), 15)):   # ограничиваем для производительности
            risk_prob = float(model.predict_proba(X.iloc[[i]])[0, 1])

            # shap_values может быть ndarray
            shap_row = shap_values[i] if len(shap_values.shape) == 2 else shap_values[i]

            feature_effects = list(zip(feature_names, shap_row))
            feature_effects.sort(key=lambda x: abs(x[1]), reverse=True)

            explanation = {
                'student_index': i,
                'risk_probability': risk_prob,
                'risk_level': 'high' if risk_prob > threshold else 'low',
                'top_features': [
                    {
                        'feature': feat,
                        'shap_value': float(val),
                        'effect': 'увеличивает риск' if val > 0 else 'снижает риск'
                    }
                    for feat, val in feature_effects[:top_n]
                ]
            }
            explanation['explanation'] = generate_text_explanation(explanation)
            explanations.append(explanation)

        return explanations

    except Exception as e:
        logger.error(f"SHAP generation failed: {str(e)}", exc_info=True)
        return []

def generate_text_explanation(exp):
    """
    Генерация человеко-читаемого объяснения
    """
    text = f"Риск отчисления: {exp['risk_probability']:.1%} ({exp['risk_level']})\n\n"
    text += "Основные факторы:\n"

    for feat in exp['top_features']:
        arrow = "↑" if feat['effect'] == 'увеличивает риск' else "↓"
        text += f"• {feat['feature']}: {arrow} ({feat['shap_value']:.3f})\n"

    return text


# ---- SHAP анализ ----
def explain_model_with_shap(model, X_explain):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_explain)
    shap.summary_plot(shap_values, X_explain, show=False)
    # Форматировать текстовое объяснение для каждого объекта
    explanations = []
    for i in range(X_explain.shape[0]):
        vals = shap_values[i].values
        # Преобразуем в одномерный массив, если нужно
        if vals.ndim > 1:
            vals = vals.flatten()
        top_idx = np.argsort(np.abs(vals))[-2:][::-1]
        top_feats = [X_explain.columns[idx] for idx in top_idx]
        top_vals = [float(vals[idx]) for idx in top_idx]
        txt = f"Риск из-за: {top_feats[0]} ({top_vals[0]:+.2f}), {top_feats[1]} ({top_vals[1]:+.2f})"
        explanations.append(txt)
    return explanations


def generate_detailed_explanation(model, X_explain, student_idx, feature_names, threshold=0.5):
    """
    Генерирует детальное текстовое объяснение для конкретного студента.
    Включает вероятность риска, топ-3 фактора и рекомендации.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_explain.iloc[[student_idx]])

    # Получаем предсказание
    proba = model.predict_proba(X_explain.iloc[[student_idx]])[0, 1]
    risk_level = "высокий" if proba >= threshold else "низкий"

    # Извлекаем SHAP значения
    vals = shap_values.values[0]
    if vals.ndim > 1:
        vals = vals.flatten()

    # Убеждаемся, что количество значений совпадает с количеством признаков
    n_features = len(feature_names)
    if len(vals) != n_features:
        # Берем только первые n_features значений
        vals = vals[:n_features]

    # Топ-3 фактора (по абсолютному значению), но не больше доступных признаков
    top_n = min(3, n_features)
    top_3_idx = np.argsort(np.abs(vals))[-top_n:][::-1]
    # Фильтруем индексы, чтобы они были в допустимых пределах
    top_3_idx = [idx for idx in top_3_idx if 0 <= idx < n_features]
    top_features = [feature_names[idx] for idx in top_3_idx]
    top_shap_vals = [float(vals[idx]) for idx in top_3_idx]

    # Формируем объяснение
    explanation_parts = [
        f"Вероятность академического риска: {proba:.1%} (уровень: {risk_level})",
        "\nОсновные факторы риска:"
    ]

    for feat, shap_val in zip(top_features, top_shap_vals):
        direction = "увеличивает" if shap_val > 0 else "снижает"
        explanation_parts.append(f"  • {feat}: {direction} риск (вклад: {shap_val:+.3f})")

    # Простые рекомендации на основе признаков
    recommendations = []
    for feat, shap_val in zip(top_features, top_shap_vals):
        if shap_val > 0:  # Фактор увеличивает риск
            if 'trend' in feat.lower() or 'grade' in feat.lower():
                recommendations.append("Рекомендуется обратить внимание на динамику успеваемости")
            elif 'attendance' in feat.lower():
                recommendations.append("Необходимо улучшить посещаемость занятий")
            elif 'stress' in feat.lower() or 'cognitive' in feat.lower():
                recommendations.append("Требуется снижение когнитивной нагрузки и уровня стресса")

    if recommendations:
        explanation_parts.append("\nРекомендации:")
        explanation_parts.extend([f"  • {rec}" for rec in recommendations[:3]])

    return "\n".join(explanation_parts)


def generate_batch_explanations(model, X_explain, feature_names, threshold=0.5, top_n=10):
    """
    Генерирует объяснения для топ-N студентов с наибольшим риском.
    Возвращает список словарей с детальными объяснениями.
    """
    probas = model.predict_proba(X_explain)[:, 1]
    top_risk_indices = np.argsort(probas)[-top_n:][::-1]

    explanations_list = []
    for idx in top_risk_indices:
        explanation = generate_detailed_explanation(
            model, X_explain, idx, feature_names, threshold
        )
        explanations_list.append({
            'student_index': int(idx),
            'risk_probability': float(probas[idx]),
            'explanation': explanation
        })

    return explanations_list


# ---- Визуализация метрик и важности признаков ----

def plot_confusion(model, X_test, y_test, name="model", output_path=None):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    if output_path is None:
        output_path = f"cm_{name}.png"
    if sns is not None:
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion matrix: {name}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    return cm


def plot_feature_importance(model, feature_names, name="model", output_path=None, top_n=10):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(6, 4))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(f"Feature importance: {name}")
    plt.tight_layout()
    if output_path is None:
        output_path = f"fi_{name}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

