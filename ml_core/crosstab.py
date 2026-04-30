"""
Кросс-таблицы для социологических исследований
"""

import pandas as pd
from scipy.stats import chi2_contingency
import plotly.express as px

from ml_core.config import config  # общая утилита
from ml_core.error_handler import logger
import numpy as np


def create_crosstab(
    df,
    row_var,
    col_var,
    values=None,
    aggfunc="count",
    normalize=False,
    auto_bin=True,
    n_bins: int = 4,
    bin_method: str = "cut",
):
    """
    Построение кросс-таблицы между двумя категориальными переменными.

    Parameters:
    -----------
    df : pd.DataFrame
        Исходные данные
    row_var : str
        Переменная для строк
    col_var : str
        Переменная для столбцов
    values : str, optional
        Переменная для агрегации (если нужна средняя оценка и т.п.)
    aggfunc : str
        Функция агрегации ('count', 'mean', 'sum')
    normalize : bool
        Нормализовать по строкам (в процентах)

    Returns:
    --------
    dict с таблицей, статистикой и визуализацией
    """

    df = df.copy()
    bin_info = {}

    # Автоматический биннинг для числовых колонок
    for var in [row_var, col_var]:
        if pd.api.types.is_numeric_dtype(df[var]) and df[var].nunique() > 20 and auto_bin:

            try:
                interval_labels = []

                if bin_method == "qcut":
                    # Равное количество наблюдений в каждой группе
                    bins_series = pd.qcut(df[var], q=n_bins, duplicates="drop", retbins=True)
                    real_bins = bins_series[1]
                    df[f"{var}_group"] = bins_series[0]

                else:  # "cut" — равные интервалы по шкале
                    min_val = df[var].min()
                    max_val = df[var].max()
                    real_bins = np.linspace(min_val, max_val, n_bins + 1)
                    df[f"{var}_group"] = pd.cut(df[var], bins=real_bins, include_lowest=True, duplicates="drop")

                # Формируем читаемые метки интервалов
                for i in range(len(real_bins) - 1):
                    lower = round(real_bins[i], 2)
                    upper = round(real_bins[i + 1], 2)
                    interval_labels.append(f"{lower} — {upper}")

                # Перезаписываем группу на красивые метки
                df[f"{var}_group"] = pd.cut(df[var], bins=real_bins, labels=interval_labels, include_lowest=True)

                bin_info[var] = {
                    "original": var,
                    "binned": f"{var}_group",
                    "intervals": interval_labels,
                    "n_bins": n_bins,
                    "method": bin_method,
                }

                if var == row_var:
                    row_var = f"{var}_group"
                else:
                    col_var = f"{var}_group"

            except Exception as e:
                logger.warning(f"Не удалось выполнить {bin_method} для {var}: {e}")
                # Если биннинг не удался — оставляем как есть

    # Построение таблицы
    if values:
        table = pd.pivot_table(df, values=values, index=row_var, columns=col_var, aggfunc=aggfunc, fill_value=0)
    else:
        table = pd.crosstab(df[row_var], df[col_var], normalize=normalize)

    # Хи-квадрат тест
    if not values and not normalize:
        table_nonzero = table.loc[(table.sum(axis=1) > 0), (table.sum(axis=0) > 0)]
        if table_nonzero.shape[0] > 1 and table_nonzero.shape[1] > 1:
            chi2, p_value, dof, expected = chi2_contingency(table_nonzero)
            chi2_result = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "significant": p_value < 0.05,
            }
        else:
            chi2_result = None
    else:
        chi2_result = None

    # Визуализация
    fig = px.imshow(
        table,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Кросс-таблица: {row_var} vs {col_var}",
    )
    fig.update_layout(xaxis_title=col_var, yaxis_title=row_var)

    table_for_bar = table.reset_index()
    melted = pd.melt(
        table_for_bar,
        id_vars=[row_var if isinstance(row_var, str) else row_var],
        var_name="category",
        value_name="count",
    )

    fig_bar = px.bar(
        melted, x=row_var, y="count", color="category", title=f"Распределение по {col_var}", barmode="stack"
    )

    return {
        "table": table,
        "chi2_test": chi2_result,
        "heatmap": fig,
        "stacked_bar": fig_bar,
        "row_var": row_var,
        "col_var": col_var,
        "bin_info": bin_info,
        "auto_binned": bool(bin_info),
    }


def create_multi_crosstab(df, variables, target_var):
    """
    Построение нескольких кросс-таблиц для набора переменных с целевой.

    Parameters:
    -----------
    df : pd.DataFrame
        Исходные данные
    variables : list
        Список переменных для анализа
    target_var : str
        Целевая переменная (например, risk_flag)

    Returns:
    --------
    dict с результатами для каждой переменной
    """
    df = df.copy()

    results = {}

    for var in variables:
        if var != target_var and var in df.columns:
            try:
                results[var] = create_crosstab(df, var, target_var)
            except Exception as e:
                results[var] = {"error": str(e)}
    return results


def simple_crosstab(df, row_var, col_var):
    """
    Упрощенная версия кросс-таблицы (только таблица и хи-квадрат, без графиков).
    Используется как fallback, если возникают проблемы с визуализацией.

    Returns:
    --------
    dict с таблицей и статистикой
    """
    df = df.copy()

    if row_var not in df.columns or col_var not in df.columns:
        return {"error": "Variable not found"}

    # Строим таблицу
    table = pd.crosstab(df[row_var], df[col_var])

    # Хи-квадрат тест
    try:
        table_nonzero = table.loc[(table.sum(axis=1) > 0), (table.sum(axis=0) > 0)]
        if table_nonzero.shape[0] > 1 and table_nonzero.shape[1] > 1:
            chi2, p_value, dof, expected = chi2_contingency(table_nonzero)
            chi2_result = {"chi2": chi2, "p_value": p_value, "significant": p_value < 0.05}
        else:
            chi2_result = {"error": "Insufficient data for chi-square test"}
    except Exception as e:
        chi2_result = {"error": str(e)}

    return {"table": table, "chi2_test": chi2_result}


def export_crosstab(result, filename="crosstab", format="excel"):
    """
    Экспорт кросс-таблицы в CSV или Excel.

    Args:
        result: dict от create_crosstab() с ключом 'table' (DataFrame)
        filename: имя файла без расширения
        format: 'csv' или 'excel'

    Returns:
        str: путь к сохранённому файлу
    """
    table = result["table"]
    out_dir = config.ANALYSIS_DATA_DIR / "processed"
    if format == "csv":
        table.to_csv(out_dir / f"{filename}.csv", encoding="utf-8")
    elif format == "excel":
        table.to_excel(out_dir / f"{filename}.xlsx", engine="openpyxl")
    return str(out_dir / f"{filename}.{format}")
