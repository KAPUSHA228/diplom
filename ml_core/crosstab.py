"""
Кросс-таблицы для социологических исследований
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import plotly.express as px

from ml_core.config import config  # общая утилита


def create_crosstab(df, row_var, col_var, values=None, aggfunc="count", normalize=False):
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

    # Проверяем, что переменные существуют в DataFrame
    if row_var not in df.columns:
        raise ValueError(f"Переменная '{row_var}' не найдена в данных")
    if col_var not in df.columns:
        raise ValueError(f"Переменная '{col_var}' не найдена в данных")
    for var in [row_var, col_var]:
        if df[var].dtype in [np.number] and df[var].nunique() > 20:
            raise ValueError(
                f"Переменная '{var}' является числовой с {df[var].nunique()} уникальными значениями. Для кросс-таблицы используйте категориальные переменные или предварительно разбейте на группы (pd.cut/pd.qcut)."
            )
    if values:
        # Агрегация по значениям (например, средняя успеваемость)
        table = pd.pivot_table(df, values=values, index=row_var, columns=col_var, aggfunc=aggfunc, fill_value=0)
    else:
        # Просто подсчет частот
        table = pd.crosstab(df[row_var], df[col_var], normalize=normalize)

    # Хи-квадрат тест (только для частот, не для средних)
    if not values and not normalize:
        # Убеждаемся, что таблица не имеет нулевых строк/столбцов
        table_nonzero = table.loc[(table.sum(axis=1) > 0), (table.sum(axis=0) > 0)]
        if table_nonzero.shape[0] > 1 and table_nonzero.shape[1] > 1:
            chi2, p_value, dof, expected = chi2_contingency(table_nonzero)
            chi2_result = {"chi2": chi2, "p_value": p_value, "dof": dof, "significant": p_value < 0.05}
        else:
            chi2_result = None
    else:
        chi2_result = None

    # Визуализация (heatmap)
    fig = px.imshow(
        table,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Кросс-таблица: {row_var} vs {col_var}",
    )
    fig.update_layout(xaxis_title=col_var, yaxis_title=row_var)

    # Создаем копию таблицы и сбрасываем индекс
    table_for_bar = table.reset_index()

    # Переименовываем колонки, чтобы избежать дублирования
    new_column_names = []
    for col in table_for_bar.columns:
        if col == row_var:
            new_column_names.append(row_var)
        else:
            # Для остальных колонок используем их строковое представление
            new_column_names.append(str(col))

    table_for_bar.columns = new_column_names

    # Преобразуем в длинный формат
    melted = pd.melt(table_for_bar, id_vars=[row_var], var_name="category", value_name="count")

    # Удаляем возможные дубликаты
    melted = melted.drop_duplicates().reset_index(drop=True)

    # Очищаем от NaN
    melted = melted.dropna(subset=["count"])

    # Убеждаемся, что колонки уникальны
    if melted.columns.duplicated().any():
        # Если есть дубликаты, создаем новый DataFrame с уникальными именами
        melted = melted.loc[:, ~melted.columns.duplicated()]

    # Переименовываем колонку category в col_var для красоты
    melted = melted.rename(columns={"category": col_var})

    # Строим stacked bar chart
    fig_bar = px.bar(
        melted, x=row_var, y="count", color=col_var, title=f"Распределение {row_var} по {col_var}", barmode="stack"
    )

    return {
        "table": table,
        "chi2_test": chi2_result,
        "heatmap": fig,
        "stacked_bar": fig_bar,
        "row_var": row_var,
        "col_var": col_var,
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
