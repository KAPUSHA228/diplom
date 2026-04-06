"""
Автоматическая обработка пропусков (null) в данных
"""

import numpy as np


def handle_missing_values(df, strategy="auto", threshold=30):
    """
    Многоуровневая обработка пропусков с автоматическим или ручным выбором стратегии.

    Args:
        df: исходный DataFrame
        strategy: стратегия ('auto', 'drop_rows', 'drop_columns', 'fill_mean',
                  'fill_median', 'fill_mode', 'interpolate')
        threshold: процент пропусков для удаления столбца при 'auto' (по умолчанию 30%)

    Returns:
        (df_clean, report): обработанный DataFrame и dict-отчёт о действиях
    """
    df = df.copy()
    df_clean = df.copy()
    report = {"original_shape": df.shape, "missing_before": df.isna().sum().sum(), "actions": []}

    for col in df.columns:
        null_pct = df[col].isna().mean() * 100

        if null_pct == 0:
            continue

        col_strategy = strategy

        if strategy == "auto":
            if null_pct > threshold:
                col_strategy = "drop_column"
            elif null_pct > 10:
                col_strategy = "fill_median" if df[col].dtype in ["int64", "float64"] else "fill_mode"
            else:
                col_strategy = "interpolate" if df[col].dtype in ["int64", "float64"] else "fill_mode"

        action = {"column": col, "null_pct": null_pct, "strategy": col_strategy}

        if col_strategy == "drop_column":
            df_clean = df_clean.drop(columns=[col])
            action["message"] = f"Удален столбец (пропусков: {null_pct:.1f}%)"

        elif col_strategy == "drop_rows":
            before = len(df_clean)
            df_clean = df_clean.dropna(subset=[col])
            action["message"] = f"Удалено строк: {before - len(df_clean)}"

        elif col_strategy == "fill_mean":
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            action["message"] = f"Заполнено средним ({df_clean[col].mean():.2f})"

        elif col_strategy == "fill_median":
            median = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median)
            action["message"] = f"Заполнено медианой ({median:.2f})"

        elif col_strategy == "fill_mode":
            mode = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "unknown"
            df_clean[col] = df_clean[col].fillna(mode)
            action["message"] = f"Заполнено модой ({mode})"

        elif col_strategy == "interpolate":
            df_clean[col] = df_clean[col].interpolate(method="linear")
            action["message"] = "Линейная интерполяция"

        elif col_strategy == "flag":
            df_clean[f"{col}_missing"] = df_clean[col].isna().astype(int)
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            action["message"] = "Добавлен флаг пропуска"

        report["actions"].append(action)

    report["missing_after"] = df_clean.isna().sum().sum()
    report["final_shape"] = df_clean.shape

    return df_clean, report


def detect_outliers(df, columns=None, method="iqr", threshold=1.5):
    """
    Обнаружение выбросов методами IQR или Z-score.

    Args:
        df: исходный DataFrame
        columns: список колонок для проверки (если None — все числовые)
        method: метод обнаружения ('iqr' или 'zscore')
        threshold: множитель для IQR (по умолчанию 1.5) или Z-score (по умолчанию 3.0)

    Returns:
        dict: {column_name: {'n_outliers', 'percentage', 'lower_bound', 'upper_bound', ...}}
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers_report = {}

    for col in columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = data[(data < lower) | (data > upper)]

        elif method == "zscore":
            z = (data - data.mean()) / data.std()
            outliers = data[abs(z) > threshold]

        outliers_report[col] = {
            "n_outliers": len(outliers),
            "percentage": len(outliers) / len(data) * 100 if len(data) > 0 else 0,
            "lower_bound": lower if method == "iqr" else None,
            "upper_bound": upper if method == "iqr" else None,
            "outlier_values": outliers.tolist(),
        }

    return outliers_report
