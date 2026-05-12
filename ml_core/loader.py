"""
Загрузчик Excel-опросников
Гибкая обработка разных типов листов
"""

import pandas as pd

from ml_core.error_handler import logger

# Служебные колонки (PII) — исключаем из превью полностью (и колонку, и данные)
# Все в нижнем регистре для сравнения с col.strip().lower()
SERVICE_COLS = {
    "user",
    "user_id",
    "vk_id",
    "vk id",
    "vk",
    "фамилия",
    "имя",
    "отчество",
    "вуз",
    "факультет",
    "группа",
    "курс",
    "пол",
    "возраст",
    "дата",
    "date",
    "направление подготовки",
    "_source_sheet",
    "_sheet_type",
}


def detect_column_type(series: pd.Series, sample_size: int = 500) -> str:
    """
    Определяет тип колонки на основе анализа значений.

    Returns:
        'numeric': все значения можно привести к числу
        'single_choice': текстовые значения, нет разделителей, мало уникальных
        'multiple_choice': есть разделители (;, ,, |)
        'skip': свободный текст, много уникальных значений
    """
    # Удаляем NaN
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return "skip"

    # Берём сэмпл для анализа
    if len(clean_series) > sample_size:
        clean_series = clean_series.sample(n=sample_size, random_state=42)

    # 1. Проверка на числовой
    numeric_test = pd.to_numeric(clean_series, errors="coerce")
    if numeric_test.notna().mean() >= 0.95:  # 95% можно привести к числу
        return "numeric"

    # Преобразуем в строки для дальнейшего анализа
    str_series = clean_series.astype(str)

    # 2. Проверка на множественный выбор (наличие разделителей)
    separators = [";", ",", "|", "/", "\\", "—"]
    for sep in separators:
        if str_series.str.contains(sep).any():
            return "multiple_choice"

    # 3. Проверка на единственный выбор
    unique_count = str_series.nunique()
    total_count = len(str_series)
    unique_ratio = unique_count / total_count
    avg_length = str_series.str.len().mean()

    # Мало уникальных значений → категориальный (single_choice)
    if unique_ratio < 0.5:
        return "single_choice"

    # Много уникальных и длинные тексты → skip (свободные ответы)
    if unique_ratio > 0.8 and avg_length > 30:
        return "skip"

    # По умолчанию single_choice
    return "single_choice"


def detect_sheet_group(df: pd.DataFrame, sample_size: int = 1000) -> str:
    """
    Определяет группу листа по содержимому колонок (4 категории).

    Args:
        columns: список имён колонок листа
        sheet_name: опционально, имя листа для более точного определения

    Returns:
        str: группа листа ('grades', 'psychology', 'survey', 'unknown', ...)
    """
    print(f"DEBUG detect_sheet_group: type(df)={type(df)}")
    print(f"DEBUG detect_sheet_group: hasattr columns={hasattr(df, 'columns')}")
    df_sample = df.head(sample_size) if len(df) > sample_size else df

    column_types = []

    for col in df_sample.columns:
        col_clean = col.strip().lower()
        if col_clean in SERVICE_COLS:
            continue

        col_type = detect_column_type(df_sample[col])
        column_types.append(col_type)

    if not column_types:
        return "skip"

    # Анализируем все колонки
    numeric_count = column_types.count("numeric")
    single_count = column_types.count("single_choice")
    multi_count = column_types.count("multiple_choice")
    skip_count = column_types.count("skip")
    total = len(column_types)

    # Определяем основной тип
    if numeric_count == total:
        return "numeric"
    elif multi_count > 0:
        return "multiple_choice"
    elif skip_count > total * 0.7:  # >70% skip-колонок
        return "skip"
    elif single_count > 0:
        return "single_choice"
    else:
        return "mixed"


def normalize_sheet_group(sheet_group: str | None) -> str:
    """
    Приводит группу листа к каноническим именам.
    Вызовы с ключами паттернов (category3_single_choice) дают то же поведение, что и single_choice.
    """
    if not sheet_group:
        return "unknown"
    sg = str(sheet_group).strip()
    if sg in ("numeric", "single_choice", "multiple_choice", "skip", "unknown"):
        return sg
    key_to_group = {
        "category1_numeric": "numeric",
        "category2_mednik": "skip",
        "category3_single_choice": "single_choice",
        "category4_multiple_choice": "multiple_choice",
    }
    return key_to_group.get(sg, sg)


def get_sheet_preview(file_path: str, sheet_name: str) -> dict:
    """
    Анализирует лист Excel и возвращает структуру для UI-маппинга.
    """
    print(f"DEBUG get_sheet_preview: file_path={file_path}, sheet_name='{sheet_name}'")

    # 1. Загружаем ТОЛЬКО первые 1000 строк для анализа
    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=1000)
    print(f"DEBUG get_sheet_preview: загружен DataFrame с колонками: {list(df.columns)[:5]}...")

    # 2. Базовая очистка
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # 3. Убеждаемся, что df — это DataFrame
    print(f"DEBUG get_sheet_preview: type(df) after load = {type(df)}")

    # 4. Определяем группу (передаём DataFrame!)
    group = detect_sheet_group(df)
    print(f"DEBUG get_sheet_preview: detected_group = '{group}'")

    cols_info = []
    for col in df.columns:
        col_clean = col.strip().lower()
        if col_clean in SERVICE_COLS:
            print(f"DEBUG preview: колонка='{col}' -> service=True")
            continue

        print(f"DEBUG preview: колонка='{col}' -> service=False")
        col_info = {"name": col, "dtype": str(df[col].dtype)}

        if df[col].dtype == "object":
            if group == "multiple_choice":
                all_values = set()
                separators = [";", ","]
                for val in df[col].dropna().unique():
                    val_str = str(val)
                    split_val = [val_str]
                    for sep in separators:
                        if sep in val_str:
                            split_val = [v.strip() for v in val_str.split(sep)]
                            break
                    all_values.update(split_val)
                col_info["unique_values"] = sorted(str(v) for v in all_values)
            else:
                vals = [str(v) for v in df[col].dropna().unique()]
                col_info["unique_values"] = sorted(set(vals))
        else:
            col_info["unique_values"] = []

        cols_info.append(col_info)

    group_labels = {
        "numeric": "📊 Числовые данные",
        "single_choice": "📝 Одиночный выбор",
        "multiple_choice": "☑️ Множественный выбор",
        "skip": "🗑️ Свободные ответы (скип)",
    }

    return {
        "sheet_name": sheet_name,
        "detected_group": group,
        "group_label": group_labels.get(group, "❓ Не определено"),
        "columns": cols_info,
    }


def preprocess_sheet(df: pd.DataFrame, sheet_group: str, sheet_name: str = None, mapping_config: dict = None) -> tuple:
    """
    Предобработка одного листа Excel.
    mapping_config: словарь настроек от пользователя {col_name: {type: ..., map: ...}}
    """
    df = df.copy()
    sheet_group = normalize_sheet_group(sheet_group)
    message_parts = [f"Лист '{sheet_name}' → группа: {sheet_group}"]

    # Базовая очистка
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    from ml_core.config import optimize_dtypes

    df = optimize_dtypes(df)
    # Поиск user_col
    user_col = next((col for col in ["user", "user_id", "VK_id", "VK", "student_id"] if col in df.columns), None)
    if not user_col:
        df["user_id"] = range(len(df))
        user_col = "user_id"

    processed_cols = set()

    # === 1. ПРИМЕНЯЕМ ПОЛЬЗОВАТЕЛЬСКИЙ КОНФИГ (mapping) ===
    columns_mapping = None
    if mapping_config and isinstance(mapping_config, dict):
        columns_mapping = mapping_config.get("columns")
        if not columns_mapping:
            reserved = {
                "sheet_name",
                "sheet_type",
                "global_settings",
                "detected_group",
                "columns",
            }
            if mapping_config.keys() and not (set(mapping_config.keys()) & reserved):
                columns_mapping = mapping_config

    if columns_mapping:
        message_parts.append("(пользовательский mapping_config)")

        for original_name, col_config in columns_mapping.items():
            # Устойчивое сопоставление имён колонок
            matching_col = None
            for existing_col in df.columns:
                if str(existing_col).strip() == str(original_name).strip():
                    matching_col = existing_col
                    break
                if str(existing_col).strip().lower().replace("–", "-").replace("—", "-") == str(
                    original_name
                ).strip().lower().replace("–", "-").replace("—", "-"):
                    matching_col = existing_col
                    break

            if matching_col is None:
                logger.warning(f"Колонка '{original_name}' не найдена в DataFrame")
                continue

            processed_cols.add(matching_col)
            col_type = col_config.get("type")

            try:
                if col_type == "skip":
                    continue

                if col_type in ["multiple_choice", "split"]:
                    sep = col_config.get("separator", ";")
                    df = process_multiple_choice_column(df, matching_col, prefix=f"{matching_col}_", separator=sep)

                elif col_type == "ordinal":
                    val_map = col_config.get("map") or col_config.get("mapping") or {}
                    if val_map:
                        mapped = df[matching_col].map(val_map)
                        df[matching_col] = pd.to_numeric(mapped, errors="coerce")

                elif col_type in ("one_hot", "categorical") or col_config.get("encoding") == "onehot":
                    dummies = pd.get_dummies(df[matching_col], prefix=matching_col, prefix_sep="_")
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[matching_col])

            except Exception as e:
                logger.error(f"Ошибка обработки колонки '{matching_col}' (type={col_type}): {e}", exc_info=True)
                raise

    # === 2. Автоматическая обработка оставшихся колонок ===
    for col in list(df.columns):
        if col in processed_cols or col == user_col:
            continue

        # Удаляем даты
        if any(x in str(col).lower() for x in ["дата", "date"]):
            df = df.drop(columns=[col])
            continue

        # Multiple choice по группе листа — dummy-столбцы по вариантам ответа
        if sheet_group == "multiple_choice" and df[col].dtype == "object":
            prefix = f"{sheet_name.replace(' ', '_')}_" if sheet_name else ""
            df = process_multiple_choice_column(df, col, prefix=prefix, separator=";")

        elif sheet_group == "single_choice" and df[col].dtype == "object":
            # Без явного маппинга: числовые строки оставляем числом, иначе one-hot (как разумный дефолт)
            num = pd.to_numeric(df[col], errors="coerce")
            if num.notna().mean() >= 0.95:
                df[col] = num
            else:
                dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="_", dummy_na=False)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)

        elif sheet_group == "unknown" and df[col].dtype == "object":
            num = pd.to_numeric(df[col], errors="coerce")
            if num.notna().mean() >= 0.95:
                df[col] = num
            else:
                dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="_", dummy_na=False)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)

        elif sheet_group == "skip" and df[col].dtype == "object":
            # Свободный текст / смысловой «скип» — не кодируем в числа (позже LLM)
            continue

        else:
            # numeric и прочие: приводим к числу
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # === 3. Удаление служебных колонок ===
    service_cols = [col for col in df.columns if str(col).strip().lower() in SERVICE_COLS]
    if service_cols:
        df = df.drop(columns=service_cols)
    # === 4. Гарантируем наличие идентификатора студента ===
    if "student_id" not in df.columns:
        df["student_id"] = [f"student_{i:06d}" for i in range(len(df))]

    # Перемещаем student_id в начало для удобства
    if "student_id" in df.columns:
        cols = ["student_id"] + [c for c in df.columns if c != "student_id"]
        df = df[cols]

    df = df.loc[:, ~df.columns.duplicated()]

    message = " | ".join(message_parts)
    return df, message


def process_multiple_choice_column(df: pd.DataFrame, col: str, prefix: str = "", separator: str = ";") -> pd.DataFrame:
    """
    Разбивает колонку с множественным выбором на бинарные dummy-колонки.
    Например: "Волонтёрство, Спорт" → column_Волонтёрство=1, column_Спорт=1.

    Args:
        df: исходный DataFrame
        col: имя колонки с множественным выбором
        prefix: префикс для новых колонок
        separator: разделитель

    Returns:
        pd.DataFrame: DataFrame с добавленными dummy-колонками
    """
    if col not in df.columns:
        return df

    # Копируем датафрейм
    df = df.copy()

    # Разделители, которые могут встречаться
    separators = [separator, ", ", "; ", ",", ";", " | "]

    if pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].astype(str).replace("nan", "")
    # Векторизованное разделение с помощью str.extractall
    # 1. Сначала создаём массив всех вариантов
    all_choices = set()
    for val in df[col].dropna():
        val_str = str(val)
        for sep in separators:
            if sep in val_str:
                all_choices.update([v.strip() for v in val_str.split(sep)])
                break
        else:
            all_choices.add(val_str)

    all_choices = sorted(all_choices)
    if not all_choices:
        return df

    # 2. Создаём dummy-колонки одной операцией через str.get_dummies
    # Преобразуем в строки и обрабатываем
    series = df[col].fillna("").astype(str)

    # Для каждого разделителя пробуем разбить
    for sep in separators:
        if series.str.contains(sep).any():
            # Используем str.get_dummies с разделителем
            dummies = series.str.get_dummies(sep=sep)
            # Переименовываем колонки с префиксом
            dummies = dummies.add_prefix(f"{prefix}")
            # Добавляем в df
            df = pd.concat([df, dummies], axis=1)
            break
    else:
        # Если разделителей нет — просто one-hot
        dummies = pd.get_dummies(series, prefix=prefix)
        df = pd.concat([df, dummies], axis=1)

    # Удаляем исходную колонку
    df = df.drop(columns=[col])

    return df


# def detect_sheet_type(sheet_name: str) -> str:
#     """
#     Определяет тип листа по его имени (ключевые слова: williams, schwartz, соц...).
#
#     Args:
#         sheet_name: имя листа Excel
#
#     Returns:
#         str: тип листа ('williams', 'schwartz', 'demographics', ...)
#     """
#     sheet_lower = sheet_name.lower()
#
#     if "вильямс" in sheet_lower:
#         return "williams"
#     elif "шварц" in sheet_lower:
#         return "schwartz"
#     elif "соц" in sheet_lower:
#         if "соц14" in sheet_lower or "соцдем" in sheet_lower:
#             return "demographics"
#         elif "соц11" in sheet_lower or "соц12" in sheet_lower or "соц13" in sheet_lower:
#             return "career"
#         elif "соц9" in sheet_lower:
#             return "grades"
#         elif "соц8" in sheet_lower:
#             return "attitudes"
#         elif "соц4" in sheet_lower or "соц5" in sheet_lower or "соц6" in sheet_lower:
#             return "activities"
#         elif "соц3" in sheet_lower:
#             return "digital"
#         elif "соц2" in sheet_lower:
#             return "interests"
#         elif "соц1" in sheet_lower:
#             return "personality"
#         else:
#             return "social"
#
#     return "unknown"
#
# def load_excel_sheet(file_path: str, sheet_name: str) -> tuple:
#     """
#     Загружает, определяет тип и предобрабатывает один лист Excel.
#
#     Args:
#         file_path: путь к Excel-файлу
#         sheet_name: имя листа
#
#     Returns:
#         (df_processed, message): обработанный DataFrame и сообщение
#     """
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     sheet_type = detect_sheet_type_by_columns(df.columns, sheet_name)
#     df_processed, msg = preprocess_sheet(df, sheet_type)
#     return df_processed, msg
#
# def get_sheet_names(file_path: str) -> list:
#     """
#     Возвращает список имён листов в Excel-файле.
#
#     Args:
#         file_path: путь к Excel-файлу
#
#     Returns:
#         list[str]: имена листов
#     """
#     xl = pd.ExcelFile(file_path)
#     return xl.sheet_names
# def process_multiple_choice_ordinal_column(df: pd.DataFrame, col: str, separator: str = ";") -> pd.DataFrame:
#     """
#     Для multiple-choice строит один ordinal-скор вместо one-hot.
#     Скор = средний ранг выбранных вариантов (ранги по алфавиту вариантов).
#     """
#     if col not in df.columns:
#         return df
#
#     df = df.copy()
#     separators = [separator, ", ", "; ", ",", ";", " | ", "|"]
#
#     def split_choices(text):
#         if pd.isna(text):
#             return []
#         text = str(text).strip()
#         for sep in separators:
#             if sep in text:
#                 return [x.strip() for x in text.split(sep) if x.strip()]
#         return [text] if text else []
#
#     all_choices = set()
#     for val in df[col].dropna():
#         all_choices.update(split_choices(val))
#     all_choices = sorted(all_choices)
#     if not all_choices:
#         df[col] = 0
#         return df
#
#     ranks = {choice: idx + 1 for idx, choice in enumerate(all_choices)}
#
#     def score_row(value):
#         parts = split_choices(value)
#         if not parts:
#             return 0
#         vals = [ranks[p] for p in parts if p in ranks]
#         if not vals:
#             return 0
#         return float(sum(vals)) / float(len(vals))
#
#     df[col] = df[col].apply(score_row)
#     return df
#
#
# def preprocess_excel_data(file_path: str) -> tuple:
#     """
#     Загружает все листы Excel, определяет типы, предобрабатывает каждый
#     и объединяет по user_id (outer join).
#
#     Args:
#         file_path: путь к Excel-файлу
#
#     Returns:
#         (df_merged, message): объединённый DataFrame и сводное сообщение
#     """
#     try:
#         xl = pd.ExcelFile(file_path)
#         sheet_names = xl.sheet_names
#
#         all_dfs = []
#         message_parts = []
#
#         for sheet_name in sheet_names:
#             df_headers = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
#             sheet_group = detect_sheet_group(df_headers.columns, sheet_name)
#
#             df_sheet = pd.read_excel(file_path, sheet_name=sheet_name)
#             df_processed, msg = preprocess_sheet(df_sheet, sheet_group, sheet_name)
#
#             message_parts.append(f"{sheet_name}: {msg}")
#
#             if not df_processed.empty:
#                 df_processed["_source_sheet"] = sheet_name
#                 df_processed["_sheet_type"] = sheet_group
#                 all_dfs.append(df_processed)
#
#         if all_dfs:
#             result_df = all_dfs[0]
#             for i, df_to_merge in enumerate(all_dfs[1:], 1):
#                 user_col = next(
#                     (
#                         col
#                         for col in ["user", "user_id", "VK_id", "VK"]
#                         if col in result_df.columns and col in df_to_merge.columns
#                     ),
#                     None,
#                 )
#                 if user_col:
#                     result_df = result_df.merge(
#                         df_to_merge, on=user_col, how="outer", suffixes=("", f"_{sheet_names[i]}")
#                     )
#                 else:
#                     result_df = pd.concat([result_df, df_to_merge], axis=1)
#
#             # Удаляем технические колонки
#             for col in ["_source_sheet", "_sheet_type"]:
#                 if col in result_df.columns:
#                     result_df = result_df.drop(columns=[col])
#
#             message = f"Обработано {len(sheet_names)} листов. Пропущено: {len(sheet_names) - len(all_dfs)}"
#             return result_df, message
#
#         return None, "Нет данных после обработки"
#
#     except Exception as e:
#         return None, f"Ошибка при обработке Excel: {str(e)}"
