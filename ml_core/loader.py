"""
Загрузчик Excel-опросников
Гибкая обработка разных типов листов
"""

import pandas as pd
import numpy as np

SHEET_TYPE_PATTERNS = {
    "category1_numeric": {  # Вильямс, Шварц, Триандис, соц14
        "keywords": [
            "Любознательность",
            "Воображение",
            "Сложность",
            "Склонность к рискy",
            "Сумма",
            "Безопасность",
            "Конформность",
            "Традиция",
            "Самостоятельность",
            "Риск–новизна",
            "Гедонизм",
            "Достижение",
            "Власть–богатство",
            "Благожелательность",
            "Универсализм",
            "Пол",
            "Возраст",
            "Курс",
            "ВУЗ",
            "Направление подготовки",
        ],
        "min_matches": 3,
        "group": "numeric",
    },
    "category2_mednik": {
        "keywords": ["случайная;", "вечерняя;", "обратно;", "далеко;", "народная;"],
        "min_matches": 3,
        "group": "skip",
    },
    "category3_single_choice": {
        "keywords": [
            "Мне нравится работать в команде",
            "организаторские способности",
            "дисциплинированный",
            "Оптимизм",
            "Мне нравится что-то делать собственными руками",
            "учиться чему-то новому",
            "социальных сетях",
            "тематический блог",
            "зарабатываю в Интернете",
            "Научно-исследовательские проекты",
            "Спортивные соревнования",
            "Волонтерская деятельность",
            "КАК ВЫ УЧИТЕСЬ",
            "СОБИРАЕТЕСЬ ЛИ ВЫ РАБОТАТЬ",
            "В КАКОЙ СФЕРЕ ВЫ ХОТЕЛИ БЫ РАБОТАТЬ",
        ],
        "min_matches": 2,
        "group": "single_choice",
    },
    "category4_multiple_choice": {
        "keywords": ["Отметьте соответствующие варианты", "Выберите 7 – 10 самых значимых"],
        "min_matches": 1,
        "group": "multiple_choice",
    },
}


def detect_sheet_group(columns, sheet_name=None):
    """
    Определяет группу листа по содержимому колонок (4 категории).

    Args:
        columns: список имён колонок листа
        sheet_name: опционально, имя листа для более точного определения

    Returns:
        str: группа листа ('grades', 'psychology', 'survey', 'unknown', ...)
    """
    cols_lower = [str(col).strip().lower() for col in columns if not str(col).startswith("Unnamed")]

    best_group = "unknown"
    best_score = 0

    for key, pattern in SHEET_TYPE_PATTERNS.items():
        matched = sum(1 for kw in pattern["keywords"] if any(kw.lower() in col for col in cols_lower))
        if matched >= pattern["min_matches"] and matched > best_score:
            best_score = matched
            best_group = pattern["group"]

    return best_group


def detect_sheet_type_by_columns(columns, sheet_name=None):
    """
    Определяет конкретный тип листа по содержимому колонок и имени.

    Args:
        columns: список имён колонок листа
        sheet_name: опционально, имя листа

    Returns:
        str: тип листа ('williams', 'schwartz', 'demographics', ...)
    """
    # Игнорируем безымянные столбцы
    cols = [str(col).strip().lower() for col in columns if not str(col).startswith("Unnamed")]

    best_type = "unknown"
    best_score = 0

    for sheet_type, pattern in SHEET_TYPE_PATTERNS.items():
        keywords = pattern["keywords"]
        min_matches = pattern.get("min_matches", 1)
        matched = sum(1 for kw in keywords if any(kw.lower() in col for col in cols))

        if matched >= min_matches and matched > best_score:
            best_score = matched
            best_type = sheet_type

    # Fallback: если не нашли, пробуем по имени листа
    if best_type == "unknown" and sheet_name:
        best_type = detect_sheet_type(sheet_name)

    return best_type


def detect_sheet_type(sheet_name: str) -> str:
    """
    Определяет тип листа по его имени (ключевые слова: williams, schwartz, соц...).

    Args:
        sheet_name: имя листа Excel

    Returns:
        str: тип листа ('williams', 'schwartz', 'demographics', ...)
    """
    sheet_lower = sheet_name.lower()

    if "вильямс" in sheet_lower:
        return "williams"
    elif "шварц" in sheet_lower:
        return "schwartz"
    elif "соц" in sheet_lower:
        if "соц14" in sheet_lower or "соцдем" in sheet_lower:
            return "demographics"
        elif "соц1" in sheet_lower:
            return "personality"
        elif "соц2" in sheet_lower:
            return "interests"
        elif "соц3" in sheet_lower:
            return "digital"
        elif "соц4" in sheet_lower or "соц5" in sheet_lower or "соц6" in sheet_lower:
            return "activities"
        elif "соц8" in sheet_lower:
            return "attitudes"
        elif "соц9" in sheet_lower:
            return "grades"
        elif "соц11" in sheet_lower or "соц12" in sheet_lower or "соц13" in sheet_lower:
            return "career"
        else:
            return "social"

    return "unknown"


def preprocess_sheet(df: pd.DataFrame, sheet_group: str, sheet_name: str = None) -> tuple:
    """
    Предобработка одного листа Excel: удаление пустых строк/колонок,
    поиск колонки пользователя, числовая конвертация, заполнение пропусков.

    Args:
        df: DataFrame с данными листа
        sheet_group: группа листа (из detect_sheet_group)
        sheet_name: опционально, имя листа

    Returns:
        (df_processed, message): обработанный DataFrame и сообщение
    """
    df = df.copy()
    message = f"Лист '{sheet_name}' → группа: {sheet_group}"

    # Базовая очистка
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Поиск user_col
    user_col = next((col for col in ["user", "user_id", "VK_id", "VK", "student_id"] if col in df.columns), None)
    if not user_col:
        df["user_id"] = range(len(df))
        user_col = "user_id"

    # Обработка по группе
    if sheet_group == "numeric":
        for col in df.columns:
            if col not in [user_col, "дата", "Дата"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    elif sheet_group == "single_choice":
        for col in df.columns:
            if col not in [user_col, "дата", "Дата"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    elif sheet_group == "multiple_choice":
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in text_cols:
            if col not in [user_col, "дата", "Дата"]:
                prefix = f"{sheet_name.replace(' ', '_')}_" if sheet_name else ""
                df = process_multiple_choice_column(df, col, prefix=prefix)
        message += " (multiple choice → one-hot encoding)"

    elif sheet_group == "skip":
        message += " → пропущен"
        return pd.DataFrame(), message

    # Заполнение пропусков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Удаление дат
    for col in list(df.columns):
        if "дата" in str(col).lower() or "date" in str(col).lower():
            df = df.drop(columns=[col])

    df = df.loc[:, ~df.columns.duplicated()]

    return df, message


def load_excel_sheet(file_path: str, sheet_name: str) -> tuple:
    """
    Загружает, определяет тип и предобрабатывает один лист Excel.

    Args:
        file_path: путь к Excel-файлу
        sheet_name: имя листа

    Returns:
        (df_processed, message): обработанный DataFrame и сообщение
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    sheet_type = detect_sheet_type_by_columns(df.columns, sheet_name)
    df_processed, msg = preprocess_sheet(df, sheet_type)
    return df_processed, msg


def get_sheet_names(file_path: str) -> list:
    """
    Возвращает список имён листов в Excel-файле.

    Args:
        file_path: путь к Excel-файлу

    Returns:
        list[str]: имена листов
    """
    xl = pd.ExcelFile(file_path)
    return xl.sheet_names


def process_multiple_choice_column(df: pd.DataFrame, col: str, prefix: str = "") -> pd.DataFrame:
    """
    Разбивает колонку с множественным выбором на бинарные dummy-колонки.
    Например: "Волонтёрство, Спорт" → column_Волонтёрство=1, column_Спорт=1.

    Args:
        df: исходный DataFrame
        col: имя колонки с множественным выбором
        prefix: префикс для новых колонок

    Returns:
        pd.DataFrame: DataFrame с добавленными dummy-колонками
    """
    if col not in df.columns:
        return df

    # Копируем датафрейм
    df = df.copy()

    # Разделители, которые могут встречаться
    separators = [", ", "; ", ",", ";", " | "]

    # Функция разбиения
    def split_choices(text):
        if pd.isna(text):
            return []
        text = str(text).strip()
        for sep in separators:
            if sep in text:
                return [x.strip() for x in text.split(sep) if x.strip()]
        return [text] if text else []

    # Получаем все уникальные варианты
    all_choices = set()
    for val in df[col].dropna():
        all_choices.update(split_choices(val))

    all_choices = sorted(all_choices)

    if not all_choices:
        return df

    # Создаём one-hot колонки
    for choice in all_choices:
        clean_choice = choice.replace(" ", "_").replace("-", "_").replace("–", "_")
        new_col_name = f"{prefix}{clean_choice}" if prefix else clean_choice
        df[new_col_name] = df[col].apply(lambda x: 1 if choice in split_choices(x) else 0)

    # Удаляем исходную колонку
    df = df.drop(columns=[col])

    return df


def preprocess_excel_data(file_path: str) -> tuple:
    """
    Загружает все листы Excel, определяет типы, предобрабатывает каждый
    и объединяет по user_id (outer join).

    Args:
        file_path: путь к Excel-файлу

    Returns:
        (df_merged, message): объединённый DataFrame и сводное сообщение
    """
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names

        all_dfs = []
        message_parts = []

        for sheet_name in sheet_names:
            df_headers = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
            sheet_group = detect_sheet_group(df_headers.columns, sheet_name)

            df_sheet = pd.read_excel(file_path, sheet_name=sheet_name)
            df_processed, msg = preprocess_sheet(df_sheet, sheet_group, sheet_name)

            message_parts.append(f"{sheet_name}: {msg}")

            if not df_processed.empty:
                df_processed["_source_sheet"] = sheet_name
                df_processed["_sheet_type"] = sheet_group
                all_dfs.append(df_processed)

        if all_dfs:
            result_df = all_dfs[0]
            for i, df_to_merge in enumerate(all_dfs[1:], 1):
                user_col = next(
                    (
                        col
                        for col in ["user", "user_id", "VK_id", "VK"]
                        if col in result_df.columns and col in df_to_merge.columns
                    ),
                    None,
                )
                if user_col:
                    result_df = result_df.merge(
                        df_to_merge, on=user_col, how="outer", suffixes=("", f"_{sheet_names[i]}")
                    )
                else:
                    result_df = pd.concat([result_df, df_to_merge], axis=1)

            # Удаляем технические колонки
            for col in ["_source_sheet", "_sheet_type"]:
                if col in result_df.columns:
                    result_df = result_df.drop(columns=[col])

            message = f"Обработано {len(sheet_names)} листов. Пропущено: {len(sheet_names) - len(all_dfs)}"
            return result_df, message

        return None, "Нет данных после обработки"

    except Exception as e:
        return None, f"Ошибка при обработке Excel: {str(e)}"
