"""
Загрузчик Excel-опросников
Гибкая обработка разных типов листов
"""

import pandas as pd
from ml_core.config import optimize_dtypes
import hashlib
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

SHEET_CATEGORY_MAP = {
    # Психометрические тесты
    "вильямс ": "numeric",
    "шварц": "numeric",
    "триандис": "numeric",
    # Креативность / эссе
    "медник": "skip",
    # Социологические опросы (single_choice)
    "соц1": "single_choice",
    "соц2": "single_choice",
    "соц3": "single_choice",
    "соц4": "single_choice",
    "соц5": "single_choice",
    "соц6": "single_choice",
    "соц8": "single_choice",
    "соц9": "single_choice",
    "соц11": "single_choice",
    "соц12": "single_choice",
    "соц13": "single_choice",
    "соц14": "single_choice",
    "соц15": "single_choice",
    # Множественный выбор
    "соц7": "multiple_choice",
    "соц10": "multiple_choice",
    # Падежи и варианты написания
    "соц 1": "single_choice",
    "соц 2": "single_choice",
    "соц 3": "single_choice",
    "соц 4": "single_choice",
    "соц 5": "single_choice",
    "соц 6": "single_choice",
    "соц 7": "multiple_choice",
    "соц 8": "single_choice",
    "соц 9": "single_choice",
    "соц 10": "multiple_choice",
    "соц 11": "single_choice",
    "соц 12": "single_choice",
    "соц 13": "single_choice",
    "соц 14": "single_choice",
    "соц 15": "single_choice",
}


def get_sheet_category(sheet_name: str) -> str:
    """
    Возвращает категорию листа по его имени.
    Если лист не найден в словаре — возвращает 'unknown'.
    """
    # Нормализуем имя: убираем лишние пробелы, приводим к нижнему регистру
    normalized = sheet_name.strip().lower()

    # Прямое совпадение
    if normalized in SHEET_CATEGORY_MAP:
        return SHEET_CATEGORY_MAP[normalized]

    # Поиск по частичному совпадению (соц1 -> соц1, соц 1)
    for key, category in SHEET_CATEGORY_MAP.items():
        if key in normalized or normalized in key:
            return category

    return "unknown"


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


SHEET_TYPE_PATTERNS = {
    "category1_numeric": {
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
        elif "соц11" in sheet_lower or "соц12" in sheet_lower or "соц13" in sheet_lower:
            return "career"
        elif "соц9" in sheet_lower:
            return "grades"
        elif "соц8" in sheet_lower:
            return "attitudes"
        elif "соц4" in sheet_lower or "соц5" in sheet_lower or "соц6" in sheet_lower:
            return "activities"
        elif "соц3" in sheet_lower:
            return "digital"
        elif "соц2" in sheet_lower:
            return "interests"
        elif "соц1" in sheet_lower:
            return "personality"
        else:
            return "social"

    return "unknown"


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


def detect_sheet_group(sheet_name: str = None) -> str:
    """
    Определяет группу листа по содержимому колонок (4 категории).

    Args:
        sheet_name: опционально, имя листа для более точного определения

    Returns:
        str: группа листа ('grades', 'psychology', 'survey', 'unknown', ...)
    """
    print(f"DEBUG detect_sheet_group: type(df)={sheet_name}")
    if sheet_name:
        category = get_sheet_category(sheet_name)
        if category != "unknown":
            print(f"Лист '{sheet_name}' определён по словарю как '{category}'")
            return category

    # 2. Fallback: автоопределение по содержимому
    # df_sample = df.head(sample_size) if len(df) > sample_size else df


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
    group = detect_sheet_group(sheet_name=sheet_name)
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
            unique_vals = col_info["unique_values"]
            # Сортируем для стабильности, преобразуем в строку
            signature_str = "|".join(sorted(str(v) for v in unique_vals))
            col_info["value_signature"] = hashlib.md5(signature_str.encode()).hexdigest()
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


def preprocess_sheet(
    df: pd.DataFrame, sheet_group: str, sheet_name: str = None, mapping_config: dict = None, chunk_size: int = 10000
) -> tuple:
    """
    Предобработка одного листа Excel с поддержкой батчинга для больших данных.

    Args:
        df: исходный DataFrame
        sheet_group: группа листа (numeric, single_choice, multiple_choice, skip)
        sheet_name: имя листа
        mapping_config: словарь настроек от пользователя
        chunk_size: размер чанка для обработки (по умолчанию 50k строк)

    Returns:
        (df_processed, message): обработанный DataFrame и сообщение
    """
    print(f"🔍 PREPROCESS SHEET: sheet_group={sheet_group}, len(df)={len(df)}")
    # Если данных мало — обрабатываем сразу
    if len(df) <= chunk_size:
        return preprocess_sheet_impl(df, sheet_group, sheet_name, mapping_config)

    # Большие данные — обрабатываем чанками
    logger.info(f"Данные большие ({len(df)} строк). Обработка чанками по {chunk_size}...")

    processed_chunks = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start : start + chunk_size].copy()
        logger.info(f"Обработка чанка {i + 1}/{total_chunks} ({len(chunk)} строк)")

        processed_chunk, _ = preprocess_sheet_impl(chunk, sheet_group, sheet_name, mapping_config)
        processed_chunks.append(processed_chunk)

    # Объединяем результаты
    result = pd.concat(processed_chunks, axis=0, ignore_index=True)
    result = optimize_dtypes(result)

    message = f"Обработано {len(df)} строк чанками по {chunk_size} (всего {total_chunks} чанков)"
    return result, message


def preprocess_sheet_impl(
    df: pd.DataFrame, sheet_group: str, sheet_name: str = None, mapping_config: dict = None
) -> tuple:
    """
    Предобработка одного листа Excel.
    mapping_config: словарь настроек от пользователя {col_name: {type: ..., map: ...}}
    """
    df = df.copy()
    df = optimize_dtypes(df)

    sheet_group = normalize_sheet_group(sheet_group)
    message_parts = [f"Лист '{sheet_name}' → группа: {sheet_group}"]

    # Базовая очистка
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Поиск user_col
    user_col = next((col for col in ["user", "user_id", "VK_id", "VK"] if col in df.columns), None)
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
                    val_map = {k: int(v) for k, v in val_map.items() if v is not None and v != ""}
                    unique_vals = df[matching_col].dropna().unique()
                    existing_mapped = set(val_map.keys())
                    missing_vals = [v for v in unique_vals if v not in existing_mapped]
                    next_idx = max(val_map.values()) + 1 if val_map else 1
                    for missing in missing_vals:
                        val_map[missing] = next_idx
                        next_idx += 1
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
        if col == "student_id":
            continue
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
    print(f"🔍 ДО добавления student_id: колонки = {list(df.columns)}")
    if "student_id" not in df.columns:
        df["student_id"] = [f"student_{i:06d}" for i in range(len(df))]
        print(f"✅ Добавлен student_id, длина = {len(df)}")
    else:
        print("⚠️ student_id уже существует")

    print(f"🔍 ПОСЛЕ добавления: колонки = {list(df.columns)}")
    print(f"🔍 PREPROCESS_SHEET_IMPL: mapping_config = {mapping_config is not None}")
    # Перемещаем student_id в начало для удобства
    if "student_id" in df.columns:
        cols = ["student_id"] + [c for c in df.columns if c != "student_id"]
        df = df[cols]

    df = df.loc[:, ~df.columns.duplicated()]

    message = " | ".join(message_parts)
    print(f"🔍 ПОСЛЕ удаления служебных: колонки = {list(df.columns)}")
    print(f"🔍 ФИНАЛЬНЫЕ значения student_id (первые 5): {df['student_id'].head().tolist()}")
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
