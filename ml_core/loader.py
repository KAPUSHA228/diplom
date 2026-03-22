"""
Загрузчик Excel-опросников
Гибкая обработка разных типов листов
"""
import pandas as pd
import numpy as np


def detect_sheet_type(sheet_name: str) -> str:
    """
    Определяет тип листа по названию и структуре.
    """
    sheet_lower = sheet_name.lower()

    if 'вильямс' in sheet_lower:
        return 'williams'
    elif 'шварц' in sheet_lower:
        return 'schwartz'
    elif 'соц' in sheet_lower:
        if 'соц14' in sheet_lower or 'соцдем' in sheet_lower:
            return 'demographics'
        elif 'соц1' in sheet_lower:
            return 'personality'
        elif 'соц2' in sheet_lower:
            return 'interests'
        elif 'соц3' in sheet_lower:
            return 'digital'
        elif 'соц4' in sheet_lower or 'соц5' in sheet_lower or 'соц6' in sheet_lower:
            return 'activities'
        elif 'соц8' in sheet_lower:
            return 'attitudes'
        elif 'соц9' in sheet_lower:
            return 'grades'
        elif 'соц11' in sheet_lower or 'соц12' in sheet_lower or 'соц13' in sheet_lower:
            return 'career'
        else:
            return 'social'

    return 'unknown'


def preprocess_sheet(df: pd.DataFrame, sheet_type: str) -> tuple:
    """
    Предобработка в зависимости от типа листа.
    Возвращает (DataFrame, сообщение)
    """
    df = df.copy()
    message = f"Обработан лист типа '{sheet_type}'"

    # Удаляем пустые строки и колонки
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # Ищем ключевую колонку
    if 'user' in df.columns:
        user_col = 'user'
    elif 'user_id' in df.columns:
        user_col = 'user_id'
    elif 'VK_id' in df.columns:
        user_col = 'VK_id'
    else:
        # Если нет ключа, создаем
        df['user_id'] = range(len(df))
        user_col = 'user_id'
        message += " (создан искусственный user_id)"

    # В зависимости от типа листа
    if sheet_type == 'williams':
        # Тест креативности
        target_cols = ['Любознательность', 'Воображение', 'Сложность', 'Склонность к рискy', 'Сумма']
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    elif sheet_type == 'schwartz':
        # Ценности — Likert шкалы
        for col in df.columns:
            if col not in [user_col, 'VK', 'VK_id', 'дата']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    elif sheet_type == 'grades':
        # Успеваемость — создаем целевую переменную
        for col in df.columns:
            if 'учитесь' in col.lower() or 'сесси' in col.lower():
                grade_map = {'отлично': 2, 'хорошо': 1, 'удовлетворительно': 0}
                df['target'] = df[col].map(grade_map)
                message += " (создана целевая переменная 'target')"

    elif sheet_type in ['personality', 'interests', 'digital', 'activities', 'attitudes', 'career']:
        # Личностные опросы — Likert шкалы
        for col in df.columns:
            if col not in [user_col, 'дата']:
                # Пробуем преобразовать в числа
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Заполняем пропуски
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Удаляем колонки с датами
    for col in df.columns:
        if 'дата' in col.lower() or 'date' in col.lower():
            df = df.drop(columns=[col])

    # Удаляем дублирующиеся колонки
    df = df.loc[:, ~df.columns.duplicated()]

    return df, message


def load_sheet(file_path: str, sheet_name: str) -> tuple:
    """
    Загружает и обрабатывает один лист.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    sheet_type = detect_sheet_type(sheet_name, df)
    df_processed, msg = preprocess_sheet(df, sheet_type)
    return df_processed, sheet_type, msg


def preprocess_excel_data(file_path: str) -> tuple:
    """
    Загружает и обрабатывает весь Excel файл с опросами.
    Возвращает объединённый DataFrame и сообщение о результате.

    Parameters:
    -----------
    file_path : str
        Путь к Excel файлу

    Returns:
    --------
    tuple: (df, message)
        df - объединённый DataFrame со всеми данными
        message - сообщение о результате обработки
    """
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names

        all_dfs = []
        message_parts = []

        for sheet_name in sheet_names:
            # Загружаем лист
            df_sheet = pd.read_excel(file_path, sheet_name=sheet_name)
            sheet_type = detect_sheet_type(sheet_name)

            # Обрабатываем
            df_processed, msg = preprocess_sheet(df_sheet, sheet_type)
            message_parts.append(f"{sheet_name}: {msg}")

            # Добавляем информацию о листе
            df_processed['_source_sheet'] = sheet_name
            df_processed['_sheet_type'] = sheet_type

            all_dfs.append(df_processed)

        # Объединяем все листы по user_id
        if all_dfs:
            # Начинаем с первого листа
            result_df = all_dfs[0]

            # Постепенно мерджим остальные
            for i, df_to_merge in enumerate(all_dfs[1:], 1):
                # Определяем колонку для объединения
                user_col = None
                for col in ['user', 'user_id', 'VK_id']:
                    if col in result_df.columns and col in df_to_merge.columns:
                        user_col = col
                        break

                if user_col:
                    # Мержим по user_id
                    result_df = result_df.merge(
                        df_to_merge,
                        on=user_col,
                        how='outer',
                        suffixes=('', f'_{sheet_names[i]}')
                    )
                else:
                    # Если нет общего ключа, просто добавляем
                    result_df = pd.concat([result_df, df_to_merge], axis=1)

            # Удаляем временные колонки
            for col in ['_source_sheet', '_sheet_type']:
                if col in result_df.columns:
                    result_df = result_df.drop(columns=[col])

            message = f"Обработано {len(sheet_names)} листов: " + "; ".join(message_parts[:3]) + (
                "..." if len(message_parts) > 3 else "")

            return result_df, message
        else:
            return None, "Нет данных для обработки"

    except Exception as e:
        return None, f"Ошибка при обработке Excel: {str(e)}"