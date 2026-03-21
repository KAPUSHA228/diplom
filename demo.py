"""
Анализатор структуры Excel-файла с опросами
Выводит названия листов и названия столбцов, не показывая содержимое
"""

import pandas as pd
import openpyxl
from pathlib import Path


def analyze_excel_structure(file_path, max_rows_preview=0, max_cols_preview=None):
    """
    Анализирует структуру Excel-файла без отображения персональных данных

    Args:
        file_path: путь к Excel файлу
        max_rows_preview: количество строк для предпросмотра (0 - только заголовки)
        max_cols_preview: максимальное количество столбцов для вывода
    """

    print(f"\n{'=' * 60}")
    print(f"📊 Анализ файла: {Path(file_path).name}")
    print(f"{'=' * 60}\n")

    # Получаем все листы
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    print(f"📑 Найдено листов: {len(sheet_names)}")
    print(f"📑 Список листов: {', '.join(sheet_names)}\n")

    for sheet_name in sheet_names:
        print(f"{'─' * 50}")
        print(f"📄 Лист: {sheet_name}")
        print(f"{'─' * 50}")

        try:
            # Читаем только заголовки
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)

            columns = list(df.columns)
            len_columns = len(columns)+1
            print(f"📌 Количество столбцов: {len_columns}")

            # Выводим названия столбцов
            print(f"📌 Названия столбцов:")
            for i, col in enumerate(columns, 1):
                print(f"   {i:3}. {col}")

            # Если нужно посмотреть типы данных
            if max_rows_preview > 0:
                df_preview = pd.read_excel(file_path, sheet_name=sheet_name,
                                           nrows=max_rows_preview)
                print(f"\n📊 Типы данных (первые {max_rows_preview} строк):")
                for col in columns[:max_cols_preview] if max_cols_preview else columns:
                    print(f"   {col}: {df_preview[col].dtype}")

            # Количество строк (если нужно)
            try:
                wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                ws = wb[sheet_name]
                total_rows = ws.max_row
                wb.close()
                print(f"\n📊 Всего строк в листе (с заголовком): {total_rows}")
            except:
                pass

        except Exception as e:
            print(f"⚠️ Ошибка при чтении листа {sheet_name}: {e}")

        print()  # пустая строка для разделения


def get_column_mapping(file_path, sheet_name):
    """
    Получить маппинг названий столбцов (русский -> английский)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)

    print(f"\nМаппинг столбцов для листа {sheet_name}:")
    print(f"{'Русское название':<30} -> {'Английское название'}")
    print(f"{'-' * 55}")

    for col in df.columns:
        # Генерируем английское название
        eng_name = (col.lower()
                    .replace(' ', '_')
                    .replace('(', '')
                    .replace(')', '')
                    .replace('/', '_')
                    .replace('-', '_')
                    .replace('№', 'n')
                    .replace('%', 'percent'))

        # Ограничиваем длину
        if len(eng_name) > 30:
            eng_name = eng_name[:30]

        print(f"{col:<30} -> {eng_name}")


# Пример использования
if __name__ == "__main__":
    # Укажите путь к вашему файлу
    FILE_PATH = "ФИ Креативность 2024.xlsx"

    # 1. Полная структура (только заголовки, без данных)
    analyze_excel_structure(FILE_PATH, max_rows_preview=0)

    # 2. Если нужно посмотреть типы данных (без содержимого)
    analyze_excel_structure(FILE_PATH, max_rows_preview=2, max_cols_preview=10)

    # 3. Получить маппинг для конкретного листа
    #get_column_mapping(FILE_PATH, "Вильямс")
