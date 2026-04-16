"""Тестирование загрузки и обработки Excel-опросников."""

import numpy as np
import pandas as pd
from ml_core.loader import (
    detect_sheet_type,
    detect_sheet_group,
    get_sheet_preview,
    get_sheet_names,
    preprocess_sheet,
    load_excel_sheet,
    process_multiple_choice_column,
    preprocess_excel_data,
)


class TestExcelSheetTypeDetection:
    """Определение типа листа Excel по имени."""

    def test_detect_williams_sheet(self):
        assert detect_sheet_type("Вильямс") == "williams"

    def test_detect_schwartz_sheet(self):
        assert detect_sheet_type("Шварц") == "schwartz"

    def test_detect_demographics_sheet(self):
        assert detect_sheet_type("Соц14") == "demographics"

    def test_detect_personality_sheet(self):
        assert detect_sheet_type("Соц1") == "personality"

    def test_detect_interests_sheet(self):
        assert detect_sheet_type("Соц2") == "interests"

    def test_detect_digital_sheet(self):
        assert detect_sheet_type("Соц3") == "digital"

    def test_detect_activities_sheet(self):
        assert detect_sheet_type("Соц4") == "activities"

    def test_detect_attitudes_sheet(self):
        assert detect_sheet_type("Соц8") == "attitudes"

    def test_detect_grades_sheet(self):
        assert detect_sheet_type("Соц9") == "grades"

    def test_detect_career_sheet(self):
        """Лист 'Соц13' → career (Соц11/Соц12 попадают в personality из-за подстроки 'Соц1')."""
        assert detect_sheet_type("Соц13") == "career"

    def test_soc11_returns_personality_substring_match(self):
        """'Соц11' → career (теперь проверяется раньше, чем 'Соц1' → personality)."""
        assert detect_sheet_type("Соц11") == "career"

    def test_unknown_sheet_returns_unknown(self):
        assert detect_sheet_type("Random Sheet") == "unknown"


class TestExcelSheetGroupDetection:
    """Определение группы листа по колонкам."""

    def test_sheet_group_by_name(self):
        group = detect_sheet_group(columns=[], sheet_name="Вильямс")
        assert group in ["numeric", "unknown", "williams"]


class TestExcelSheetPreview:
    """Превью листа Excel для UI-маппинга."""

    def test_numeric_sheet_preview(self, tmp_path):
        df = pd.DataFrame({"user_id": [1, 2], "a": [10.0, 20.0], "b": [1, 2]})
        filepath = str(tmp_path / "preview.xlsx")
        df.to_excel(filepath, sheet_name="Test", index=False)
        preview = get_sheet_preview(filepath, sheet_name="Test")
        assert "detected_group" in preview
        assert "columns" in preview

    def test_single_choice_preview(self, tmp_path):
        df = pd.DataFrame({"Q1": ["Да", "Нет", "Да"]})
        filepath = str(tmp_path / "preview.xlsx")
        df.to_excel(filepath, sheet_name="Test", index=False)
        preview = get_sheet_preview(filepath, sheet_name="Test")
        assert "columns" in preview

    def test_multiple_choice_preview(self, tmp_path):
        df = pd.DataFrame({"choices": ["a;b", "b;c", "a;c"]})
        filepath = str(tmp_path / "preview.xlsx")
        df.to_excel(filepath, sheet_name="Test", index=False)
        preview = get_sheet_preview(filepath, sheet_name="Test")
        assert "detected_group" in preview


class TestExcelSheetPreprocessing:
    """Предобработка листа Excel."""

    def test_preprocess_numeric_sheet(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10, 20, 30, 40]})
        result, meta = preprocess_sheet(df, sheet_group="numeric")
        assert "a" in result.columns
        assert isinstance(result, pd.DataFrame)

    def test_preprocess_with_ordinal_mapping(self):
        df = pd.DataFrame({"level": ["Низкая", "Средняя", "Высокая"], "value": [1, 2, 3]})
        mapping = {"level": {"type": "ordinal", "mapping": {"Низкая": 1, "Средняя": 2, "Высокая": 3}}}
        result, meta = preprocess_sheet(df, sheet_group="single_choice", mapping_config=mapping)
        assert result["level"].dtype in [np.float64, np.int64, float, int]

    def test_preprocess_with_onehot_encoding(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        mapping = {"color": {"type": "onehot"}}
        result, meta = preprocess_sheet(df, sheet_group="single_choice", mapping_config=mapping)
        assert len(result.columns) > 1

    def test_preprocess_with_split(self):
        df = pd.DataFrame({"choices": ["a;b", "b;c", "a"]})
        mapping = {"choices": {"type": "split", "separator": ";"}}
        result, meta = preprocess_sheet(df, sheet_group="multiple_choice", mapping_config=mapping)
        assert len(result.columns) >= 2

    def test_preprocess_skip_sheet(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result, meta = preprocess_sheet(df, sheet_group="skip")
        assert result is None or len(result) == 0


class TestMultipleChoiceColumnProcessing:
    """Обработка колонки с множественным выбором."""

    def test_semicolon_separator(self):
        df = pd.DataFrame({"choices": ["a;b", "b;c", "a"]})
        result = process_multiple_choice_column(df, "choices")
        assert len(result.columns) > 1

    def test_comma_separator(self):
        df = pd.DataFrame({"choices": ["a,b", "b,c", "a,c"]})
        result = process_multiple_choice_column(df, "choices")
        assert len(result.columns) > 1


class TestExcelSheetLoading:
    """Загрузка отдельного листа Excel."""

    def test_load_numeric_sheet(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        filepath = str(tmp_path / "test.xlsx")
        df.to_excel(filepath, sheet_name="Вильямс", index=False)
        data, group = load_excel_sheet(filepath, sheet_name="Вильямс")
        assert len(data) == 2
        assert "williams" in group.lower() or "numeric" in group.lower()

    def test_load_unknown_sheet_type(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        filepath = str(tmp_path / "test.xlsx")
        df.to_excel(filepath, sheet_name="Random", index=False)
        data, group = load_excel_sheet(filepath, sheet_name="Random")
        assert len(data) == 2


class TestExcelMultiSheetProcessing:
    """Обработка многлистовых Excel-файлов."""

    def test_merge_multiple_sheets(self, tmp_path):
        df1 = pd.DataFrame({"user_id": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"user_id": [1, 2], "b": [30, 40]})
        filepath = str(tmp_path / "survey.xlsx")
        with pd.ExcelWriter(filepath) as writer:
            df1.to_excel(writer, sheet_name="Вильямс", index=False)
            df2.to_excel(writer, sheet_name="Шварц", index=False)
        merged, details = preprocess_excel_data(filepath)
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) >= 2

    def test_single_sheet_processing(self, tmp_path):
        df = pd.DataFrame({"user_id": [1, 2], "a": [10, 20]})
        filepath = str(tmp_path / "single.xlsx")
        df.to_excel(filepath, sheet_name="Вильямс", index=False)
        merged, details = preprocess_excel_data(filepath)
        assert len(merged) == 2


class TestExcelSheetNames:
    """Получение имён листов из Excel-файла."""

    def test_get_all_sheet_names(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        filepath = str(tmp_path / "multi.xlsx")
        with pd.ExcelWriter(filepath) as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)
            df.to_excel(writer, sheet_name="Sheet2", index=False)
        names = get_sheet_names(filepath)
        assert "Sheet1" in names
        assert "Sheet2" in names
