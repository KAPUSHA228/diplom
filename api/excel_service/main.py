# api/excel_service/main.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import pandas as pd
from typing import Optional

from ml_core.loader import get_sheet_preview
from shared.utils import safe_json_serializable
from .schemas import ImputationRequest
import numpy as np

router = APIRouter(prefix="/api/v1/analyze")


@router.post("/excel/preview")
async def excel_preview(file: UploadFile = File(...), sheet_name: str = Form("0")):
    """
    Возвращает метаданные листа: типы колонок и уникальные значения для настройки маппинга.
    """
    import tempfile
    import os

    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Если передан индекс (цифра), находим реальное имя листа
        xl = pd.ExcelFile(tmp_path)
        all_sheets = xl.sheet_names
        target_name = sheet_name

        print(f"DEBUG excel_preview: получен sheet_name='{sheet_name}'")
        print(f"DEBUG excel_preview: все листы в файле: {all_sheets}")

        if sheet_name.isdigit():
            idx = int(sheet_name)
            if 0 <= idx < len(all_sheets):
                target_name = all_sheets[idx]
            else:
                raise ValueError(f"Лист с индексом {idx} не найден")
        else:
            # Проверяем точное совпадение
            if sheet_name not in all_sheets:
                # Пробуем найти по частичному совпадению (без учёта регистра и пробелов)
                normalized = sheet_name.strip().lower()
                found = [s for s in all_sheets if s.strip().lower() == normalized]
                if found:
                    target_name = found[0]
                    print(f"DEBUG excel_preview: найдено частичное совпадение: '{sheet_name}' → '{target_name}'")
                else:
                    # Если не нашли — ищем по началу имени (первые символы)
                    found_prefix = [s for s in all_sheets if s.lower().startswith(normalized[:6])]
                    if found_prefix:
                        target_name = found_prefix[0]
                        print(f"DEBUG excel_preview: найдено по префиксу: '{sheet_name}' → '{target_name}'")
                    else:
                        print(
                            f"DEBUG excel_preview: лист '{sheet_name}' НЕ НАЙДЕН! Доступны: {all_sheets}. Использую первый лист '{all_sheets[0]}'"
                        )
                        target_name = all_sheets[0]
            else:
                print(f"DEBUG excel_preview: точное совпадение '{sheet_name}'")

        print(f"DEBUG excel_preview: ФИНАЛЬНОЕ target_name='{target_name}'")

        preview = get_sheet_preview(tmp_path, target_name)
        xl.close()  # Обязательно закрываем файл перед удалением!
        return safe_json_serializable(preview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка превью: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass  # На Windows иногда глючит блокировка


@router.post("/excel/process")
async def excel_process(
    file: UploadFile = File(...),
    sheet_name: str = Form("0"),
    sheet_group: str = Form("numeric"),
    mapping_config: Optional[str] = Form(None),
):
    """
    Обрабатывает лист Excel с учетом пользовательских настроек маппинга.
    """
    import tempfile
    import os
    import json

    config = None
    if mapping_config:
        try:
            config = json.loads(mapping_config)
        except Exception:
            pass

    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        xl = pd.ExcelFile(tmp_path)
        target_name = sheet_name
        if sheet_name.isdigit():
            idx = int(sheet_name)
            if 0 <= idx < len(xl.sheet_names):
                target_name = xl.sheet_names[idx]
            else:
                raise ValueError(f"Лист с индексом {idx} не найден")
        xl.close()

        from ml_core.loader import preprocess_sheet

        # Читаем файл уже по правильному имени
        df = pd.read_excel(tmp_path, sheet_name=target_name)

        # Передаем реальное имя листа в процессор
        df_processed, msg = preprocess_sheet(df, sheet_group, target_name, mapping_config=config)

        # Заменяем NaN и inf перед сериализацией
        df_clean = df_processed.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.where(df_clean.notna(), None)

        return {"message": msg, "data": safe_json_serializable(df_clean.to_dict("records")), "rows": len(df_processed)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.post("/imputation/handle")
async def handle_imputation(request: ImputationRequest):
    """Обработка пропусков и выбросов в данных."""
    try:
        from ml_core.imputation import handle_missing_values, detect_outliers

        df = pd.DataFrame(request.df)
        df_clean, report = handle_missing_values(df, strategy=request.strategy, threshold=request.threshold)
        outliers = detect_outliers(df_clean)

        return {
            "data": safe_json_serializable(df_clean.to_dict("records")),
            "report": safe_json_serializable(report),
            "outliers": safe_json_serializable({k: v for k, v in outliers.items() if v.get("n_outliers", 0) > 0}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
