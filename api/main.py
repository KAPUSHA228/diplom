import os
import tempfile

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictRequest, PredictResponse, TaskStatus, TrainResponse
from api.routers.endpoints import router as analyze_router
from ml_core.analysis import correlation_analysis
from ml_core.models import ModelTrainer
import numpy as np

# Celery работает на Windows (в отличие от RQ).
from celery_app import app_celery as celery_app
from workers.tasks import train_model_task, shap_task

app = FastAPI(
    title="ML Analytics Service",
    description="API для ML-модуля системы мониторинга академических рисков",
    version="1.0.0",
)

# CORS для React (пока разрешаем всё, потом ограничим)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутер АРМ исследователя (новый API)
app.include_router(analyze_router)

# Инициализация трейнера (единый экземпляр)
trainer = ModelTrainer()

# ==================== Legacy Endpoints ====================


@app.get("/", response_model=dict)
async def root():
    return {
        "service": "ML Analytics Service",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ml-analytics"}


@app.post("/api/v1/ml/train_async")
async def train_async_json(data: Dict[str, Any]):
    """
    Запуск асинхронного обучения (принимает JSON массив данных из React).
    """
    import tempfile
    import pandas as pd

    if "df" not in data or not data["df"]:
        raise HTTPException(400, "Data is empty or invalid format")

    # Сохраняем JSON в CSV для Celery
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        df = pd.DataFrame(data["df"])
        df.to_csv(f, index=False)
        tmp_path = f.name

    # Запускаем задачу в Celery
    from workers.tasks import train_model_task

    task = train_model_task.delay(tmp_path)

    return {"task_id": task.id, "status": "started"}


@app.post("/api/v1/ml/predict")
async def predict(request: PredictRequest):
    """
    Синхронное предсказание для одного студента.
    Загружает последнюю сохранённую модель или обучает на лету.
    Время: <100ms (с загруженной моделью)
    """
    try:
        # Преобразуем в DataFrame
        input_df = pd.DataFrame([request.data])

        # Пытаемся загрузить последнюю сохранённую модель
        model, model_name, meta = trainer.get_best_model()

        if model is None:
            # Если сохранённой модели нет — обучаем на лету из переданных данных
            raise FileNotFoundError("No saved model found")

        # Определяем признаки из метаданных модели
        features = meta.get("features", [])
        if not features:
            # Фоллбэк: берём все числовые колонки из входных данных
            features = input_df.select_dtypes(include="number").columns.tolist()

        # Подготовка входных данных
        X = input_df[features].fillna(input_df[features].median(numeric_only=True))

        # Предсказание
        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

        return PredictResponse(
            prediction=prediction,
            probability=round(probability, 4),
            model_name=model_name,
            features_used=features,
        )

    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(404, f"Model not found or missing features: {e}")
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


def _scrub(obj):
    """Рекурсивная очистка от nan/inf/numpy для JSON."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return _scrub(obj.tolist())
    if isinstance(obj, (np.floating, np.float64)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if hasattr(obj, "to_dict"):
        return _scrub(obj.to_dict())
    return obj


@app.post("/api/v1/ml/correlation")
async def correlation(file: UploadFile = File(...), sheet_name: Optional[str] = Form(None)):
    """
    Корреляционный анализ (поддерживает CSV и Excel)
    """
    import io

    content = await file.read()
    filename = file.filename or ""
    is_excel = filename.lower().endswith((".xlsx", ".xls"))

    df = None

    if is_excel:
        # Читаем Excel (выбранный лист или первый по умолчанию)
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name if sheet_name else 0)
        except Exception:
            # Если имя листа не найдено, пробуем первый
            try:
                df = pd.read_excel(io.BytesIO(content))
            except Exception as e2:
                raise HTTPException(400, f"Ошибка чтения Excel: {str(e2)}")
    else:
        # Читаем CSV с авто-определением кодировки и разделителя
        for enc in ["utf-8", "cp1251", "latin1"]:
            try:
                # Декодируем, нормализуем newlines
                text = content.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
                # Пробуем разные разделители
                for sep in [None, ";", "\t", ","]:
                    try:
                        if sep is None:
                            df = pd.read_csv(io.StringIO(text), engine="python")
                        else:
                            df = pd.read_csv(io.StringIO(text), sep=sep)
                        if df is not None and len(df.columns) > 1:
                            break
                    except Exception:
                        df = None
                if df is not None:
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue

    if df is None:
        raise HTTPException(400, "Не удалось прочитать файл. Проверьте формат и целостность данных.")

    # Проверяем размер
    if len(df) > 10000:
        raise HTTPException(400, "Data too large, use async endpoint")

    # Ищем целевую колонку
    target_col = "risk_flag"
    if target_col not in df.columns:
        possible = [c for c in df.columns if "risk" in c.lower() or "target" in c.lower()]
        target_col = possible[0] if possible else None

    # Берём ТОЛЬКО числовые колонки для корреляции
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Исключаем служебные/ID колонки
    service_patterns = [
        "user",
        "_id",
        "vk",
        "фио",
        "фамилия",
        "имя",
        "отчество",
        "дата",
        "date",
        "группа",
        "group",
        "курс",
        "номер",
        "cluster",
        "risk_flag",
    ]
    numeric_cols = [c for c in numeric_cols if not any(p in c.lower() for p in service_patterns)]

    if not numeric_cols:
        raise HTTPException(400, f"В файле нет числовых колонок для корреляции. Доступные: {list(df.columns)}")

    # Вызываем ml_core (единая функция)
    final_target = target_col if target_col else numeric_cols[-1]
    corr_result = correlation_analysis(df, numeric_cols, final_target)

    if not corr_result:
        raise HTTPException(400, "Не удалось построить корреляционную матрицу")

    corr = corr_result["full_matrix"]

    # Генерируем Plotly heatmap
    import plotly.express as px

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Корреляционная матрица",
    )
    fig.update_layout(height=600, width=800)

    response = {
        "correlation_matrix": _scrub(corr.to_dict()),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "heatmap": _scrub(fig.to_plotly_json()),
    }
    return JSONResponse(content=response)


# Настройки для Celery (Redis URL)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@app.post("/api/v1/ml/train", response_model=TrainResponse)
async def train_model(file: UploadFile = File(...)):
    """
    Запуск обучения модели в фоне через Celery.
    Возвращает task_id для отслеживания.
    """
    # Валидация размера файла (макс 50 MB)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            400, f"File too large: {len(content) / 1024 / 1024:.1f} MB (max {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"
        )

    # Сохраняем загруженный файл временно
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Отправляем задачу в Celery
    task = train_model_task.delay(tmp_path)

    return TrainResponse(task_id=task.id, status="started")


@app.get("/api/v1/ml/train/{task_id}", response_model=TaskStatus)
async def get_train_status(task_id: str):
    """
    Получить статус задачи обучения
    """
    try:
        from celery.result import AsyncResult

        res = AsyncResult(task_id, app=celery_app)

        response = {"task_id": task_id, "status": res.status}

        if res.status == "SUCCESS":
            response["result"] = res.result
        elif res.status == "FAILURE":
            response["error"] = str(res.result)
        elif res.status == "PROGRESS":
            response.update(res.result)  # stage, progress

        return response
    except Exception:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@app.get("/api/v1/ml/tasks/{task_id}", response_model=TaskStatus)
async def get_unified_task_status(task_id: str):
    """
    Унифицированный статус для любых фоновых задач (train/shap).
    """
    return await get_train_status(task_id)


@app.post("/api/v1/ml/shap", response_model=TrainResponse)
async def generate_shap(file: UploadFile = File(...), model_id: str = "XGB"):
    """
    Запуск SHAP-объяснений в фоне через Celery.
    """
    # Валидация размера файла (макс 50 MB)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            400, f"File too large: {len(content) / 1024 / 1024:.1f} MB (max {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Отправляем задачу в Celery
    task = shap_task.delay(model_id, tmp_path)

    return TrainResponse(task_id=task.id, status="started")
