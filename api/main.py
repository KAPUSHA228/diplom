import json
import os
import tempfile
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictRequest, PredictResponse, TaskStatus, TrainResponse
from api.routers.endpoints import router as analyze_router
from ml_core.analysis import correlation_analysis
from ml_core.models import ModelTrainer

# RQ не работает на Windows (требует fork).
# На Linux/macOS импортируем, на Windows — заглушки.
if sys.platform != "win32":
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from workers.tasks import train_model_task, shap_task

    RQ_AVAILABLE = True
else:
    Redis = None
    Queue = None
    Job = None
    train_model_task = None
    shap_task = None
    RQ_AVAILABLE = False

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


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ml-analytics"}


@app.get("/")
async def root():
    return {"service": "ML Analytics Service", "docs": "/docs", "health": "/health"}


# Инициализация
trainer = ModelTrainer()


@app.post("/api/v1/ml/predict")
async def predict(request: PredictRequest):
    """
    Синхронное предсказание для одного студента.
    Время: <100ms
    """
    # Преобразуем в DataFrame
    _df = pd.DataFrame([request.data])

    # TODO: загрузить текущую модель из БД
    # пока используем заглушку
    prediction = 0
    probability = 0.75

    return PredictResponse(prediction=prediction, probability=probability)


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

    # Возвращаем target_col, если он был числовым и мы его случайно выкинули
    if target_col and target_col in df.select_dtypes(include="number").columns and target_col not in numeric_cols:
        numeric_cols.append(target_col)

    # Вызываем ml_core
    final_target = target_col if target_col else numeric_cols[-1]
    corr = correlation_analysis(df, numeric_cols, final_target)

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

    return {
        "correlation_matrix": corr.to_dict(),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "heatmap": fig.to_plotly_json(),
    }


# Подключение к Redis (только на Linux/macOS + если доступен)
if RQ_AVAILABLE:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        redis_conn = Redis.from_url(REDIS_URL)
        queue = Queue(connection=redis_conn)
    except Exception:
        redis_conn = None
        queue = None
else:
    redis_conn = None
    queue = None


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@app.post("/api/v1/ml/train", response_model=TrainResponse)
async def train_model(file: UploadFile = File(...)):
    """
    Запуск обучения модели в фоне.
    Возвращает task_id для отслеживания.
    """
    if queue is None:
        if not RQ_AVAILABLE:
            raise HTTPException(501, "Async tasks not available on Windows. Use Docker or Linux for async endpoints.")
        raise HTTPException(503, "Redis is not available")

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

    # Ставим задачу в очередь
    job = queue.enqueue(train_model_task, tmp_path, result_ttl=86400)  # хранить результат 24 часа

    return TrainResponse(task_id=job.id, status="started")


@app.get("/api/v1/ml/train/{task_id}", response_model=TaskStatus)
async def get_train_status(task_id: str):
    """
    Получить статус задачи обучения
    """
    if redis_conn is None:
        raise HTTPException(503, "Redis is not available")
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if job.is_finished:
        # Получаем результат из Redis
        result = redis_conn.get(f"task_result:{task_id}")
        if result:
            return TaskStatus(task_id=task_id, status="completed", result=json.loads(result))
        else:
            return TaskStatus(task_id=task_id, status="completed", result={"error": "result expired"})

    elif job.is_failed:
        return TaskStatus(task_id=task_id, status="failed", error=str(job.exc_info))

    elif job.is_queued:
        return TaskStatus(task_id=task_id, status="pending")

    elif job.is_started:
        return TaskStatus(
            task_id=task_id,
            status="in_progress",
            progress=job.meta.get("progress", 0),
            stage=job.meta.get("stage", "unknown"),
        )

    else:
        return TaskStatus(task_id=task_id, status="unknown")


@app.get("/api/v1/ml/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Унифицированный статус для любых фоновых задач (train/shap).
    """
    return await get_train_status(task_id)


@app.post("/api/v1/ml/shap", response_model=TrainResponse)
async def generate_shap(file: UploadFile = File(...), model_id: str = "XGB"):
    """
    Запуск SHAP-объяснений в фоне.
    """
    if queue is None:
        if not RQ_AVAILABLE:
            raise HTTPException(501, "Async tasks not available on Windows. Use Docker or Linux for async endpoints.")
        raise HTTPException(503, "Redis is not available")

    # Валидация размера файла (макс 50 MB)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            400, f"File too large: {len(content) / 1024 / 1024:.1f} MB (max {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    job = queue.enqueue(shap_task, model_id, tmp_path, result_ttl=86400)

    return TrainResponse(task_id=job.id, status="started")
