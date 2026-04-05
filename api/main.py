import json
import os
import tempfile

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from rq import Queue
from rq.job import Job

from api.schemas import PredictRequest, PredictResponse, TaskStatus, TrainResponse
from ml_core.analysis import correlation_analysis
from workers.tasks import train_model_task, shap_task
from ml_core.models import ModelTrainer

app = FastAPI(
    title="ML Analytics Service",
    description="API для ML-модуля системы мониторинга академических рисков",
    version="1.0.0"
)

# CORS для React (пока разрешаем всё, потом ограничим)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ml-analytics"}

@app.get("/")
async def root():
    return {
        "service": "ML Analytics Service",
        "docs": "/docs",
        "health": "/health"
    }




# Инициализация
trainer = ModelTrainer()


@app.post("/api/ml/predict")
async def predict(request: PredictRequest):
    """
    Синхронное предсказание для одного студента.
    Время: <100ms
    """
    # Преобразуем в DataFrame
    df = pd.DataFrame([request.data])

    # TODO: загрузить текущую модель из БД
    # пока используем заглушку
    prediction = 0
    probability = 0.75

    return PredictResponse(
        prediction=prediction,
        probability=probability
    )


@app.post("/api/ml/correlation")
async def correlation(file: UploadFile = File(...)):
    """
    Корреляционный анализ (синхронно для небольших данных)
    """
    df = pd.read_csv(file.file)

    # Проверяем размер (если >10000 строк — возвращаем предупреждение)
    if len(df) > 10000:
        raise HTTPException(400, "Data too large, use async endpoint")

    # Вызываем ml_core
    corr = correlation_analysis(df, df.columns.tolist(), 'risk_flag')

    return {
        "correlation_matrix": corr.to_dict(),
        "n_rows": len(df),
        "n_columns": len(df.columns)
    }


# Подключение к Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(connection=redis_conn)


@app.post("/api/ml/train", response_model=TrainResponse)
async def train_model(file: UploadFile = File(...)):
    """
    Запуск обучения модели в фоне.
    Возвращает task_id для отслеживания.
    """
    # Сохраняем загруженный файл временно
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Ставим задачу в очередь
    job = queue.enqueue(
        train_model_task,
        tmp_path,
        result_ttl=86400  # хранить результат 24 часа
    )

    return TrainResponse(
        task_id=job.id,
        status="started"
    )


@app.get("/api/ml/train/{task_id}", response_model=TaskStatus)
async def get_train_status(task_id: str):
    """
    Получить статус задачи обучения
    """
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if job.is_finished:
        # Получаем результат из Redis
        result = redis_conn.get(f'task_result:{task_id}')
        if result:
            return TaskStatus(
                task_id=task_id,
                status="completed",
                result=json.loads(result)
            )
        else:
            return TaskStatus(
                task_id=task_id,
                status="completed",
                result={"error": "result expired"}
            )

    elif job.is_failed:
        return TaskStatus(
            task_id=task_id,
            status="failed",
            error=str(job.exc_info)
        )

    elif job.is_queued:
        return TaskStatus(
            task_id=task_id,
            status="pending"
        )

    elif job.is_started:
        return TaskStatus(
            task_id=task_id,
            status="in_progress",
            progress=job.meta.get('progress', 0),
            stage=job.meta.get('stage', 'unknown')
        )

    else:
        return TaskStatus(
            task_id=task_id,
            status="unknown"
        )


@app.get("/api/ml/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Унифицированный статус для любых фоновых задач (train/shap).
    """
    return await get_train_status(task_id)


@app.post("/api/ml/shap", response_model=TrainResponse)
async def generate_shap(file: UploadFile = File(...), model_id: str = "XGB"):
    """
    Запуск SHAP-объяснений в фоне.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job = queue.enqueue(
        shap_task,
        model_id,
        tmp_path,
        result_ttl=86400
    )

    return TrainResponse(
        task_id=job.id,
        status="started"
    )

