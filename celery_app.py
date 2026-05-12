"""
Настройка Celery для асинхронных задач.
Использует Redis как брокер сообщений и бэкенд результатов.
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Redis URL берем из окружения или дефолт
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app_celery = Celery(
    "ml_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["workers.tasks"],
)

# Настройки сериализации
app_celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Moscow",
    enable_utc=True,
    # Настройки для Windows (пул потоков вместо форков)
    worker_pool="solo",
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    result_expires=3600,
    # 🔥 Ключевые настройки для частых обновлений
    task_ignore_result=False,
    result_backend="redis://localhost:6379/0",
    result_persistent=True,  # сохранять результаты даже после перезапуска
)
