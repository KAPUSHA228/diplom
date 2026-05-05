"""
Модуль для логирования метрик моделей
"""

import logging
import json
from datetime import datetime
import os
import pandas as pd
from ml_core.config import config


class MLLogger:
    """Логирование ML-метрик для мониторинга"""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config.LOGS_DIR
        os.makedirs(self.log_dir, exist_ok=True)

        # Настройка логирования
        self.logger = logging.getLogger("ml_monitoring")
        self.logger.setLevel(logging.INFO)

        # Файловый обработчик
        fh = logging.FileHandler(f"{self.log_dir}/ml_events.log")
        fh.setLevel(logging.INFO)

        # Формат
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Отдельный файл для метрик моделей
        self.metrics_file = f"{self.log_dir}/model_metrics.csv"

        # Создаем CSV если нет
        if not os.path.exists(self.metrics_file):
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "model_name",
                    "f1_score",
                    "roc_auc",
                    "precision",
                    "recall",
                    "data_size",
                    "features",
                ]
            ).to_csv(self.metrics_file, index=False)

    def log_event(self, event_type, details):
        """
        Логирует произвольное событие в JSON-формате.

        Args:
            event_type: тип события (строка)
            details: dict с деталями события
        """
        self.logger.info(f"{event_type}: {json.dumps(details, default=str)}")

    def log_model_metrics(self, model_name, metrics):
        """
        Сохраняет метрики модели в CSV-файл истории и логирует событие.

        Args:
            model_name: имя модели
            metrics: dict с ключом 'test' → {f1, roc_auc, precision, recall}
        """
        # Загружаем существующие метрики
        df = pd.read_csv(self.metrics_file)

        # Добавляем новую запись
        new_row = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "f1_score": metrics.get("test", {}).get("f1"),
            "roc_auc": metrics.get("test", {}).get("roc_auc"),
            "precision": metrics.get("test", {}).get("precision"),
            "recall": metrics.get("test", {}).get("recall"),
            "data_size": metrics.get("data_size", "unknown"),
            "features": json.dumps(metrics.get("features", [])),
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.metrics_file, index=False)

        self.logger.info(f"Metrics saved for {model_name}")

    def log_features(self, features):
        """
        Логирует список использованных признаков.

        Args:
            features: list[str] имён признаков
        """
        self.log_event("features_used", {"features": features})

    def log_prediction(self, student_id, prediction, probability):
        """
        Логирует предсказание для аудита.

        Args:
            student_id: идентификатор студента
            prediction: предсказанная метка (0 или 1)
            probability: вероятность положительного класса
        """
        self.log_event(
            "prediction", {"student_id": student_id, "prediction": int(prediction), "probability": float(probability)}
        )
