import json
import logging
import os

import pandas as pd
import redis
from rq import get_current_job

from ml_core.data import prepare_data_for_training
from ml_core.evaluation import generate_shap_explanations
from ml_core.features import add_composite_features, get_base_features, preprocess_data, select_features
from ml_core.models import ModelTrainer

logger = logging.getLogger(__name__)

# Подключение к Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)

trainer = ModelTrainer()


def _cleanup(data_path):
    """Удаление временного файла."""
    try:
        os.remove(data_path)
    except OSError:
        pass


def train_model_task(data_path: str):
    """
    Задача обучения модели.
    Выполняется в фоне.
    """
    job = get_current_job()

    try:
        logger.info("train_model_task: started, data_path=%s", data_path)

        # Обновляем статус
        job.meta["stage"] = "loading"
        job.meta["progress"] = 10
        job.save_meta()

        # Загружаем данные (пробуем разные кодировки)
        try:
            df = pd.read_csv(data_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding="cp1251")
        except Exception:
            df = pd.read_csv(data_path, encoding="utf-8", errors="replace")

        logger.info("train_model_task: loaded %d rows", len(df))

        job.meta["stage"] = "preprocessing"
        job.meta["progress"] = 30
        job.save_meta()

        # Ищем целевую колонку (риск или target)
        target_col = "risk_flag"
        if target_col not in df.columns:
            possible = [c for c in df.columns if "risk" in c.lower() or "target" in c.lower()]
            target_col = possible[0] if possible else None
            if target_col is None:
                raise ValueError(f"Dataset must contain 'risk_flag' or similar column. Found: {list(df.columns)}")

        df = add_composite_features(df)
        feature_cols = get_base_features(df)
        for extra in [
            "trend_grades",
            "grade_stability",
            "cognitive_load",
            "overall_satisfaction",
            "psychological_wellbeing",
            "academic_activity",
        ]:
            if extra in df.columns and extra not in feature_cols:
                feature_cols.append(extra)

        X_res, y_res = preprocess_data(df, feature_cols, target_col)
        X_sel, selected_cols = select_features(
            X_res, y_res, top_n=min(10, X_res.shape[1]), final_n=min(5, X_res.shape[1])
        )

        X_train_sel, X_test_sel, y_train, y_test = prepare_data_for_training(
            pd.concat([X_sel, pd.Series(y_res, name=target_col)], axis=1),
            feature_cols=list(X_sel.columns),
            target_col=target_col,
        )

        job.meta["stage"] = "training"
        job.meta["progress"] = 60
        job.save_meta()

        # Обучение
        best_model, best_name, metrics = trainer.train_best_model(X_train_sel, y_train, X_test_sel, y_test)
        logger.info("train_model_task: best_model=%s, metrics=%s", best_name, metrics)

        job.meta["stage"] = "saving"
        job.meta["progress"] = 90
        job.save_meta()

        # Сохраняем модель
        model_path, metadata = trainer.save_model(best_model, best_name, metrics, selected_cols)

        # Сохраняем результат в Redis
        result = {
            "status": "completed",
            "model_id": os.path.basename(model_path).replace(".pkl", ""),
            "model_path": model_path,
            "model_name": best_name,
            "metrics": metrics,
            "features": selected_cols,
        }

        # Кэшируем результат
        redis_client.setex(f"task_result:{job.id}", 3600 * 24, json.dumps(result, default=str))

        logger.info("train_model_task: completed, job_id=%s", job.id)
        _cleanup(data_path)
        return result

    except Exception as e:
        logger.exception("train_model_task: failed, job_id=%s, error=%s", job.id, str(e))
        job.meta["stage"] = "failed"
        job.meta["error"] = str(e)
        job.save_meta()
        _cleanup(data_path)
        raise


def shap_task(model_id: str, data_path: str, threshold: float = 0.5):
    """
    Генерация SHAP-объяснений
    """
    job = get_current_job()

    try:
        logger.info("shap_task: started, model_id=%s, data_path=%s", model_id, data_path)

        job.meta["stage"] = "loading_model"
        job.meta["progress"] = 20
        job.save_meta()

        # Загружаем модель
        model = trainer.load_model(model_name=model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")

        job.meta["stage"] = "loading_data"
        job.meta["progress"] = 40
        job.save_meta()

        # Загружаем данные (пробуем разные кодировки)
        try:
            df = pd.read_csv(data_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding="cp1251")
        except Exception:
            df = pd.read_csv(data_path, encoding="utf-8", errors="replace")
        logger.info("shap_task: loaded %d rows", len(df))

        job.meta["stage"] = "computing_shap"
        job.meta["progress"] = 70
        job.save_meta()

        # Генерируем объяснения
        explanations = generate_shap_explanations(model, df, df.columns.tolist(), threshold=threshold, top_n=5)

        result = {"status": "completed", "explanations": explanations, "n_students": len(df)}

        redis_client.setex(f"task_result:{job.id}", 3600 * 24, json.dumps(result, default=str))

        logger.info("shap_task: completed, job_id=%s", job.id)
        _cleanup(data_path)
        return result

    except Exception as e:
        logger.exception("shap_task: failed, job_id=%s, error=%s", job.id, str(e))
        job.meta["stage"] = "failed"
        job.meta["error"] = str(e)
        job.save_meta()
        _cleanup(data_path)
        raise
