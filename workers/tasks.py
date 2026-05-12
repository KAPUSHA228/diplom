"""
Фоновые задачи Celery для обучения моделей и SHAP-объяснений.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from celery_app import app_celery
from ml_core.features import add_composite_features, get_base_features, preprocess_data_for_smote
from ml_core.models import ModelTrainer
from ml_core.evaluation import generate_shap_explanations
from ml_core.error_handler import logger
from celery import current_task


def _cleanup(path):
    """Удаление временного файла."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {path}: {e}")


@app_celery.task(bind=True, name="workers.tasks.train_model_task")
def train_model_task(self, data_path: str):
    """
    Задача обучения модели.
    1. Загружает CSV.
    2. Создает признаки.
    3. Обучает лучшую модель.
    4. Сохраняет модель.
    """
    self.update_state(state="PROGRESS", meta={"stage": "loading", "progress": 10})

    try:
        # Чтение данных (попытка разных кодировок)
        df = None
        for encoding in ["utf-8", "cp1251"]:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            # Фоллбэк на errors='ignore'
            df = pd.read_csv(data_path, encoding="utf-8", errors="ignore")

        self.update_state(state="PROGRESS", meta={"stage": "preprocessing", "progress": 30})

        # Обработка
        df = add_composite_features(df)
        feature_cols = get_base_features(df)
        target_col = "risk_flag"

        # Если risk_flag нет, ищем другую бинарную цель
        if target_col not in df.columns:
            possible = [c for c in df.select_dtypes(include="number").columns if c not in feature_cols]
            target_col = possible[0] if possible else feature_cols[0]

        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
        y = df[target_col]

        # АВТО-БИНАРИЗАЦИЯ: если target — это числа (continuous), делим по медиане
        if y.dtype in ["float64", "int64"] and y.nunique() > 2:
            median_val = y.median()
            y = (y > median_val).astype(int)

        # SMOTE (если нужно)
        if y.nunique() > 1 and y.min() == 0 and y.max() == 1:
            try:
                X, y = preprocess_data_for_smote(X, y)
            except Exception as e:
                logger.info(f"SMOTE preprocessing failed: {e}. Continuing without SMOTE.")

        self.update_state(state="PROGRESS", meta={"stage": "training", "progress": 60})

        # Обучение
        trainer = ModelTrainer()
        model, model_name, metrics = trainer.train_best_model(X, y)

        self.update_state(state="PROGRESS", meta={"stage": "saving", "progress": 90})

        # Сохранение
        model_path, meta = trainer.save_model(model, model_name, metrics, feature_cols)

        return {
            "status": "success",
            "model_id": model_name,
            "model_path": model_path,
            "metrics": metrics.get("test", {}),
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        _cleanup(data_path)


@app_celery.task(bind=True, name="workers.tasks.shap_task")
def shap_task(self, model_id: str, data_path: str, threshold: float = 0.5):
    """
    Задача генерации SHAP-объяснений.
    """
    self.update_state(state="PROGRESS", meta={"stage": "loading_model", "progress": 20})

    try:
        trainer = ModelTrainer()
        # Загрузка модели
        model = trainer.load_model(model_name=model_id)

        if model is None:
            # Пытаемся загрузить любую последнюю модель
            model, model_name, meta = trainer.get_best_model()
            if model is None:
                raise FileNotFoundError(
                    f"Model '{model_id}' not found and no saved models available. "
                    "Please train a model first using /api/v1/ml/train."
                )
            logger.info(f"Model '{model_id}' not found, using latest: {model_name}")

        self.update_state(state="PROGRESS", meta={"stage": "loading_data", "progress": 40})

        df = pd.read_csv(data_path)

        self.update_state(state="PROGRESS", meta={"stage": "computing_shap", "progress": 70})

        feature_cols = [c for c in df.columns if c not in ["risk_flag", "user_id", "user"]]
        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

        explanations = generate_shap_explanations(model, X, feature_cols, threshold=threshold)

        return {
            "status": "success",
            "explanations": explanations,
            "n_students": len(explanations),
        }

    except Exception as e:
        logger.error(f"SHAP generation failed: {e}")
        raise
    finally:
        _cleanup(data_path)


@app_celery.task(bind=True, name="workers.tasks.full_analysis_task")
def full_analysis_task(self, data: list, params: dict):
    """
    Асинхронный полный анализ (ML + SHAP + графики).
    """

    def report_progress(stage, progress):
        print(f"🔵🔵🔵 REPORT_PROGRESS CALLED: {stage} - {progress}%")
        self.update_state(state="PROGRESS", meta={"stage": stage, "progress": progress})

    try:
        # ПРОВЕРКА 1: Вызываем report_progress ДО анализа
        report_progress("НАЧАЛО АНАЛИЗА (тест)", 1)

        df = pd.DataFrame(data)

        # ПРОВЕРКА 2: Ещё один вызов
        report_progress("ЗАГРУЗКА ДАННЫХ (тест)", 5)

        from ml_core.analyzer import ResearchAnalyzer

        analyzer = ResearchAnalyzer()

        # ПРОВЕРКА 3: Перед вызовом основного метода
        report_progress("ЗАПУСК АНАЛИЗАТОРА (тест)", 10)
        result = analyzer.run_full_analysis(
            df=df,
            target_col=params.get("target_col", "risk_flag"),
            n_clusters=params.get("n_clusters", 3),
            risk_threshold=params.get("risk_threshold", 0.5),
            corr_threshold=params.get("corr_threshold", 0.3),
            is_synthetic=params.get("is_synthetic", False),
            use_smote=params.get("use_smote", True),
            use_lr=params.get("use_lr", True),
            use_rf=params.get("use_rf", True),
            use_xgb=params.get("use_xgb", True),
            optimization_metric=params.get("optimization_metric", None),
            progress_callback=report_progress,
        )

        # Сериализуем результат (очищаем от numpy типов)
        from shared.utils import safe_json_serializable

        report_progress("АНАЛИЗ ЗАВЕРШЁН (тест)", 100)
        serialized_result = safe_json_serializable(result.__dict__)

        return {
            "status": "success",
            "result": serialized_result,
        }

    except Exception as e:
        current_task.update_state(
            state="FAILURE", meta={"stage": f"❌ Ошибка: {str(e)}", "progress": 0, "error": str(e)}
        )
        raise
