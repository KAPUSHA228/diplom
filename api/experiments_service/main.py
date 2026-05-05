from fastapi import APIRouter, HTTPException
import pandas as pd
import os

from ml_core.experiment_tracker import ExperimentTracker
from shared.utils import safe_json_serializable
from api.experiments_service.schemas import ExperimentSaveRequest

router = APIRouter(prefix="/api/v1/analyze")


@router.get("/metrics/history")
async def metrics_history():
    """История метрик моделей из логов."""
    try:
        from ml_core.config import config as ml_config

        metrics_path = f"{ml_config.LOGS_DIR}/model_metrics.csv"
        if not os.path.exists(metrics_path):
            metrics_path = "logs/model_metrics.csv"
        if not os.path.exists(metrics_path):
            return {"metrics": [], "message": "Файл model_metrics.csv не найден"}

        df = pd.read_csv(metrics_path)
        return {
            "metrics": safe_json_serializable(df.to_dict("records")),
            "total": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/save")
async def save_experiment(request: ExperimentSaveRequest):
    """Сохранение эксперимента."""
    try:
        tracker = ExperimentTracker()
        exp_data = {
            "metrics": request.metrics,
            "features": request.features,
            "description": request.description,
            "config": request.config or {},
        }
        exp_id = tracker.save_experiment(request.name, exp_data)
        return {"experiment_id": exp_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/list")
async def list_experiments(limit: int = 20):
    """Список сохранённых экспериментов."""
    try:
        tracker = ExperimentTracker()
        experiments = tracker.list_experiments(limit=limit)
        return {
            "experiments": safe_json_serializable(experiments.to_dict("records")),
            "total": len(experiments),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Загрузка конкретного эксперимента по ID."""
    try:
        tracker = ExperimentTracker()
        data = tracker.load_experiment(experiment_id)
        return data
    except Exception:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
