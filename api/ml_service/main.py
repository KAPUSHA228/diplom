"""
ML Service — предсказания, обучение моделей, SHAP.
"""

import tempfile

from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any

from celery_app import app_celery as celery_app
from ml_core.error_handler import logger
from workers.tasks import train_model_task, shap_task
from ml_core.models import ModelTrainer

from ..ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    TaskStatus,
    TrainResponse,
    SubsetRequest,
    CompositeRequest,
)
from fastapi import APIRouter, HTTPException
import pandas as pd
from ml_core.analyzer import ResearchAnalyzer
from .schemas import AnalysisRequest
from shared.utils import safe_json_serializable
from shared.utils import scrub

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


app = FastAPI(title="ML Service", description="ML модели и обучение", version="1.0.0")
router = APIRouter(prefix="/api/v1/ml")

# Инициализация трейнера
trainer = ModelTrainer()


@router.post("/train_async")
async def train_async_json(data: Dict[str, Any]):
    """Запуск асинхронного обучения (принимает JSON)."""
    if "df" not in data or not data["df"]:
        raise HTTPException(400, "Data is empty or invalid format")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        df = pd.DataFrame(data["df"])
        df.to_csv(f, index=False)
        tmp_path = f.name

    task = train_model_task.delay(tmp_path)
    return {"task_id": task.id, "status": "started"}


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Синхронное предсказание для одного студента."""
    try:
        input_df = pd.DataFrame([request.data])
        model, model_name, meta = trainer.get_best_model()

        if model is None:
            raise FileNotFoundError("No saved model found")

        features = meta.get("features", [])
        if not features:
            features = input_df.select_dtypes(include="number").columns.tolist()

        X = input_df[features].fillna(input_df[features].median(numeric_only=True))

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


@router.post("/train", response_model=TrainResponse)
async def train_model(file: UploadFile = File(...)):
    """Запуск обучения модели в фоне через Celery."""
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large: {len(content) / 1024 / 1024:.1f} MB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    task = train_model_task.delay(tmp_path)
    return TrainResponse(task_id=task.id, status="started")


@router.get("/train/{task_id}", response_model=TaskStatus)
async def get_train_status(task_id: str):
    """Получить статус задачи обучения."""
    try:
        from celery.result import AsyncResult

        res = AsyncResult(task_id, app=celery_app)
        response = {"task_id": task_id, "status": res.status}

        if res.status == "SUCCESS":
            response["result"] = res.result
        elif res.status == "FAILURE":
            response["error"] = str(res.result)
        elif res.status == "PROGRESS":
            response.update(res.result)

        return response
    except Exception:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_unified_task_status(task_id: str):
    """Унифицированный статус для любых фоновых задач."""
    return await get_train_status(task_id)


@router.post("/shap", response_model=TrainResponse)
async def generate_shap(file: UploadFile = File(...), model_id: str = "XGB"):
    """Запуск SHAP-объяснений в фоне через Celery."""
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large: {len(content) / 1024 / 1024:.1f} MB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    task = shap_task.delay(model_id, tmp_path)
    return TrainResponse(task_id=task.id, status="started")


router = APIRouter(prefix="/api/v1/analyze")

# Один экземпляр анализатора на всё приложение
analyzer = ResearchAnalyzer()


@router.post("/full")
async def full_analysis(request: AnalysisRequest):
    try:
        df = pd.DataFrame(request.df)

        target = request.target_col
        if not target or target not in df.columns:
            # Ищем первую числовую колонку (исключая ID)
            num_cols = df.select_dtypes(include="number").columns
            exclude = ["user", "_id", "vk"]
            possible = [c for c in num_cols if not any(x in c.lower() for x in exclude)]
            target = possible[0] if possible else num_cols[0]
        # Передаем все параметры, которые пришли от фронтенда
        print(f"DEBUG full_analysis: n_clusters={request.n_clusters}, target={target}")
        result = analyzer.run_full_analysis(
            df=df,
            target_col=target,
            n_clusters=request.n_clusters,
            corr_threshold=request.corr_threshold,
            use_smote=request.use_smote,
            # Передаем настройки моделей из сайдбара
            use_lr=request.use_lr,
            use_rf=request.use_rf,
            use_xgb=request.use_xgb,
            optimization_metric=request.optimization_metric,
        )

        # Сериализуем графики Plotly в JSON
        def plot_to_json(fig):
            if fig is None:
                return None
            try:
                return scrub(fig.to_plotly_json())  # Чистим данные графика!
            except Exception:
                return None

        # Генерируем predictions
        predictions = []
        model_name = result.model_name or "RF"
        if result.last_y_pred is not None and analyzer.last_X_test is not None:
            model = analyzer.trainer.models.get(model_name)
            if model and hasattr(model, "predict_proba"):
                # Обернём X_test в DataFrame с именами колонок
                X_test_df = pd.DataFrame(analyzer.last_X_test, columns=result.selected_features or None)
                proba = model.predict_proba(X_test_df)[:, 1].tolist()
                y_pred_list = (
                    result.last_y_pred if isinstance(result.last_y_pred, list) else result.last_y_pred.tolist()
                )
                for i in range(len(y_pred_list)):
                    predictions.append(
                        {
                            "student_index": i,
                            "prediction": int(y_pred_list[i]),
                            "probability": round(float(proba[i]), 4),
                        }
                    )

        # result — это объект Pydantic (AnalysisResult), обращаемся через точку
        # Возвращаем СЛОВАРЬ, а не объект модели, чтобы избежать валидации Pydantic

        # Добавляем конфигурацию для воспроизводимости
        analysis_config = {
            "model_name": result.model_name or "unknown",
            "target_col": target,
            "n_samples": len(df),
            "n_features": len(result.selected_features),
            "n_clusters": request.n_clusters,
            "use_smote": request.use_smote,
            "corr_threshold": request.corr_threshold,
            "optimization_metric": request.optimization_metric,
        }

        return {
            "status": result.status,
            "message": result.message,
            "config": analysis_config,
            "metrics": scrub(result.metrics),
            "test_metrics": scrub(result.test_metrics),
            "cv_results": scrub(result.cv_results),
            "selected_features": scrub(result.selected_features),
            "cluster_profiles": scrub(result.cluster_profiles),
            "explanations": scrub(result.explanations),
            "predictions": safe_json_serializable(predictions),
            "data_with_clusters": result.data_with_clusters,
            # Передаем графики
            "fig_cm": plot_to_json(result.fig_cm),
            "fig_roc": plot_to_json(result.fig_roc),
            "fig_fi": plot_to_json(result.fig_fi),
            "fig_clusters": plot_to_json(result.fig_clusters),
            "fig_corr": plot_to_json(result.fig_corr),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@router.post("/composite/create")
async def create_composite(request: CompositeRequest):
    try:
        df = pd.DataFrame(request.df)
        df_new, score_name = analyzer.create_composite_score(df, request.feature_weights, request.score_name)

        # 1. Статистика
        stats = df_new[score_name].describe()

        # 2. Корреляции с другими числовыми колонками
        numeric_cols = df_new.select_dtypes(include="number").columns
        correlations = {}
        if score_name in df_new.columns:
            for col in numeric_cols:
                if col != score_name:
                    corr = df_new[score_name].corr(df_new[col])
                    correlations[col] = round(corr, 4) if not pd.isna(corr) else 0

        # 3. Сортируем по абсолютному значению
        sorted_corrs = dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "score_name": score_name,
            "statistics": safe_json_serializable(stats.to_dict()),
            "correlations": safe_json_serializable(sorted_corrs),
            "values": safe_json_serializable(df_new[score_name].tolist()),  # Для гистограммы
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subset/select")
async def select_subset(request: SubsetRequest):
    try:
        df = pd.DataFrame(request.df)

        print("Subset request columns:", df.columns.tolist())
        subset = analyzer.select_subset(
            df,
            condition=request.condition,
            n_samples=request.n_samples,
            by_cluster=request.by_cluster,
            random_seed=request.random_seed,
        )
        return {"count": len(subset), "data": safe_json_serializable(subset.to_dict("records"))}
    except Exception as e:
        logger.error(f"Error in select_subset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Подключаем роутер к приложению
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
