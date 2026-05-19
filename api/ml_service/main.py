"""
ML Service — предсказания, обучение моделей, SHAP.
"""

from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any
import ray
from ray.exceptions import RayTaskError
from ml_core.error_handler import logger
from workers.tasks import train_model_task, shap_task, init_ray, full_analysis_task, ProgressActor, correlation_task
from ml_core.models import ModelTrainer
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException
import pandas as pd
from ml_core.analyzer import ResearchAnalyzer
from .schemas import (
    AnalysisRequest,
    PredictRequest,
    PredictResponse,
    TaskStatus,
    TrainResponse,
    SubsetRequest,
    CompositeRequest,
)
from shared.utils import safe_json_serializable
from shared.utils import scrub
from api.ml_service.websocket import router as websocket_router

# Хранилище активных задач (для отмены)
active_tasks = {}
active_progress = {}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
CACHE_TTL = 3600


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Код, который выполняется ПРИ ЗАПУСКЕ (Startup) ---
    print("🟢 Приложение запускается...")
    # Инициализируем Ray
    init_ray()
    progress_actor = ProgressActor.remote()
    app.state.progress_actor = progress_actor
    print("✅ Ray инициализирован")

    # Передаём управление приложению
    yield

    # --- Код, который выполняется ПРИ ОСТАНОВКЕ (Shutdown) ---
    print("🔴 Приложение останавливается...")
    # Здесь можно добавить код для graceful shutdown Ray, если нужно
    # например, ray.shutdown() — но обычно не требуется, т.к. Ray живёт отдельно


app = FastAPI(title="ML Service", description="ML модели и обучение", version="1.0.0", lifespan=lifespan)

# Два роутера: не переиспользовать одну переменную — иначе теряются маршруты первого префикса.
router_ml = APIRouter(prefix="/api/v1/ml")
router_analyze_ml = APIRouter(prefix="/api/v1/analyze")

# Инициализация трейнера
trainer = ModelTrainer()


@router_ml.post("/train_async")
async def train_async_json(data: Dict[str, Any]):
    """Запуск асинхронного обучения через Ray"""
    if "df" not in data or not data["df"]:
        raise HTTPException(400, "Data is empty or invalid format")
    target_col = data.get("target_col", None)
    try:
        task_ref = train_model_task.remote(data["df"], {"target_col": target_col})
        task_id = task_ref.hex()
        active_tasks[task_id] = task_ref
        return {"task_id": task_id, "status": "started"}
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(500, str(e))


@router_ml.post("/predict", response_model=PredictResponse)
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


@router_ml.post("/train", response_model=TrainResponse)
async def train_model(file: UploadFile = File(...)):
    """Запуск обучения модели в фоне через Ray."""
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large: {len(content) / 1024 / 1024:.1f} MB")

    import io

    df = pd.read_csv(io.BytesIO(content))
    df_data = df.to_dict(orient="records")

    task_ref = train_model_task.remote(df_data, {})
    task_id = task_ref.hex()
    active_tasks[task_id] = task_ref
    return TrainResponse(task_id=task_id, status="started")


@router_ml.get("/train/{task_id}", response_model=TaskStatus)
async def get_train_status(task_id: str):
    """Получить статус задачи обучения."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    obj_ref = active_tasks[task_id]
    ready, _ = ray.wait([obj_ref], timeout=0)

    if ready:
        try:
            result = ray.get(obj_ref)
            del active_tasks[task_id]
            return {"task_id": task_id, "status": "SUCCESS", "result": result}
        except Exception as e:
            del active_tasks[task_id]
            return {"task_id": task_id, "status": "FAILURE", "error": str(e)}

    return {"task_id": task_id, "status": "PROGRESS"}


@router_ml.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_unified_task_status(task_id: str):
    """Унифицированный статус для любых фоновых задач."""
    return await get_train_status(task_id)


@router_ml.post("/shap", response_model=TrainResponse)
async def generate_shap(file: UploadFile = File(...), model_id: str = "XGB"):
    """Запуск SHAP-объяснений в фоне через Celery."""
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large: {len(content) / 1024 / 1024:.1f} MB")

    # Читаем CSV из файла
    import io

    df = pd.read_csv(io.BytesIO(content))
    df_data = df.to_dict(orient="records")

    task_ref = shap_task.remote(model_id, df_data)
    task_id = str(task_ref)
    active_tasks[task_id] = task_ref
    return TrainResponse(task_id=task_id, status="started")


# Один экземпляр анализатора на всё приложение
analyzer = ResearchAnalyzer()


@router_analyze_ml.post("/full")
async def full_analysis(request: AnalysisRequest):
    try:
        df = pd.DataFrame(request.df)

        target = request.target_col
        opt_metric = request.optimization_metric

        if not opt_metric or opt_metric == "default" or opt_metric == "null" or opt_metric == "Null":
            display_metric = "F1 (по умолчанию)"
        elif opt_metric == "f1":
            display_metric = "F1-score"
        elif opt_metric == "roc_auc":
            display_metric = "ROC-AUC"
        elif opt_metric == "precision":
            display_metric = "Precision"
        elif opt_metric == "recall":
            display_metric = "Recall"
        else:
            display_metric = opt_metric

        if not target or target not in df.columns:
            # Ищем первую числовую колонку (исключая ID)
            num_cols = df.select_dtypes(include="number").columns
            exclude = ["user", "_id", "vk"]
            possible = [c for c in num_cols if not any(x in c.lower() for x in exclude)]
            target = possible[0] if possible else num_cols[0]
        # Передаем все параметры, которые пришли от фронтенда
        print(
            f"DEBUG full_analysis: n_clusters={request.n_clusters}, target={target}, metrics={display_metric}, use_hp_tuning: {request.use_hp_tuning}, n_iter_tuning: {request.n_iter_tuning}"
        )
        result = analyzer.run_full_analysis(
            df=df,
            target_col=target,
            n_clusters=request.n_clusters,
            corr_threshold=request.corr_threshold,
            risk_threshold=request.risk_threshold,
            use_smote=request.use_smote,
            # Передаем настройки моделей из сайдбара
            use_lr=request.use_lr,
            use_rf=request.use_rf,
            use_xgb=request.use_xgb,
            optimization_metric=display_metric,
            n_features_to_select=getattr(request, "n_features_to_select", None),
            shap_top_n=getattr(request, "shap_top_n", 5),
            use_hp_tuning=getattr(request, "use_hp_tuning", False),
            n_iter_tuning=getattr(request, "n_iter_tuning", 20),
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
            "optimization_metric": display_metric,
        }
        print(f"🔵🔵🔵 ПЕРЕД RETURN: target = {target}")
        print(f"🔵🔵🔵 ТИП target = {type(target)}")
        response = {
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
            "target_col": target,
            "fig_cm": plot_to_json(result.fig_cm),
            "fig_roc": plot_to_json(result.fig_roc),
            "fig_fi": plot_to_json(result.fig_fi),
            "fig_clusters": plot_to_json(result.fig_clusters),
            "fig_corr": plot_to_json(result.fig_corr),
        }

        print(f"🔵🔵🔵 КЛЮЧИ ОТВЕТА: {list(response.keys())}")
        print(f"🔵🔵🔵 target_col в ответе: {response.get('target_col')}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@router_analyze_ml.post("/composite/create")
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


@router_analyze_ml.post("/subset/select")
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


@router_ml.post("/full_async")
async def full_analysis_async(request: AnalysisRequest):
    """Запуск полного анализа через Ray"""
    try:
        # Создаём Actor для прогресса
        progress_actor = ProgressActor.remote()
        print("🔵 в async [BACKEND] Получен запрос:")
        print(f"  - use_hp_tuning: {request.use_hp_tuning}")
        print(f"  - n_iter_tuning: {request.n_iter_tuning}")
        print(f"  - target_col: {request.target_col}")
        # Запускаем задачу
        obj_ref = full_analysis_task.remote(request.df, request.dict(), progress_actor)

        # Генерируем ID (можно использовать hex из ObjectRef)
        task_id = obj_ref.hex()
        active_tasks[task_id] = obj_ref
        active_progress[task_id] = progress_actor

        return {"task_id": task_id, "status": "started"}
    except Exception as e:
        logger.error(f"Failed to start task: {e}")
        raise HTTPException(500, str(e))


@router_ml.post("/correlation_async")
async def correlation_async(data: Dict[str, Any]):
    """Асинхронная корреляция для больших данных"""
    try:
        df_data = data.get("df")
        target_col = data.get("target_col")

        if not df_data:
            raise HTTPException(400, "No data provided")

        task_ref = correlation_task.remote(df_data, target_col)
        task_id = task_ref.hex()
        active_tasks[task_id] = task_ref

        return {"task_id": task_id, "status": "started"}

    except Exception as e:
        logger.error(f"Failed to start correlation: {e}")
        raise HTTPException(500, str(e))


@router_ml.get("/correlation_async/{task_id}")
async def get_correlation_status(task_id: str):
    """Статус асинхронной корреляции"""
    if task_id not in active_tasks:
        return {"task_id": task_id, "status": "UNKNOWN"}

    obj_ref = active_tasks[task_id]
    ready, _ = ray.wait([obj_ref], timeout=0)

    if ready:
        try:
            result = ray.get(obj_ref)
            del active_tasks[task_id]
            return {"task_id": task_id, "status": "SUCCESS", "result": result}
        except Exception as e:
            del active_tasks[task_id]
            return {"task_id": task_id, "status": "FAILURE", "error": str(e)}

    return {"task_id": task_id, "status": "PROGRESS"}


@router_ml.get("/full_async/{task_id}")
async def get_full_analysis_status(task_id: str):
    """Статус фонового анализа."""
    if task_id not in active_tasks:
        return {"task_id": task_id, "status": "UNKNOWN"}

    obj_ref = active_tasks[task_id]

    # Проверяем готовность
    ready_refs, _ = ray.wait([obj_ref], timeout=0)

    # Получаем прогресс из Actor
    progress = None
    if task_id in active_progress:
        try:
            progress = ray.get(active_progress[task_id].get.remote())
        except Exception:
            pass

    if ready_refs:
        import time

        start_get = time.time()
        try:
            result = ray.get(obj_ref)
            print(f"🔵 RAY.GET ЗАВЕРШЕН за {time.time() - start_get:.2f} сек")
            print(f"🔵 РАЗМЕР РЕЗУЛЬТАТА: {len(str(result)) / 1024 / 1024:.2f} MB")
            del active_tasks[task_id]
            if task_id in active_progress:
                del active_progress[task_id]
            return {"task_id": task_id, "status": "SUCCESS", "result": result}
        except RayTaskError as e:
            del active_tasks[task_id]
            if task_id in active_progress:
                del active_progress[task_id]
            return {"task_id": task_id, "status": "FAILURE", "error": str(e)}

    # Задача ещё выполняется
    return {
        "task_id": task_id,
        "status": "PROGRESS",
        "stage": progress.get("stage", "Выполняется...") if progress else "Выполняется...",
        "progress": progress.get("progress", 0) if progress else 0,
    }


@router_ml.post("/full_async/{task_id}/cancel")
async def cancel_full_analysis(task_id: str):
    """Отмена задачи"""
    if task_id in active_tasks:
        ray.cancel(active_tasks[task_id])
        del active_tasks[task_id]
        if task_id in active_progress:
            del active_progress[task_id]
        return {"status": "cancelled", "message": f"Task {task_id} cancelled"}
    return {"status": "not_found", "message": "Task not found"}


app.include_router(router_ml)
app.include_router(router_analyze_ml)
app.include_router(websocket_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
