from fastapi import APIRouter, HTTPException
from ml_core.analyzer import ResearchAnalyzer
from ml_core.imputation import handle_missing_values, detect_outliers
from ml_core.crosstab import create_crosstab, simple_crosstab
from ml_core.timeseries import (
    analyze_student_trajectory,
    detect_negative_dynamics,
    forecast_grades,
)
from ml_core.drift_detector import DataDriftDetector
from ml_core.experiment_tracker import ExperimentTracker
from ..schemas import AnalysisRequest, CompositeRequest, SubsetRequest, AnalysisResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd

router = APIRouter(prefix="/api/v1/analyze")

# Один экземпляр анализатора на всё приложение
analyzer = ResearchAnalyzer()


# ==================== Дополнительные Pydantic-схемы ====================


class ImputationRequest(BaseModel):
    df: List[Dict[str, Any]]
    strategy: str = "auto"
    threshold: float = 30.0


class CrosstabRequest(BaseModel):
    df: List[Dict[str, Any]]
    row_var: str
    col_var: str
    values: Optional[str] = None
    aggfunc: str = "count"


class TrajectoryRequest(BaseModel):
    df: List[Dict[str, Any]]
    student_id: Any
    value_col: str = "avg_grade"
    time_col: str = "semester"


class ForecastRequest(BaseModel):
    df: List[Dict[str, Any]]
    student_id: Any
    value_col: str = "avg_grade"
    time_col: str = "semester"
    future_semesters: int = 2


class DriftCheckRequest(BaseModel):
    reference_data: List[Dict[str, Any]]
    current_data: List[Dict[str, Any]]
    model_name: str = "unknown"


class ExperimentSaveRequest(BaseModel):
    name: str
    metrics: Dict[str, Any] = {}
    features: List[str] = []
    description: str = ""


@router.post("/full", response_model=AnalysisResponse)
async def full_analysis(request: AnalysisRequest):
    try:
        df = pd.DataFrame(request.df)

        result = analyzer.run_full_analysis(
            df=df,
            target_col=request.target_col,
            n_clusters=request.n_clusters,
            corr_threshold=request.corr_threshold,
            use_smote=request.use_smote,
        )

        return AnalysisResponse(
            metrics=result.get("metrics", {}),
            selected_features=result.get("selected_features", []),
            cluster_profiles=result.get("cluster_profiles", {}),
            explanations=result.get("explanations", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@router.post("/composite/create")
async def create_composite(request: CompositeRequest):
    try:
        df = pd.DataFrame(request.df)
        df_new, score_name = analyzer.create_composite_score(df, request.feature_weights, request.score_name)

        return {"score_name": score_name, "statistics": df_new[score_name].describe().to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subset/select")
async def select_subset(request: SubsetRequest):
    try:
        df = pd.DataFrame(request.df)
        subset = analyzer.select_subset(
            df, condition=request.condition, n_samples=request.n_samples, by_cluster=request.by_cluster
        )
        return {"count": len(subset), "data": subset.to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Обработка пропусков ====================


@router.post("/imputation/handle")
async def handle_imputation(request: ImputationRequest):
    """Обработка пропусков и выбросов в данных."""
    try:
        df = pd.DataFrame(request.df)
        df_clean, report = handle_missing_values(df, strategy=request.strategy, threshold=request.threshold)
        outliers = detect_outliers(df_clean)

        return {
            "data": df_clean.to_dict("records"),
            "report": report,
            "outliers": {k: v for k, v in outliers.items() if v.get("n_outliers", 0) > 0},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Кросс-таблицы ====================


@router.post("/crosstab")
async def build_crosstab(request: CrosstabRequest):
    """Построение кросс-таблицы с χ²-тестом."""
    try:
        df = pd.DataFrame(request.df)
        result = create_crosstab(df, request.row_var, request.col_var, values=request.values, aggfunc=request.aggfunc)
        return {
            "table": result["table"].to_dict(),
            "chi2_test": result.get("chi2_test"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crosstab/simple")
async def build_simple_crosstab(request: CrosstabRequest):
    """Упрощённая кросс-таблица (только таблица + χ²)."""
    try:
        df = pd.DataFrame(request.df)
        result = simple_crosstab(df, request.row_var, request.col_var)
        return {
            "table": result["table"].to_dict(),
            "chi2_test": result.get("chi2_test"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Временные ряды ====================


@router.post("/timeseries/trajectory")
async def student_trajectory(request: TrajectoryRequest):
    """Анализ траектории конкретного студента."""
    try:
        df = pd.DataFrame(request.df)
        result = analyze_student_trajectory(
            df, request.student_id, time_col=request.time_col, value_col=request.value_col
        )
        return {
            "trend": result.get("trend"),
            "status": result.get("status"),
            "first_value": result.get("first_value"),
            "last_value": result.get("last_value"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries/negative_dynamics")
async def find_negative_dynamics(request: TrajectoryRequest):
    """Поиск студентов с отрицательной динамикой."""
    try:
        df = pd.DataFrame(request.df)
        result = detect_negative_dynamics(
            df, student_id_col="student_id", time_col=request.time_col, value_col=request.value_col
        )
        return {
            "n_students_analyzed": result.get("n_students_analyzed", 0),
            "at_risk_count": len(result.get("at_risk_students", [])),
            "risk_percentage": result.get("risk_percentage", 0),
            "at_risk_students": (
                result.get("at_risk_students", []).to_dict("records")
                if isinstance(result.get("at_risk_students"), pd.DataFrame)
                else result.get("at_risk_students", [])
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries/forecast")
async def forecast_student(request: ForecastRequest):
    """Прогноз оценок студента на будущие семестры."""
    try:
        df = pd.DataFrame(request.df)
        result = forecast_grades(
            df,
            request.student_id,
            time_col=request.time_col,
            value_col=request.value_col,
            future_semesters=request.future_semesters,
        )
        return {
            "future_semesters": result.get("future_semesters", []),
            "predictions": result.get("predictions", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Дрейф данных ====================


@router.post("/drift/check")
async def check_drift(request: DriftCheckRequest):
    """Проверка дрейфа распределений между reference и current данными."""
    try:
        ref_df = pd.DataFrame(request.reference_data)
        cur_df = pd.DataFrame(request.current_data)

        detector = DataDriftDetector(ref_df, model_name=request.model_name)
        report = detector.detect_drift(cur_df)

        return {
            "overall_drift": report["overall_drift"],
            "drift_percentage": report["drift_percentage"],
            "drifted_features": report["drifted_features"],
            "recommendations": report["recommendations"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Эксперименты ====================


@router.post("/experiments/save")
async def save_experiment(request: ExperimentSaveRequest):
    """Сохранение эксперимента."""
    try:
        tracker = ExperimentTracker()
        exp_data = {
            "metrics": request.metrics,
            "features": request.features,
            "description": request.description,
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
            "experiments": experiments.to_dict("records"),
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
