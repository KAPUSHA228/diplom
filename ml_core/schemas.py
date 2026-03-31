from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import pandas as pd


class AnalysisRequest(BaseModel):
    df: Any  # будем передавать как dict или использовать custom validator
    target_col: str
    n_clusters: int = 3
    risk_threshold: float = 0.5
    use_hp_tuning: bool = False


class CompositeScoreRequest(BaseModel):
    df: Any
    feature_weights: Dict[str, float]
    score_name: str = "custom_score"
    normalize: bool = True


class AnalysisResult(BaseModel):
    metrics: Dict[str, Any]
    test_metrics: Dict[str, float]
    selected_features: List[str]
    cluster_profiles: Dict[str, Any]
    explanations: List[Dict[str, Any]]
    drift_report: Optional[Dict[str, Any]] = None
    status: str = "success"
    message: Optional[str] = None
    composite_scores_created: Optional[List[str]] = None
    cv_results: dict = Field(default_factory=dict)


class TrajectoryRequest(BaseModel):
    """Запрос на анализ траектории студента"""
    df: Any
    student_id: Any
    value_col: str = "avg_grade"
    time_col: str = "semester"
