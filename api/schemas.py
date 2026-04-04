from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AnalysisRequest(BaseModel):
    """Запрос на полный анализ"""
    df: List[Dict[str, Any]]                    # данные в виде списка словарей
    target_col: str = "risk_flag"
    n_clusters: int = Field(3, ge=2, le=8)
    corr_threshold: float = Field(0.3, ge=0.0, le=1.0)
    use_smote: bool = True
    n_features_to_select: int = Field(7, ge=3, le=15)


class CompositeRequest(BaseModel):
    """Запрос на создание композитной оценки"""
    df: List[Dict[str, Any]]
    feature_weights: Dict[str, float]
    score_name: str = "custom_score"


class SubsetRequest(BaseModel):
    """Запрос на выделение подмножества респондентов"""
    df: List[Dict[str, Any]]
    condition: Optional[str] = None          # pandas query строка
    n_samples: Optional[int] = None
    by_cluster: Optional[int] = None


class AnalysisResponse(BaseModel):
    """Ответ от АРМ"""
    status: str = "success"
    message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    selected_features: List[str] = Field(default_factory=list)
    cluster_profiles: Dict[str, Any] = Field(default_factory=dict)
    explanations: List[Dict[str, Any]] = Field(default_factory=list)
    data: Optional[List[Dict[str, Any]]] = None   # обработанные данные, если нужно
