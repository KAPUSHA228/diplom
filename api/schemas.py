from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AnalysisRequest(BaseModel):
    """Запрос на полный анализ"""

    df: List[Dict[str, Any]]  # данные в виде списка словарей
    target_col: str = "risk_flag"
    n_clusters: int = Field(3, ge=2, le=8)
    corr_threshold: float = Field(0.3, ge=0.0, le=1.0)
    use_smote: bool = True
    n_features_to_select: int = Field(7, ge=3, le=15)

    # Новые поля из React-сайдбара
    use_hp_tuning: bool = False
    optimization_metric: Optional[str] = None
    shap_top_n: int = Field(5, ge=1, le=20)
    use_lr: bool = True
    use_rf: bool = True
    use_xgb: bool = True


class CompositeRequest(BaseModel):
    """Запрос на создание композитной оценки"""

    df: List[Dict[str, Any]]
    feature_weights: Dict[str, float]
    score_name: str = "custom_score"


class SubsetRequest(BaseModel):
    """Запрос на выделение подмножества респондентов"""

    df: List[Dict[str, Any]]
    condition: Optional[str] = None  # pandas query строка
    n_samples: Optional[int] = None
    by_cluster: Optional[int] = None
    random_seed: Optional[int] = 42


class FeatureCombinationRequest(BaseModel):
    """Запрос на создание комбинированных признаков"""

    df: List[Dict[str, Any]]
    numerical_cols: Optional[List[str]] = None
    text_cols: Optional[List[str]] = None
    max_pairs: int = 15
    target_col: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Ответ от АРМ"""

    status: str = "success"
    message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    cv_results: Dict[str, Any] = Field(default_factory=dict)
    selected_features: List[str] = Field(default_factory=list)
    cluster_profiles: Dict[str, Any] = Field(default_factory=dict)
    explanations: List[Dict[str, Any]] = Field(default_factory=list)

    # Данные для экспорта (предсказания для студентов)
    predictions: Optional[List[Dict[str, Any]]] = None

    # Графики (в виде JSON-структур Plotly)
    fig_cm: Optional[Dict[str, Any]] = None
    fig_roc: Optional[Dict[str, Any]] = None
    fig_fi: Optional[Dict[str, Any]] = None
    fig_clusters: Optional[Dict[str, Any]] = None

    data: Optional[List[Dict[str, Any]]] = None


# ==================== Схемы для legacy-эндпоинтов (main.py) ====================


class PredictRequest(BaseModel):
    """Синхронное предсказание для одного студента"""

    data: Dict[str, Any]  # признаки студента


class PredictResponse(BaseModel):
    """Ответ предсказания"""

    prediction: int
    probability: float
    model_name: Optional[str] = None
    features_used: Optional[List[str]] = None


class TaskStatus(BaseModel):
    """Статус фоновой задачи (RQ)"""

    task_id: str
    status: str  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    stage: Optional[str] = None


class TrainResponse(BaseModel):
    """Ответ при запуске обучения"""

    task_id: str
    status: str = "started"
