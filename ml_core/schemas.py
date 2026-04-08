from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

# class AnalysisRequest(BaseModel):
#     df: Any  # будем передавать как dict или использовать custom validator
#     target_col: str
#     n_clusters: int = 3
#     risk_threshold: float = 0.5
#     use_hp_tuning: bool = False


class AnalysisRequest(BaseModel):
    """Запрос на выполнение полного анализа от бэкенда или UI"""

    # Основные данные
    df: Any = Field(..., description="Данные в виде DataFrame, dict или list[dict]")

    # Обязательные параметры анализа
    target_col: str = Field(..., description="Название целевой переменной")

    # Параметры кластеризации и анализа
    n_clusters: int = Field(3, ge=2, le=8)
    risk_threshold: float = Field(0.5, ge=0.0, le=1.0)
    corr_threshold: float = Field(0.3, ge=0.0, le=1.0)

    # Флаги поведения
    is_synthetic: bool = Field(False, description="Являются ли данные синтетическими")
    use_smote: bool = Field(True, description="Использовать SMOTE для балансировки")
    use_hp_tuning: bool = Field(False, description="Использовать оптимизацию гиперпараметров")
    use_composite_features: bool = Field(True, description="Применять автоматические композитные признаки")

    # Дополнительные метаданные (для логирования и трекинга)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    data_source_id: Optional[str] = None  # ID записи в БД/системе хранения
    data_source_type: Optional[str] = Field(None, description="survey | lab | research | synthetic")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompositeScoreRequest(BaseModel):
    """Запрос на создание композитной оценки — взвешенная сумма признаков."""

    df: Any
    feature_weights: Dict[str, float]
    score_name: str = "custom_score"
    normalize: bool = True


class AnalysisResult(BaseModel):
    """Полный результат анализа АРМ исследователя"""

    status: str = "success"
    message: Optional[str] = None

    # Основные результаты анализа
    metrics: Dict[str, Any] = Field(default_factory=dict)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    cv_results: Dict[str, Any] = Field(default_factory=dict)
    selected_features: List[str] = Field(default_factory=list)

    # Кластеризация
    cluster_profiles: Dict[str, Any] = Field(default_factory=dict)
    cluster_labels: Optional[List[int]] = None

    # Объяснимость
    explanations: List[Dict[str, Any]] = Field(default_factory=list)

    # Графики (Plotly фигуры)
    fig_cm: Optional[Any] = None  # Confusion Matrix
    fig_roc: Optional[Any] = None  # ROC-кривые
    fig_fi: Optional[Any] = None  # Feature Importance
    fig_clusters: Optional[Any] = None  # PCA кластеры
    fig_corr: Optional[Any] = None  # Correlation heatmap

    # Дополнительные данные для UI
    last_y_test: Optional[List] = None
    last_y_pred: Optional[List] = None
    last_X_test: Optional[Any] = None  # может быть DataFrame или ndarray

    # Дрейф
    drift_report: Optional[Dict[str, Any]] = None

    # Композитные оценки
    composite_scores_created: Optional[List[str]] = None

    # Метаданные
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    model_name: Optional[str] = None
    timestamp: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# class AnalysisResult(BaseModel):
#     metrics: Dict[str, Any]
#     test_metrics: Dict[str, float]
#     selected_features: List[str]
#     cluster_profiles: Dict[str, Any]
#     explanations: List[Dict[str, Any]]
#     drift_report: Optional[Dict[str, Any]] = None
#     status: str = "success"
#     message: Optional[str] = None
#     composite_scores_created: Optional[List[str]] = None
#     cv_results: dict = Field(default_factory=dict)


class TrajectoryRequest(BaseModel):
    """Запрос на анализ траектории студента"""

    df: Any
    student_id: Any
    value_col: str = "avg_grade"
    time_col: str = "semester"
