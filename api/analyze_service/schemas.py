from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class FeatureCombinationRequest(BaseModel):
    """Запрос на создание комбинированных признаков"""

    df: List[Dict[str, Any]]
    numerical_cols: Optional[List[str]] = None
    text_cols: Optional[List[str]] = None
    max_pairs: int = 15
    target_col: Optional[str] = None


class CrosstabRequest(BaseModel):
    df: List[Dict[str, Any]]
    row_var: str
    col_var: str
    values: Optional[str] = None
    aggfunc: str = "count"
    normalize: bool = False
    n_bins: int = Field(4, ge=2, le=10)
    bin_method: str = Field("cut", pattern="^(cut|qcut)$")


class TrajectoryRequest(BaseModel):
    df: List[Dict[str, Any]]
    student_id: Optional[Any] = None
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
