from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class ExperimentSaveRequest(BaseModel):
    name: str
    metrics: Dict[str, Any] = {}
    features: List[str] = []
    description: str = ""
    config: Dict[str, Any] = {}
    target_col: Optional[str] = None
    model_name: Optional[str] = None
