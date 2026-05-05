from pydantic import BaseModel
from typing import Dict, Any, List


class ExperimentSaveRequest(BaseModel):
    name: str
    metrics: Dict[str, Any] = {}
    features: List[str] = []
    description: str = ""
    config: Dict[str, Any] = {}
