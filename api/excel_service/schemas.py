from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class ColumnMapping(BaseModel):
    """Описание обработки одной колонки"""

    type: str = Field(..., description="numeric | categorical | multiple_choice | ordinal | text")
    scale: Optional[str] = Field(None, description="likert | binary | custom")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    separator: Optional[str] = Field(None, description="Для multiple_choice, например ';' или ','")
    encoding: Optional[str] = Field("auto", description="onehot | label | ordinal | none")
    imputation: str = Field("auto", description="auto | median | mean | mode | drop")
    drop: bool = Field(False, description="Полностью исключить колонку")


class MappingConfig(BaseModel):
    """Полная конфигурация маппинга для листа/датасета"""

    sheet_name: Optional[str] = None
    sheet_type: str = Field("numeric", description="numeric | single_choice | multiple_choice | mixed | skip")

    columns: Dict[str, ColumnMapping] = Field(
        default_factory=dict, description="Ключ — имя колонки, значение — правила обработки"
    )

    global_settings: Dict[str, Any] = Field(default_factory=dict, description="Глобальные настройки обработки")

    detected_group: Optional[str] = None


class ExcelPreviewRequest(BaseModel):
    sheet_name: str


class ExcelProcessRequest(BaseModel):
    sheet_name: str
    mapping_config: Optional[Dict[str, Any]] = None


class ImputationRequest(BaseModel):
    df: List[Dict[str, Any]]
    strategy: str = "auto"
    threshold: float = 30.0
