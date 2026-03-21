# from pydantic import BaseModel, Field
# from typing import Optional, List, Dict, Any
# from datetime import datetime
#
# class PredictRequest(BaseModel):
#     """Запрос на предсказание для одного студента"""
#     data: Dict[str, Any] = Field(..., description="Признаки студента")
#
# class PredictResponse(BaseModel):
#     """Ответ на предсказание"""
#     prediction: int = Field(..., description="0 - нет риска, 1 - риск")
#     probability: float = Field(..., description="Вероятность риска (0-1)")
#     student_id: Optional[int] = None
#
# class TrainRequest(BaseModel):
#     """Запрос на обучение модели"""
#     dataset_name: str = Field(..., description="Название датасета")
#     params: Optional[Dict[str, Any]] = Field(default_factory=dict)
#
# class TrainResponse(BaseModel):
#     """Ответ на запуск обучения"""
#     task_id: str = Field(..., description="ID задачи для отслеживания")
#     status: str = Field(..., description="started")
#
# class TaskStatus(BaseModel):
#     """Статус задачи"""
#     task_id: str
#     status: str  # pending, in_progress, completed, failed
#     progress: Optional[int] = None
#     stage: Optional[str] = None
#     result: Optional[Dict[str, Any]] = None
#     error: Optional[str] = None