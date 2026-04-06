"""
FastAPI зависимости (Dependency Injection).
Используются в эндпоинтах через Depends().
"""

from functools import lru_cache

from ml_core.analyzer import ResearchAnalyzer
from ml_core.models import ModelTrainer


@lru_cache
def get_analyzer() -> ResearchAnalyzer:
    """Один экземпляр анализатора на всё приложение."""
    return ResearchAnalyzer()


@lru_cache
def get_trainer() -> ModelTrainer:
    """Один экземпляр трейнера моделей."""
    return ModelTrainer(models_dir="models")
