"""
Модуль кеширования результатов ML-операций через Redis.
Если Redis недоступен, декоратор работает как обычный вызов функции (без кеширования).
"""

import hashlib
import pickle

import pandas as pd
from ml_core.error_handler import logger

# Redis — опциональная зависимость
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# REDIS_URL берём из окружения напрямую (без импорта config.settings, чтобы избежать циклического импорта)
import os

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Подключение к Redis (если доступен)
redis_client = redis.from_url(_REDIS_URL, decode_responses=False) if REDIS_AVAILABLE else None

CACHE_TTL = 3600  # 1 час


def _make_key(func_name: str, *args, **kwargs) -> str:
    """Генерирует уникальный ключ для кеширования."""
    # Хешируем аргументы, чтобы создать уникальный ключ
    # Для pandas DataFrame используем хеш содержимого
    hash_obj = hashlib.md5()

    # Простая сериализация аргументов для ключа
    key_data = f"{func_name}|"
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            key_data += str(hash(pd.util.hash_pandas_object(arg).sum()))
        else:
            key_data += str(arg)
    key_data += str(kwargs)

    hash_obj.update(key_data.encode("utf-8"))
    return f"ml_cache:{func_name}:{hash_obj.hexdigest()}"


def cache_result(func):
    """Декоратор для кеширования результатов функции в Redis.

    Если Redis недоступен, просто вызывает функцию без кеширования.
    """

    def wrapper(*args, **kwargs):
        # Если Redis недоступен, просто вызываем функцию
        if redis_client is None:
            return func(*args, **kwargs)

        try:
            redis_client.ping()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logger.debug(f"Redis unavailable: {e}. Skipping cache.")
            return func(*args, **kwargs)

        key = _make_key(func.__name__, *args, **kwargs)

        # Попытка получить из кеша
        try:
            cached = redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        except (ValueError, ImportError, AttributeError) as e:
            logger.warning(f"SMOTE failed: {e}. Continuing without SMOTE.")

        # Вызов функции
        result = func(*args, **kwargs)

        # Сохранение в кеш
        try:
            redis_client.setex(key, CACHE_TTL, pickle.dumps(result))
        except (redis.exceptions.RedisError, pickle.PickleError, TypeError) as e:
            logger.warning(f"Failed to cache result for {func.__name__}: {e}")

        return result

    return wrapper
