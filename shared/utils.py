"""
Общие утилиты для всех сервисов.
"""

import numpy as np
from typing import Any


def scrub(obj: Any) -> Any:
    """Рекурсивная очистка от nan/inf/numpy для JSON."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return scrub(obj.tolist())
    if isinstance(obj, (np.floating, np.float64)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if hasattr(obj, "to_dict"):
        return scrub(obj.to_dict())
    return obj


def safe_json_serializable(obj: Any) -> Any:
    """Рекурсивная очистка данных от nan/inf/numpy-типов для JSON."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return safe_json_serializable(obj.tolist())
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: safe_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json_serializable(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if hasattr(obj, "to_dict"):
        return safe_json_serializable(obj.to_dict())
    return obj
