# ml_core/error_handler.py
import logging
import json
import sys
from pathlib import Path
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """JSON-форматтер для микросервисных логов."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "ml_core",
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


# Настройка глобального логгера
logger = logging.getLogger("ml_core")
logger.setLevel(logging.INFO)

# Создаём обработчик для файла
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

file_handler = logging.FileHandler(log_dir / "ml_core.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(JSONFormatter())

# Добавляем обработчик (если ещё не добавлен)
if not logger.handlers:
    logger.addHandler(file_handler)

# Также выводим в консоль (JSON)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(JSONFormatter())
console_handler.setLevel(logging.WARNING)  # только предупреждения и ошибки в консоль
logger.addHandler(console_handler)


def safe_execute(func, *args, error_msg="Ошибка при выполнении", **kwargs):
    """
    Безопасное выполнение функции с логированием ошибок.

    Args:
        func: вызываемая функция
        *args: позиционные аргументы для func
        error_msg: префикс сообщения об ошибке
        **kwargs: именованные аргументы для func

    Returns:
        результат func или переисключение ошибки с логом
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}", exc_info=True)
        raise
