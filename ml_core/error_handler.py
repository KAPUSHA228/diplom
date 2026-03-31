# ml_core/error_handler.py
import logging
import sys
from pathlib import Path

# Настройка глобального логгера
logger = logging.getLogger("ml_core")
logger.setLevel(logging.INFO)

# Создаём обработчик для файла
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

file_handler = logging.FileHandler(log_dir / "ml_core.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Формат логов
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

# Добавляем обработчик (если ещё не добавлен)
if not logger.handlers:
    logger.addHandler(file_handler)

# Также выводим в консоль (опционально)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)   # только предупреждения и ошибки в консоль
logger.addHandler(console_handler)


def safe_execute(func, *args, error_msg="Ошибка при выполнении", **kwargs):
    """Безопасное выполнение функции с логированием"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}", exc_info=True)
        raise
