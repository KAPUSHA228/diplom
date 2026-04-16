# config/settings.py
"""
Настройки проекта (аналог settings.py в Django, но проще)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Импортируем оттуда для обратной совместимости
from ml_core.config import config as _ml_config

# Загружаем .env файл если есть
load_dotenv()

# Базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Режим отладки
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Настройки данных
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Создаем директории если их нет
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Настройки базы данных (для бэкендера)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3")

# Настройки Redis (для кэширования)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Пути к синтетическим данным
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_students.csv"

# ML-константы — единый источник в ml_core/config.py


RANDOM_SEED = _ml_config.RANDOM_SEED
DEFAULT_N_CLUSTERS = _ml_config.DEFAULT_N_CLUSTERS
RISK_THRESHOLD = _ml_config.RISK_THRESHOLD
SHAP_TOP_N = _ml_config.SHAP_TOP_N
CV_FOLDS = _ml_config.CV_FOLDS
TEST_SIZE = _ml_config.TEST_SIZE
