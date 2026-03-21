# config/settings.py
"""
Настройки проекта (аналог settings.py в Django, но проще)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env файл если есть
load_dotenv()

# Базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Режим отладки
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Настройки данных
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Создаем директории если их нет
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Настройки базы данных (для бэкендера)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///db.sqlite3')

# Настройки Redis (для кэширования)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Настройки моделей ML
RANDOM_SEED = 42
DEFAULT_N_CLUSTERS = 3
RISK_THRESHOLD = 0.5

# Пути к синтетическим данным
SYNTHETIC_DATA_PATH = DATA_DIR / 'synthetic_students.csv'