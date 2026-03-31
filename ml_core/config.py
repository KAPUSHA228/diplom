from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    EXPERIMENTS_DIR = BASE_DIR / "experiments"
    ANALYSIS_DATA_DIR = BASE_DIR / "analysis_data"

    RANDOM_SEED = 42
    DEFAULT_N_CLUSTERS = 3
    RISK_THRESHOLD = 0.5
    SHAP_TOP_N = 5
    CV_FOLDS = 5
    TEST_SIZE = 0.2

    # Пути создаются автоматически при импорте
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    ANALYSIS_DATA_DIR.mkdir(exist_ok=True)

    def __init__(self):
        # Автоматическое создание всех необходимых папок
        for directory in [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.EXPERIMENTS_DIR,
            self.ANALYSIS_DATA_DIR,
            self.ANALYSIS_DATA_DIR / "synthetic",
            self.ANALYSIS_DATA_DIR / "monitoring",
            self.ANALYSIS_DATA_DIR / "processed"
        ]:
            directory.mkdir(parents=True, exist_ok=True)


config = Config()