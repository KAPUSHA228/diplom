"""
Фоновые задачи Ray для обучения моделей и SHAP-объяснений.
"""

import os
import sys
import ray
import pandas as pd
from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_core.features import add_composite_features, get_base_features, preprocess_data_for_smote
from ml_core.models import ModelTrainer
from ml_core.evaluation import generate_shap_explanations
from ml_core.error_handler import logger

# Глобальный флаг инициализации
_ray_initialized = False


def ensure_ray():
    """Гарантирует инициализацию Ray"""

    global _ray_initialized
    if not _ray_initialized and not ray.is_initialized():

        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            try:
                ray.init(address=ray_address, ignore_reinit_error=True)
                print(f"✅ Ray подключился к внешнему кластеру ({ray_address})")
            except Exception as e:
                print(f"❌ Ошибка подключения к Ray: {e}")
                raise
        else:
            # Локальный запуск: создаём новый кластер
            try:
                ray.init(ignore_reinit_error=True, num_cpus=4)
                print("✅ Ray запущен локально")
            except Exception as e:
                print(f"❌ Ошибка запуска локального Ray: {e}")
                raise
        _ray_initialized = True
    return ray.is_initialized()


def serialize_figure(fig):
    """Конвертирует Plotly фигуру в JSON"""
    if fig is None:
        return None
    try:
        return fig.to_plotly_json()
    except Exception as e:
        logger.error(f"Failed to serialize figure: {e}")
        return None


def init_ray():
    """Инициализация Ray (вызывать при старте сервера)"""
    ensure_ray()
    print("✅ Ray инициализирован")


@ray.remote
class ProgressActor:
    def __init__(self):
        self.stage = "Начало"
        self.progress = 0

    def update(self, stage, progress):
        self.stage = stage
        self.progress = progress

    def get(self):
        return {"stage": self.stage, "progress": self.progress}


@ray.remote
def train_model_task(df_data: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """Обучение модели через Ray"""
    ensure_ray()

    try:
        df = pd.DataFrame(df_data)

        # Обработка
        df = add_composite_features(df)
        feature_cols = get_base_features(df)
        target_col = params.get("target_col", None)
        print(f"🔍 Целевая переменная: {target_col}")
        print(f"🔍 Признаки: {feature_cols}")
        print(f"🔍 Целевая переменная в признаках? {target_col in feature_cols}")
        if not target_col:
            raise ValueError("target_col must be specified. Please select a target column in the UI.")
        # Если risk_flag нет, ищем другую бинарную цель
        if target_col not in df.columns:
            possible = [c for c in df.select_dtypes(include="number").columns if c not in feature_cols]
            target_col = possible[0] if possible else feature_cols[0]
        if target_col in feature_cols:
            print(f"⚠️ Удаляем целевую переменную '{target_col}' из признаков")
            feature_cols.remove(target_col)

        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
        y = df[target_col]

        # Авто-бинаризация
        if y.dtype in ["float64", "int64"] and y.nunique() > 2:
            median_val = y.median()
            y = (y > median_val).astype(int)

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # SMOTE
        if y.nunique() > 1 and y.min() == 0 and y.max() == 1:
            try:
                X_train, y_train = preprocess_data_for_smote(X_train, y_train)
            except Exception as e:
                logger.info(f"SMOTE preprocessing failed: {e}.  Continuing without SMOTE.")

        # Обучение
        trainer = ModelTrainer()
        model, model_name, metrics = trainer.train_best_model(X_train, y_train, X_test, y_test)

        # Сохранение
        model_path, meta = trainer.save_model(model, model_name, metrics, feature_cols)

        return {
            "status": "success",
            "model_id": model_name,
            "model_path": model_path,
            "metrics": metrics.get("test", {}),
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"status": "error", "error": str(e)}


@ray.remote
def correlation_task(data: list, target_col: str = None):
    """Асинхронная корреляция для больших данных"""
    import pandas as pd

    df = pd.DataFrame(data)

    # Берём только числовые колонки
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col and target_col in numeric_df.columns:
        # Вычисляем корреляции только с целевой колонкой (быстрее)
        correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
        return {
            "correlations": correlations.to_dict(),
            "target_col": target_col,
            "n_rows": len(df),
            "n_columns": len(numeric_df.columns),
        }
    else:
        # Полная матрица
        corr_matrix = numeric_df.corr()
        return {"correlation_matrix": corr_matrix.to_dict(), "n_rows": len(df), "n_columns": len(numeric_df.columns)}


@ray.remote
def shap_task(model_id: str, df_data: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    """SHAP объяснения через Ray"""
    ensure_ray()

    try:
        df = pd.DataFrame(df_data)
        trainer = ModelTrainer()

        # Загрузка модели
        model = trainer.load_model(model_name=model_id)
        if model is None:
            model, model_name, meta = trainer.get_best_model()
            if model is None:
                return {"status": "error", "error": f"Model '{model_id}' not found"}
            logger.info(f"Using latest model: {model_name}")

        feature_cols = [c for c in df.columns if c not in ["risk_flag", "user_id", "user"]]
        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

        explanations = generate_shap_explanations(model, X, feature_cols, threshold=threshold)

        return {
            "status": "success",
            "explanations": explanations,
            "n_students": len(explanations),
        }

    except Exception as e:
        logger.error(f"SHAP generation failed: {e}")
        return {"status": "error", "error": str(e)}


@ray.remote
def full_analysis_task(
    data: List[Dict[str, Any]], params: Dict[str, Any], progress_actor: ray.actor.ActorHandle = None
) -> Dict[str, Any]:
    """Полный анализ через Ray"""
    ensure_ray()

    def report_progress(stage, progress):
        if progress_actor:
            progress_actor.update.remote(stage, progress)
        print(f"📊 {stage}: {progress}%")

    try:
        # ПРОВЕРКА 1: Вызываем report_progress ДО анализа
        report_progress("НАЧАЛО АНАЛИЗА (тест)", 1)

        df = pd.DataFrame(data)

        # ПРОВЕРКА 2: Ещё один вызов
        report_progress("ЗАГРУЗКА ДАННЫХ (тест)", 5)
        target_col = params.get("target_col", "risk_flag")

        # ✅ ПРОВЕРКА: есть ли хоть один признак (кроме целевой переменной)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_target_cols = [c for c in numeric_cols if c != target_col]

        if not non_target_cols:
            error_msg = (
                f"❌ Анализ невозможен: в данных нет признаков для анализа.\n"
                f"Была выбрана целевая переменная '{target_col}', "
                "но после её исключения не осталось числовых колонок.\n"
                f"Доступные колонки: {list(df.columns)}\n"
                "Добавьте в файл хотя бы один числовой признак."
            )
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        from ml_core.analyzer import ResearchAnalyzer

        analyzer = ResearchAnalyzer()

        # ПРОВЕРКА 3: Перед вызовом основного метода
        report_progress("ЗАПУСК АНАЛИЗАТОРА (тест)", 10)
        result = analyzer.run_full_analysis(
            df=df,
            target_col=params.get("target_col", "risk_flag"),
            n_clusters=params.get("n_clusters", 3),
            risk_threshold=params.get("risk_threshold", 0.5),
            corr_threshold=params.get("corr_threshold", 0.3),
            is_synthetic=params.get("is_synthetic", False),
            use_smote=params.get("use_smote", True),
            use_lr=params.get("use_lr", True),
            use_rf=params.get("use_rf", True),
            use_xgb=params.get("use_xgb", True),
            optimization_metric=params.get("optimization_metric", None),
            n_features_to_select=params.get("n_features_to_select", 7),
            shap_top_n=params.get("shap_top_n", 5),
            use_hp_tuning=params.get("use_hp_tuning", False),
            n_iter_tuning=params.get("n_iter_tuning", 20),
            progress_callback=report_progress,
        )
        opt_metric = params.get("optimization_metric")

        if opt_metric is None or opt_metric == "default":
            display_metric = "F1 (по умолчанию)"
        elif opt_metric == "f1":
            display_metric = "F1-score"
        elif opt_metric == "roc_auc":
            display_metric = "ROC-AUC"
        elif opt_metric == "precision":
            display_metric = "Precision"
        elif opt_metric == "recall":
            display_metric = "Recall"
        else:
            display_metric = opt_metric
        # Сериализуем результат (очищаем от numpy типов)
        from shared.utils import safe_json_serializable

        result_dict = {
            "status": result.status,
            "message": result.message,
            "metrics": result.metrics,
            "test_metrics": result.test_metrics,
            "cv_results": result.cv_results,
            "selected_features": result.selected_features,
            "cluster_profiles": result.cluster_profiles,
            "explanations": result.explanations,
            "predictions": result.predictions,
            "data_with_clusters": result.data_with_clusters,
            "target_col": result.target_col,
            "config": {
                "model_name": result.model_name,
                "target_col": result.target_col,
                "n_samples": len(df),
                "n_features": len(result.selected_features),
                "n_clusters": params.get("n_clusters"),
                "use_smote": params.get("use_smote"),
                "corr_threshold": params.get("corr_threshold"),
                "optimization_metric": display_metric,
                "risk_threshold": params.get("risk_threshold"),
                "shap_top_n": params.get("shap_top_n"),
                "use_lr": params.get("use_lr"),
                "use_rf": params.get("use_rf"),
                "use_xgb": params.get("use_xgb"),
                "use_hp_tuning": params.get("use_hp_tuning"),
                "n_iter_tuning": params.get("n_iter_tuning"),
            },
            # Сериализуем графики
            "fig_cm": serialize_figure(result.fig_cm),
            "fig_roc": serialize_figure(result.fig_roc),
            "fig_fi": serialize_figure(result.fig_fi),
            "fig_clusters": serialize_figure(result.fig_clusters),
            "fig_corr": serialize_figure(result.fig_corr),
        }
        report_progress("АНАЛИЗ ЗАВЕРШЁН (тест)", 100)
        print("🔵🔵🔵 ОТЛАДКА ГРАФИКОВ 🔵🔵🔵")
        print(f"fig_cm type: {type(result.fig_cm)}")
        print(f"fig_cm is None: {result.fig_cm is None}")

        if result.fig_cm is not None:
            try:
                test_json = result.fig_cm.to_plotly_json()
                print(f"fig_cm JSON keys: {test_json.keys() if test_json else 'None'}")
                print(f"fig_cm has data: {len(test_json.get('data', [])) if test_json else 0}")
            except Exception as e:
                print(f"fig_cm serialization error: {e}")
        print("🔵 НАЧАЛО СЕРИАЛИЗАЦИИ РЕЗУЛЬТАТА")
        import time

        start_serialize = time.time()

        serialized_result = safe_json_serializable(result_dict)
        print(f"🔵 serialized_result keys: {serialized_result.keys()}")
        print(f"🔵 serialized_result has fig_cm: {'fig_cm' in serialized_result}")
        print(f"🔵 serialized_result fig_cm: {serialized_result.get('fig_cm')}")
        print(f"🔵 СЕРИАЛИЗАЦИЯ ЗАВЕРШЕНА за {time.time() - start_serialize:.2f} сек")
        print(f"🔵 РАЗМЕР РЕЗУЛЬТАТА: {len(str(serialized_result)) / 1024 / 1024:.2f} MB")
        return {
            "status": "success",
            **serialized_result,
            "target_col": params.get("target_col", "risk_flag"),
        }

    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        if progress_actor:
            progress_actor.update.remote(f"Ошибка: {str(e)}", 0)
        return {"status": "error", "error": str(e)}
