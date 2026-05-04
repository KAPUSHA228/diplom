from typing import Optional
import pandas as pd
from .config import config
from .features import add_composite_features, build_composite_score, get_base_features, preprocess_data_for_smote
from .analysis import correlation_analysis, cluster_students, analyze_cluster_profiles, plot_corr_heatmap
from .models import ModelTrainer
from .evaluation import generate_shap_explanations, plot_confusion_matrix, plot_roc_curves, plot_feature_importance
from .error_handler import safe_execute, logger
from .schemas import AnalysisRequest, AnalysisResult
from .timeseries import detect_negative_dynamics, analyze_student_trajectory, forecast_grades
from .text_processor import extract_text_features
from sklearn.model_selection import train_test_split
import numpy as np


class ResearchAnalyzer:
    """Главный класс АРМ исследователя — единая точка входа.
    Предоставляет чистый и единый интерфейс ко всем возможностям ml_core.
    """

    def __init__(self):
        self.trainer = ModelTrainer(models_dir=str(config.MODELS_DIR))

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        target_col: str = "risk_flag",
        n_clusters: int = 3,
        risk_threshold: float = 0.5,
        corr_threshold: float = 0.3,
        is_synthetic: bool = False,
        use_smote: bool = True,
        # Параметры выбора моделей (из сайдбара)
        use_lr: bool = True,
        use_rf: bool = True,
        use_xgb: bool = True,
        optimization_metric: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Полный пайплайн АРМ исследователя: текст → композиты → корреляция →
        кластеризация → SMOTE → отбор признаков → CV → обучение → SHAP → графики.

        Args:
            df: исходный DataFrame
            target_col: имя целевой переменной
            n_clusters: число кластеров K-means
            risk_threshold: порог классификации риска
            corr_threshold: порог для корреляционного анализа
            is_synthetic: являются ли данные синтетическими
            use_smote: применять ли SMOTE

        Returns:
            AnalysisResult: полный результат анализа
        """
        try:
            df = df.copy()

            # Проверяем наличие целевой колонки
            if target_col not in df.columns:
                # Ищем похожую (risk, target, flag, class)
                possible = [c for c in df.columns if any(k in c.lower() for k in ["risk", "target", "flag", "class"])]
                if possible:
                    target_col = possible[0]
                    logger.info(f"Колонка '{target_col}' не найдена, используем '{possible[0]}'")
                else:
                    raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

            if df[target_col].nunique() > 2:
                median_val = df[target_col].median()
                df[target_col] = (df[target_col] > median_val).astype(int)
                logger.info(f"Target '{target_col}' бинаризирован по медиане {median_val:.2f}")

            # TODO рудимиент
            if "essay_text" in df.columns:
                df = extract_text_features(df, "essay_text")

            df = safe_execute(add_composite_features, df)
            all_features = get_base_features(df, is_synthetic=is_synthetic)

            corr_result = safe_execute(
                correlation_analysis, df, all_features, target_col, corr_threshold=corr_threshold
            )

            # Генерируем heatmap корреляций
            fig_corr = None
            print(f"DEBUG corr: corr_result type={type(corr_result)}")
            if corr_result:
                print(f"DEBUG corr: corr_result keys={corr_result.keys()}")
                fm = corr_result.get("full_matrix")
                print(
                    f"DEBUG corr: full_matrix type={type(fm)}, columns={list(fm.columns) if fm is not None else None}"
                )
                if fm is not None:
                    fig_corr = safe_execute(plot_corr_heatmap, fm)
                    print(f"DEBUG corr: fig_corr type={type(fig_corr)}, success={fig_corr is not None}")
            else:
                print("DEBUG corr: corr_result is None")

            cluster_labels, _, _ = safe_execute(cluster_students, df, n_clusters=n_clusters, feature_cols=all_features)
            df = df.copy()
            # Перед вызовом кластеризации
            if not all_features:
                logger.warning("all_features пустой. Используем все числовые колонки кроме student_id")
                all_features = [col for col in df.select_dtypes(include=[np.number]).columns if col != "student_id"]
            df["cluster"] = cluster_labels
            cluster_profiles = analyze_cluster_profiles(df, all_features)

            # === ГЕНЕРАЦИЯ ГРАФИКА КЛАСТЕРОВ ===
            fig_clusters = None
            try:
                from ml_core.analysis import plot_clusters_pca

                fig_clusters = plot_clusters_pca(df, cluster_labels, all_features)
            except Exception as e:
                logger.error(f"Ошибка при создании графика кластеров: {e}")
            # ==================================

            # === Train / Test сплит ===
            # X = df[all_features].fillna(df[all_features].median(numeric_only=True))
            X = df[all_features]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            if use_smote:
                X_train, y_train = preprocess_data_for_smote(X_train, y_train)  # новая функция

            # === УЧИТЫВАЕМ НАСТРОЙКИ САЙДБАРА (какие модели включены) ===
            active_models = {}
            if use_lr:
                active_models["LR"] = self.trainer.models.get("LR")
            if use_rf:
                active_models["RF"] = self.trainer.models.get("RF")
            if use_xgb:
                active_models["XGB"] = self.trainer.models.get("XGB")

            # Если ничего не выбрано — берем LR для безопасности
            if not active_models:
                active_models = {"LR": self.trainer.models.get("LR")}

            # Временно подменяем модели в тренере
            original_models_backup = self.trainer.models
            self.trainer.models = active_models
            # ==========================================================

            model, model_name, metrics = self.trainer.train_best_model(
                X_train, y_train, X_test, y_test, scoring=optimization_metric
            )

            # Возвращаем оригинальный набор моделей обратно
            self.trainer.models = original_models_backup
            # ==========================================================

            # SHAP
            explanations = safe_execute(
                generate_shap_explanations,
                model,
                pd.DataFrame(X_test, columns=all_features),
                all_features,
                threshold=risk_threshold,
                target_name=target_col,
            )

            self.last_df = df
            self.last_metrics = metrics
            self.last_features = all_features
            self.last_model_name = model_name
            self.last_selected_cols = all_features
            self.last_X_test = X_test
            self.last_y_test = y_test
            self.last_y_pred = model.predict(X_test)

            # === ГЕНЕРАЦИЯ ГРАФИКОВ (с отладкой) ===
            try:
                print("DEBUG: Начинаю создание графиков...")

                # 1. Confusion Matrix
                fig_cm = plot_confusion_matrix(y_test, self.last_y_pred, model_name)
                print(f"DEBUG: fig_cm создан: {type(fig_cm)}")

                # 2. ROC-кривые (создаем СВЕЖИЕ модели согласно настройкам сайдбара)
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from xgboost import XGBClassifier
                from sklearn.base import clone

                all_models_for_roc = {}

                # Словарь с чистыми шаблонами
                clean_templates = {
                    "LR": LogisticRegression(max_iter=1000),
                    "RF": RandomForestClassifier(random_state=42),
                    "XGB": XGBClassifier(eval_metric="logloss", random_state=42),
                }

                # Проходим по шаблонам и смотрим, включена ли модель
                flags = {"LR": use_lr, "RF": use_rf, "XGB": use_xgb}

                for name, tpl in clean_templates.items():
                    if not flags.get(name, True):
                        continue  # Если модель выключена в сайдбаре — пропускаем

                    if name == model_name:
                        all_models_for_roc[name] = model  # Лучшая модель уже обучена
                    else:
                        try:
                            m = clone(tpl)  # Берем свежую модель
                            m.fit(X_train, y_train)
                            all_models_for_roc[name] = m
                            print(f"DEBUG: Модель {name} обучена для ROC с нуля")
                        except Exception as e:
                            print(f"DEBUG: Ошибка обучения {name}: {e}")

                print(f"DEBUG: Передаю в plot_roc_curves моделей: {list(all_models_for_roc.keys())}")
                fig_roc = plot_roc_curves(all_models_for_roc, X_test, y_test)
                print(f"DEBUG: fig_roc создан: {type(fig_roc)}")

                # 3. Feature Importance
                fig_fi = plot_feature_importance(model, all_features)
                print(f"DEBUG: fig_fi создан: {type(fig_fi)}")

            except Exception as e:
                import traceback

                print(f"!!! ОШИБКА ПРИ СОЗДАНИИ ГРАФИКОВ: {e}")
                traceback.print_exc()
                # Если что-то упало, ставим None, чтобы не ломать весь ответ
                fig_cm = None
                fig_roc = None
                fig_fi = None

            df_with_clusters = df.copy()
            # Убеждаемся, что колонка существует
            if "cluster" not in df_with_clusters.columns:
                df_with_clusters["cluster"] = cluster_labels

            self.last_df = df_with_clusters

            # Добавляем в результат
            return AnalysisResult(
                metrics=metrics,
                test_metrics=metrics.get("test", {}),
                selected_features=all_features,
                cluster_profiles=(
                    cluster_profiles.to_dict() if hasattr(cluster_profiles, "to_dict") else cluster_profiles
                ),
                explanations=explanations or [],
                cv_results=metrics.get("cv_results", {}),
                status="success",
                model_name=model_name,
                fig_cm=fig_cm,
                fig_roc=fig_roc,
                fig_fi=fig_fi,
                fig_clusters=fig_clusters,
                fig_corr=fig_corr,
                last_y_test=y_test.tolist() if hasattr(y_test, "tolist") else y_test,
                last_y_pred=self.last_y_pred.tolist() if hasattr(self.last_y_pred, "tolist") else self.last_y_pred,
                data_with_clusters=df_with_clusters.to_dict("records"),
            )

        except Exception as e:
            logger.error(f"Ошибка в run_full_analysis: {str(e)}", exc_info=True)
            return AnalysisResult(
                metrics={},
                test_metrics={},
                selected_features=[],
                cluster_profiles={},
                explanations=[],
                status="error",
                message=str(e),
            )

    # Можно добавить отдельные методы:
    def create_composite_score(self, df: pd.DataFrame, feature_weights: dict, score_name: str = "custom_score"):
        """
        Создаёт композитную оценку с заданными весами.

        Args:
            df: DataFrame с данными
            feature_weights: dict {признак: вес}
            score_name: имя новой колонки

        Returns:
            (df, score_name): DataFrame с новой колонкой и имя
        """
        return build_composite_score(df, feature_weights, score_name)

    def analyze_student_trajectory(self, df: pd.DataFrame, student_id, value_col: str = "avg_grade"):
        """
        Анализирует траекторию студента: тренд, статус, график.

        Args:
            df: DataFrame с данными по семестрам
            student_id: идентификатор студента
            value_col: колонка показателя

        Returns:
            dict: {'trend', 'status', 'figure', ...}
        """
        return analyze_student_trajectory(df, student_id, value_col=value_col)

    def detect_negative_dynamics(self, df: pd.DataFrame, value_col: str = "avg_grade"):
        """
        Находит студентов с ухудшающейся динамикой показателя.

        Args:
            df: DataFrame с данными по семестрам
            value_col: колонка показателя

        Returns:
            dict: {'n_students_analyzed', 'at_risk_students', 'risk_percentage'}
        """
        return detect_negative_dynamics(df, value_col=value_col)

    def forecast_for_student(self, df: pd.DataFrame, student_id, future_semesters: int = 2):
        """
        Прогнозирует показатель студента на будущие семестры.

        Args:
            df: DataFrame с данными по семестрам
            student_id: идентификатор студента
            future_semesters: число семестров для прогноза

        Returns:
            dict: {'future_semesters', 'predictions'}
        """
        return forecast_grades(df, student_id, future_semesters=future_semesters)

    def select_subset(
        self,
        df: pd.DataFrame,
        condition: str = None,
        n_samples: int = None,
        random_seed: int = 42,
        by_cluster: int = None,
    ) -> pd.DataFrame:
        """
        Выделяет подмножество респондентов по условию, случайной выборке или кластеру.

        Args:
            df: исходный DataFrame
            condition: pandas query строка (например "avg_grade > 4.0")
            n_samples: размер случайной выборки
            random_state: seed для семплирования
            by_cluster: номер кластера для фильтрации

        Returns:
            pd.DataFrame: подмножество данных
        """
        df = df.copy()

        if condition:
            try:
                subset = df.query(condition)
            except Exception as e:
                raise ValueError(f"Ошибка в pandas query: {e}")

        elif by_cluster is not None:
            # Ищем колонку с кластерами (самое частое — cluster_label или cluster)
            cluster_col = None
            for possible in ["cluster_label", "cluster", "Cluster", "cluster_id"]:
                if possible in df.columns:
                    cluster_col = possible
                    break

            if cluster_col is None:
                available = [col for col in df.columns if "cluster" in col.lower()]
                if available:
                    cluster_col = available[0]
                else:
                    raise ValueError(f"Колонка с кластерами не найдена. Доступные колонки: {list(df.columns)}")

            subset = df[df[cluster_col] == by_cluster]

        elif n_samples is not None:
            subset = df.sample(n=min(n_samples, len(df)), random_state=random_seed)
        else:
            subset = df

        return subset.reset_index(drop=True)

    def save_experiment(self, name: str, additional_data: dict = None) -> str:
        """
        Сохраняет текущий эксперимент (метрики, модель, объяснения).

        Args:
            name: название эксперимента
            additional_data: дополнительные данные для сохранения

        Returns:
            str: идентификатор эксперимента
        """
        from .experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker()

        data = {
            "metrics": getattr(self, "last_metrics", {}),
            "features": getattr(self, "last_features", []),
            "model_name": getattr(self, "last_model_name", "unknown"),
            "n_samples": len(getattr(self, "last_df", pd.DataFrame())),
            "description": name,
            **(additional_data or {}),
        }

        exp_id = tracker.save_experiment(name, data)
        logger.info(f"Эксперимент сохранён: {exp_id}")
        return exp_id

    def load_experiment(self, experiment_id: str) -> dict:
        """
        Загружает сохранённый эксперимент.

        Args:
            experiment_id: идентификатор эксперимента

        Returns:
            dict: метаданные, модель, объяснения, предсказания
        """
        from .experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker()
        return tracker.load_experiment(experiment_id)

    def run_full_analysis_by_request(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Высокоуровневый метод для вызова из бэкенда через API.
        Принимает Pydantic-запрос и возвращает структурированный результат.

        Args:
            request: AnalysisRequest с данными и параметрами анализа

        Returns:
            AnalysisResult: полный результат или ошибка
        """
        try:
            # Здесь в будущем будет загрузка данных из БД по request.data_source_id
            # Пока оставляем прямой приём df (для совместимости)
            if request.df is None:
                raise ValueError("DataFrame не передан в запросе. Используйте run_full_analysis напрямую.")

            # Преобразуем Any в DataFrame, если пришёл dict/list
            if isinstance(request.df, (dict, list)):
                df = pd.DataFrame(request.df)
            else:
                df = request.df

            return self.run_full_analysis(
                df=df,
                target_col=request.target_col,
                n_clusters=request.n_clusters,
                risk_threshold=request.risk_threshold,
                corr_threshold=request.corr_threshold,
                is_synthetic=request.is_synthetic,
                use_smote=request.use_smote,
            )

        except Exception as e:
            logger.error(f"Ошибка в run_full_analysis_by_request: {str(e)}", exc_info=True)
            return AnalysisResult(
                metrics={},
                test_metrics={},
                selected_features=[],
                cluster_profiles={},
                explanations=[],
                status="error",
                message=str(e),
            )

    def run_full_analysis_by_source(
        self,
        source_id: int,
        source_type: str = "prepared_survey",
        n_clusters: int = 3,
        risk_threshold: float = 0.5,
        corr_threshold: float = 0.3,
        use_smote: bool = True,
        **kwargs,
    ) -> AnalysisResult:
        """
        Запускает анализ по ID источника данных из хранилища (заглушка).

        Args:
            source_id: идентификатор источника данных
            source_type: тип источника ('prepared_survey', 'synthetic', ...)
            n_clusters: число кластеров K-means
            risk_threshold: порог классификации риска
            corr_threshold: порог корреляционного анализа
            use_smote: применять ли SMOTE

        Returns:
            AnalysisResult: результат анализа или ошибка
        """
        try:
            # Здесь в будущем будет вызов репозитория / ORM
            # TODO Пока — заглушка. Заменить на реальную загрузку из БД
            df = self._load_prepared_data(source_id, source_type)

            if df is None or df.empty:
                return AnalysisResult(status="error", message=f"Данные с source_id={source_id} не найдены или пусты")

            # Определяем is_synthetic для get_base_features
            is_synthetic = source_type.startswith("synthetic")

            return self.run_full_analysis(
                df=df,
                target_col=kwargs.get("target_col", "risk_flag"),
                n_clusters=n_clusters,
                risk_threshold=risk_threshold,
                corr_threshold=corr_threshold,
                is_synthetic=is_synthetic,
                use_smote=use_smote,
            )

        except Exception as e:
            logger.error(f"Ошибка в run_full_analysis_by_source: {str(e)}", exc_info=True)
            return AnalysisResult(status="error", message=str(e))

    # Вспомогательный приватный метод-заглушка
    def _load_prepared_data(self, source_id: int, source_type: str) -> Optional[pd.DataFrame]:
        """
        В будущем здесь будет обращение к БД / хранилищу.
        Сейчас — заглушка для отладки.
        """
        # Пример: если source_type == "synthetic", можно загрузить из data.py
        if source_type == "synthetic":
            from .data import load_data

            result = load_data(category="grades", n_students=500)
            return result["data"]

        # Для реальных данных — здесь будет запрос в БД
        logger.warning(f"Заглушка: данные для source_id={source_id} не загружены")
        return None
