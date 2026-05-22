import time
from typing import Optional
import pandas as pd
from .config import config
from .features import add_composite_features, build_composite_score, get_base_features, preprocess_data_for_smote
from .analysis import correlation_analysis, cluster_students, analyze_cluster_profiles, plot_corr_heatmap
from .models import ModelTrainer
from .evaluation import generate_shap_explanations, plot_confusion_matrix, plot_roc_curves, plot_feature_importance
from .error_handler import safe_execute, logger
from api.ml_service.schemas import AnalysisResponse
from .text_processor import extract_text_features
from sklearn.model_selection import train_test_split
import numpy as np
import cProfile
import pstats
from io import StringIO
from ml_core.logger import MLLogger as ml_log


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
        n_features_to_select: int = 7,
        shap_top_n: int = 5,
        use_hp_tuning: bool = False,
        n_iter_tuning: int = 20,
        progress_callback=None,
    ) -> AnalysisResponse:
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
            use_lr: использование Linear Regression,
            use_rf: использование Random Forest,
            use_xgb: использование XGBoost,
            optimization_metric: основная выбранная метрика,
            n_features_to_select: выборка признаков,
            shap_top_n: ограничение для объяснений SHAP,
            use_hp_tuning: использование тонкой настройки XGBoost,
            n_iter_tuning: количество итераций тонкой настройки,
            progress_callback: функция callback для прогресс-бара,

        Returns:
            AnalysisResult: полный результат анализа
        """

        def report(stage, progress):
            print(f"🔵 REPORT CALLED: stage={stage}, progress={progress}")  # ← отладочный вывод
            if progress_callback:
                print(f"🔵 CALLING CALLBACK: {stage} - {progress}%")  # ← отладочный вывод
                progress_callback(stage, progress)
            logger.info(f"[PROGRESS] {stage}: {progress}%")
            time.sleep(0.05)

        try:
            df = df.copy()

            report("Проверка целевой переменной", 5)
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

            report("Создание композитных признаков", 10)
            df = safe_execute(add_composite_features, df)

            report("Определение признаков", 15)
            all_features = get_base_features(df, is_synthetic=is_synthetic)
            if target_col in all_features:
                print(f"[WARNING] Target column {target_col} is in features! Removing...")
                all_features.remove(target_col)

            if not all_features:
                raise ValueError(
                    "Нет признаков для анализа. "
                    "В датасете не осталось числовых колонок после исключения целевой переменной. "
                    "Убедитесь, что файл содержит хотя бы один числовой признак."
                )
            if n_features_to_select and n_features_to_select < len(all_features):
                report(f"Отбор {n_features_to_select} лучших признаков", 18)

                from sklearn.feature_selection import SelectKBest, f_classif

                # Временно убираем target из признаков (если затесался)
                temp_features = [f for f in all_features if f != target_col]

                X_temp = df[temp_features].fillna(df[temp_features].median())
                y_temp = df[target_col]

                selector = SelectKBest(f_classif, k=min(n_features_to_select, len(temp_features)))
                selector.fit(X_temp, y_temp)

                # Получаем маску отобранных признаков
                selected_mask = selector.get_support()
                all_features = [f for f, flag in zip(temp_features, selected_mask) if flag]

                logger.info(f"Отобрано {len(all_features)} признаков из {len(temp_features)}")
                report(f"Отобрано признаков: {len(all_features)}", 19)

            report("Корреляционный анализ", 20)
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

            report("Кластеризация студентов", 30)
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

            report("Разделение на train/test", 40)
            # === Train / Test сплит ===
            # X = df[all_features].fillna(df[all_features].median(numeric_only=True))
            X = df[all_features]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            if use_smote:
                report("Балансировка классов (SMOTE)", 45)
                X_train, y_train = preprocess_data_for_smote(X_train, y_train)  # новая функция

            report("Выбор лучшей модели", 50)
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
            if use_hp_tuning and "XGB" in active_models:
                report("Оптимизация гиперпараметров XGBoost", 55)

                best_model, best_params, best_score = self.trainer.tune_xgboost(
                    X_train, y_train, n_iter=n_iter_tuning, cv_folds=3  # можно вынести в параметр
                )

                # Заменяем XGB модель на оптимизированную
                self.trainer.models["XGB"] = best_model
                logger.info(f"Лучшие параметры XGBoost: {best_params}")
                logger.info(f"Лучший F1 (CV): {best_score:.4f}")

                report("Оптимизация завершена", 58)

            report("Обучение моделей", 60)
            model, model_name, results = self.trainer.train_models_parallel(
                X_train, y_train, X_test, y_test, scoring=optimization_metric, cv_folds=5
            )
            cv_results = {}
            for name, res in results.items():
                # Достаём metrics из res
                metrics_dict = res.get("metrics", {})

                # Проверяем наличие cv_scores в metrics_dict
                if "cv_scores" in metrics_dict:
                    cv_results[name] = {
                        "mean": metrics_dict.get("cv_mean", 0),
                        "std": metrics_dict.get("cv_std", 0),
                        "scores": metrics_dict.get("cv_scores", []),
                    }
                else:
                    print(f"⚠️ Нет cv_scores для {name}: {metrics_dict.keys()}")

            print(f"🔍 DEBUG cv_results after build: {cv_results}")

            test_metrics = {
                "f1": results[model_name]["metrics"].get("f1", 0),
                "precision": results[model_name]["metrics"].get("precision", 0),
                "recall": results[model_name]["metrics"].get("recall", 0),
                "roc_auc": results[model_name]["metrics"].get("roc_auc", 0),
            }

            metrics = {"test": test_metrics, "cv_results": cv_results}
            ml_log.log_model_metrics(model_name, metrics)

            # Возвращаем оригинальный набор моделей обратно
            self.trainer.models = original_models_backup

            # ==========================================================
            predictions = []
            if hasattr(model, "predict_proba") and X_test is not None:
                proba = model.predict_proba(X_test)[:, 1].tolist()
                y_pred_list = model.predict(X_test).tolist()

                for i in range(len(y_pred_list)):
                    predictions.append(
                        {
                            "student_index": i,
                            "prediction": int(y_pred_list[i]),
                            "probability": round(float(proba[i]), 4),
                        }
                    )

            # SHAP
            report("SHAP объяснения", 80)
            explanations = safe_execute(
                generate_shap_explanations,
                model,
                pd.DataFrame(X_test, columns=all_features),
                all_features,
                threshold=risk_threshold,
                target_name=target_col,
                top_n=shap_top_n,
                n_students=5,
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
                report("Построение графиков", 90)
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
            report("Завершение", 100)
            print("🔍 DEBUG cv_results:", metrics.get("cv_results", {}))
            print("🔍 DEBUG cv_results keys:", metrics.get("cv_results", {}).keys())
            # Добавляем в результат
            return AnalysisResponse(
                metrics=metrics,
                target_col=target_col,
                test_metrics=metrics.get("test", {}),
                selected_features=all_features,
                cluster_profiles=(
                    cluster_profiles.to_dict() if hasattr(cluster_profiles, "to_dict") else cluster_profiles
                ),
                explanations=explanations or [],
                predictions=predictions,
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
            return AnalysisResponse(
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

    def profile_analysis(self, df: pd.DataFrame, target_col: str, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = self.run_full_analysis(df, target_col, **kwargs)

        profiler.disable()
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)

        with open("profile_report.txt", "w") as f:
            f.write(s.getvalue())

        return result
