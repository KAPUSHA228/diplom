from typing import Dict, Any, Optional
import pandas as pd
from .config import config
from .data import load_data
from .features import add_composite_features, build_composite_score, get_base_features, preprocess_data_for_smote
from .analysis import correlation_analysis_enhanced, cluster_students, analyze_cluster_profiles
from .models import ModelTrainer
from .evaluation import generate_shap_explanations, plot_confusion_matrix, plot_roc_curves, plot_feature_importance
from .crosstab import create_crosstab
from .imputation import handle_missing_values
from .error_handler import safe_execute, logger
from .schemas import AnalysisRequest, AnalysisResult
from .timeseries import detect_negative_dynamics, analyze_student_trajectory, forecast_grades
from .text_processor import extract_text_features
from sklearn.model_selection import train_test_split


class ResearchAnalyzer:
    """Главный класс АРМ исследователя — единая точка входа.
       Предоставляет чистый и единый интерфейс ко всем возможностям ml_core.
    """

    def __init__(self):
        self.trainer = ModelTrainer(models_dir=str(config.MODELS_DIR))

    def run_full_analysis(self, df: pd.DataFrame,
                          target_col: str = "risk_flag",
                          n_clusters: int = 3,
                          risk_threshold: float = 0.5,
                          corr_threshold: float = 0.3,
                          is_synthetic: bool = False,
                          use_smote: bool = True) -> AnalysisResult:
        try:
            df = df.copy()

            if df[target_col].nunique() > 2:
                median_val = df[target_col].median()
                df[target_col] = (df[target_col] > median_val).astype(int)
                logger.info(f"Target '{target_col}' бинаризирован по медиане {median_val:.2f}")

            if 'essay_text' in df.columns:
                df = extract_text_features(df, 'essay_text')

            df = safe_execute(add_composite_features, df)
            all_features = get_base_features(df, is_synthetic=is_synthetic)

            corr_result = safe_execute(
                correlation_analysis_enhanced,
                df, all_features, target_col, corr_threshold=corr_threshold
            )

            cluster_labels, _, _ = safe_execute(
                cluster_students, df, n_clusters=n_clusters, feature_cols=all_features
            )
            df = df.copy()
            df['cluster'] = cluster_labels
            cluster_profiles = analyze_cluster_profiles(df, all_features)

            # === Train / Test сплит ===
            X = df[all_features].fillna(df[all_features].median(numeric_only=True))
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            if use_smote:
                X_train, y_train = preprocess_data_for_smote(X_train, y_train)  # новая функция


            model, model_name, metrics = self.trainer.train_best_model(
                X_train, y_train, X_test, y_test
            )

            # SHAP
            explanations = safe_execute(
                generate_shap_explanations,
                model,
                pd.DataFrame(X_test, columns=all_features),
                all_features,
                threshold=risk_threshold
            )

            self.last_df = df
            self.last_metrics = metrics
            self.last_features = all_features
            self.last_model_name = model_name
            self.last_selected_cols = all_features
            self.last_X_test = X_test
            self.last_y_test = y_test
            self.last_y_pred = model.predict(X_test)

            # Возвращаем результат
            # Сохраняем всё необходимое для визуализации
            self.last_X_test = X_test
            self.last_y_test = y_test
            self.last_y_pred = model.predict(X_test)
            self.last_model_name = model_name
            self.last_features = all_features

            # Генерируем графики сразу
            fig_cm = plot_confusion_matrix(y_test, self.last_y_pred, model_name)
            fig_roc = plot_roc_curves({model_name: model}, X_test, y_test)
            fig_fi = plot_feature_importance(model, all_features)

            # Добавляем в результат
            return AnalysisResult(
                metrics=metrics,
                test_metrics=metrics.get('test', {}),
                selected_features=all_features,
                cluster_profiles=cluster_profiles.to_dict(),
                explanations=explanations or [],
                cv_results=metrics.get('cv_results', {}),
                status="success",
                fig_cm=fig_cm,
                fig_roc=fig_roc,
                fig_fi=fig_fi,
                last_y_test=y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
                last_y_pred=self.last_y_pred.tolist()
            )

        except Exception as e:
            logger.error(f"Ошибка в run_full_analysis: {str(e)}", exc_info=True)
            return AnalysisResult(
                metrics={}, test_metrics={}, selected_features=[],
                cluster_profiles={}, explanations=[],
                status="error", message=str(e)
            )

    # Можно добавить отдельные методы:
    def create_composite_score(self, df: pd.DataFrame, feature_weights: dict, score_name: str = "custom_score"):
        """Создать композитную оценку"""
        return build_composite_score(df, feature_weights, score_name)

    def analyze_student_trajectory(self, df: pd.DataFrame, student_id, value_col: str = "avg_grade"):
        """Анализ траектории одного студента"""
        return analyze_student_trajectory(df, student_id, value_col=value_col)

    def detect_negative_dynamics(self, df: pd.DataFrame, value_col: str = "avg_grade"):
        """Выявление студентов с негативной динамикой"""
        return detect_negative_dynamics(df, value_col=value_col)

    def forecast_for_student(self, df: pd.DataFrame, student_id, future_semesters: int = 2):
        """Прогноз на будущие семестры"""
        return forecast_grades(df, student_id, future_semesters=future_semesters)

    def save_experiment(self, name: str, additional_data: dict = None) -> str:
        """Сохраняет текущий эксперимент"""
        from .experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker()

        data = {
            'metrics': getattr(self, 'last_metrics', {}),
            'features': getattr(self, 'last_features', []),
            'model_name': getattr(self, 'last_model_name', 'unknown'),
            'n_samples': len(getattr(self, 'last_df', pd.DataFrame())),
            'description': name,
            **(additional_data or {})
        }

        exp_id = tracker.save_experiment(name, data)
        logger.info(f"Эксперимент сохранён: {exp_id}")
        return exp_id

    def load_experiment(self, experiment_id: str) -> dict:
        """Загружает сохранённый эксперимент"""
        from .experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        return tracker.load_experiment(experiment_id)