"""
Модуль с моделями машинного обучения
ВАША ЗОНА ОТВЕТСТВЕННОСТИ
"""
import joblib
from datetime import datetime
import json
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import seaborn as sns
except ImportError:
    sns = None


class ModelTrainer:
    """Класс для обучения и сохранения моделей"""

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models = {
            'LR': LogisticRegression(max_iter=1000, random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGB': XGBClassifier(eval_metric='logloss', random_state=42)
        }

    def cross_validate(self, X, y, cv_folds=5, scoring='f1'):
        """
        Кросс-валидация всех моделей
        """
        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }

        return results

    def train_best_model(self, X, y, X_test=None, y_test=None):
        """
        Обучение лучшей модели (по F1-score)
        """
        # Сначала находим лучшую модель через кросс-валидацию
        cv_results = self.cross_validate(X, y)
        best_model_name = max(cv_results, key=lambda x: cv_results[x]['mean'])

        # Обучаем на всех данных
        best_model = self.models[best_model_name]
        best_model.fit(X, y)

        # Вычисляем метрики
        metrics = {'cv_results': cv_results}

        if X_test is not None and y_test is not None:
            from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

            preds = best_model.predict(X_test)
            proba = best_model.predict_proba(X_test)[:, 1]

            metrics['test'] = {
                'f1': f1_score(y_test, preds),
                'roc_auc': roc_auc_score(y_test, proba),
                'precision': precision_score(y_test, preds),
                'recall': recall_score(y_test, preds)
            }

        return best_model, best_model_name, metrics

    def save_model(self, model, model_name, metrics, features):
        """
        Сохранение модели с метаданными
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)

        # Сохраняем модель
        joblib.dump(model, model_path)

        # Сохраняем метаданные
        metadata = {
            'model_name': model_name,
            'filename': model_filename,
            'timestamp': timestamp,
            'metrics': metrics,
            'features': features,
            'model_type': type(model).__name__
        }

        metadata_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}_meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return model_path, metadata

    def load_model(self, model_path=None, model_name=None):
        """
        Загрузка модели
        """
        if model_path:
            return joblib.load(model_path)

        # Загружаем последнюю модель по имени
        if model_name:
            model_files = [f for f in os.listdir(self.models_dir)
                           if f.startswith(model_name) and f.endswith('.pkl')]
            if model_files:
                latest = sorted(model_files)[-1]
                return joblib.load(os.path.join(self.models_dir, latest))

        return None

    # ---- Обучение и сравнение моделей ----

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        models = {
            'LR': LogisticRegression(max_iter=1000),
            'RF': RandomForestClassifier(random_state=42),
            'XGB': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
        }
        results = {}
        best_model = None
        best_f1 = 0
        for name, model in models.items():
            model.fit(X_train, y_train)
            # вероятности для ROC-кривой
            proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(X_test)
            )
            preds = (proba >= 0.5).astype(int)
            roc_auc = roc_auc_score(y_test, proba)
            f1 = f1_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            results[name] = {
                'ROC-AUC': roc_auc,
                'F1': f1,
                'Precision': prec,
                'Recall': rec
            }
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
        return results, best_model

    # ---- Кросс-валидация моделей ----
    def cross_validate_models(self, X, y, models, cv_folds=5, scoring='f1'):
        """
        Проводит кросс-валидацию для набора моделей.
        Использует StratifiedKFold для корректной работы с дисбалансом классов.
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}

        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }

        return cv_results

    def print_cv_results(self, cv_results):
        """Выводит результаты кросс-валидации в читаемом виде."""
        print("\n=== Результаты кросс-валидации (F1-score) ===")
        for name, res in cv_results.items():
            print(f"{name}: {res['mean']:.4f} (+/- {res['std']:.4f})")

    # ---- Гиперпараметрический поиск для XGBoost ----
    def tune_xgboost(self, X_train, y_train, n_iter=20, cv_folds=3, random_state=42):
        """
        Оптимизация гиперпараметров XGBoost с помощью RandomizedSearchCV.
        Возвращает лучшую модель и лучшие параметры.
        """
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }

        base_model = XGBClassifier(eval_metric='logloss', random_state=random_state)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )

        search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_, search.best_score_

    def print_tuning_results(self, best_params, best_score):
        """Выводит результаты оптимизации гиперпараметров."""
        print("\n=== Оптимизация гиперпараметров XGBoost ===")
        print(f"Лучший F1-score (CV): {best_score:.4f}")
        print("Лучшие параметры:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
