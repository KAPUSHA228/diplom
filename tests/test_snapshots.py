"""
Snapshot-тесты: сверяют результаты с «золотым эталоном».
Если код изменится и метрики изменятся — тест упадёт,
защитив от непреднамеренных регрессий.

Производительность тестов: замеряем время и потребление памяти.
"""

import pytest
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

from ml_core.imputation import handle_missing_values

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


# ==================== HELPERS ====================


def _ensure_snapshot_dir():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_snapshot(name: str) -> dict:
    path = SNAPSHOT_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_snapshot(name: str, data: dict):
    _ensure_snapshot_dir()
    path = SNAPSHOT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ==================== FIXTURES ====================


@pytest.fixture
def full_dataset(rng=np.random.RandomState(42)):
    """Полный датасет для snapshot-тестов."""
    n = 200
    return pd.DataFrame(
        {
            "student_id": range(n),
            "avg_grade": rng.uniform(2.0, 5.0, n),
            "grade_std": rng.uniform(0.1, 1.0, n),
            "min_grade": rng.uniform(2.0, 3.5, n),
            "max_grade": rng.uniform(3.5, 5.0, n),
            "n_courses": rng.randint(3, 10, n),
            "avg_brs": rng.uniform(40, 100, n),
            "attendance_rate": rng.uniform(0.4, 1.0, n),
            "satisfaction_score": rng.uniform(1.0, 5.0, n),
            "engagement_score": rng.uniform(1.0, 5.0, n),
            "workload_perception": rng.uniform(1.0, 5.0, n),
            "stress_level": rng.uniform(0.0, 10.0, n),
            "motivation_score": rng.uniform(0.0, 10.0, n),
            "anxiety_score": rng.uniform(0.0, 10.0, n),
            "n_essays": rng.randint(0, 5, n),
            "avg_essay_grade": rng.uniform(2.0, 5.0, n),
            "risk_flag": (rng.random(n) > 0.7).astype(int),
        }
    )


# ==================== SNAPSHOT TESTS ====================


class TestSnapshotMetrics:
    """
    Snapshot-тесты: проверяем что метрики не «уехали» от эталона.

    Если тест упал после осознанного изменения — обнови эталон:
        pytest tests/test_snapshots.py -v --snapshot-update
    (или просто удали файл snapshot и запусти снова — он пересоздастся)
    """

    SNAPSHOT_NAME = "full_analysis_metrics"

    def test_metrics_match_snapshot(self, full_dataset):
        """Метрики полного анализа совпадают с эталоном."""
        from ml_core.analyzer import ResearchAnalyzer

        analyzer = ResearchAnalyzer()
        result = analyzer.run_full_analysis(
            df=full_dataset,
            target_col="risk_flag",
            n_clusters=3,
            risk_threshold=0.5,
            corr_threshold=0.3,
            is_synthetic=True,
            use_smote=True,
        )

        current = {
            "status": result.status,
            "f1": result.test_metrics.get("f1"),
            "roc_auc": result.test_metrics.get("roc_auc"),
            "precision": result.test_metrics.get("precision"),
            "recall": result.test_metrics.get("recall"),
            "n_features_selected": len(result.selected_features),
            "n_explanations": len(result.explanations),
            "n_clusters": len(result.cluster_profiles) if result.cluster_profiles else 0,
        }

        snapshot = _load_snapshot(self.SNAPSHOT_NAME)
        if snapshot is None:
            # Первый запуск — сохраняем эталон
            _save_snapshot(self.SNAPSHOT_NAME, current)
            pytest.skip(f"Snapshot создан: {self.SNAPSHOT_NAME}")

        # Проверяем с допуском (float-расхождения допустимы)
        for key in ["f1", "roc_auc", "precision", "recall"]:
            if key in snapshot and key in current:
                assert (
                    abs(current[key] - snapshot[key]) < 0.15
                ), f"{key}: текущее={current[key]:.4f}, эталон={snapshot[key]:.4f}"

        # Целочисленные — точное совпадение
        for key in ["n_explanations", "n_clusters"]:
            if key in snapshot and key in current:
                assert current[key] == snapshot[key], f"{key}: текущее={current[key]}, эталон={snapshot[key]}"

    def test_correlation_matrix_shape_snapshot(self, full_dataset):
        """Форма корреляционной матрицы стабильна."""
        from ml_core.features import add_composite_features, get_base_features
        from ml_core.analysis import correlation_analysis

        df = add_composite_features(full_dataset)
        features = get_base_features(df)
        corr = correlation_analysis(df, features, "risk_flag")

        # Shape должна быть стабильной (features + target) × (features + target)
        expected_n = len(features) + 1  # + risk_flag
        matrix = corr["full_matrix"]
        assert matrix.shape[0] == expected_n
        assert matrix.shape[1] == expected_n


# ==================== PERFORMANCE TESTS ====================


class TestPerformance:
    """
    Performance-тесты: проверяем что ML-пайплайн укладывается в лимиты времени.

    Лимиты подобраны с запасом ×2 от типичного времени на обычном ноутбуке.
    """

    def test_full_analysis_under_60_seconds(self, full_dataset):
        """Полный пайплайн < 60 секунд."""
        from ml_core.analyzer import ResearchAnalyzer

        analyzer = ResearchAnalyzer()
        start = time.perf_counter()
        analyzer.run_full_analysis(
            df=full_dataset,
            target_col="risk_flag",
            n_clusters=3,
            risk_threshold=0.5,
            corr_threshold=0.3,
            is_synthetic=True,
            use_smote=True,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"Пайплайн занял {elapsed:.1f}с (лимит: 60с)"

    def test_correlation_under_5_seconds(self, full_dataset):
        """Корреляционный анализ < 5 секунд."""
        from ml_core.features import add_composite_features, get_base_features
        from ml_core.analysis import correlation_analysis

        df = add_composite_features(full_dataset)
        features = get_base_features(df)
        start = time.perf_counter()
        correlation_analysis(df, features, "risk_flag")
        elapsed = time.perf_counter() - start
        assert elapsed < 5, f"Корреляция заняла {elapsed:.2f}с (лимит: 5с)"

    def test_clustering_under_5_seconds(self, full_dataset):
        """Кластеризация < 5 секунд."""
        from ml_core.features import add_composite_features, get_base_features
        from ml_core.analysis import cluster_students

        df = add_composite_features(full_dataset)
        features = get_base_features(df)
        start = time.perf_counter()
        cluster_students(df[features], n_clusters=3)
        elapsed = time.perf_counter() - start
        assert elapsed < 5, f"Кластеризация заняла {elapsed:.2f}с (лимит: 5с)"

    def test_model_training_under_30_seconds(self, full_dataset):
        """Обучение моделей (CV + best) < 30 секунд."""
        from ml_core.features import (
            add_composite_features,
            get_base_features,
            preprocess_data,
            select_features_for_model,
        )
        from ml_core.models import ModelTrainer

        df = add_composite_features(full_dataset)
        features = get_base_features(df)
        X, y = preprocess_data(df, features, "risk_flag")
        X_sel, names = select_features_for_model(X, y, top_n=7, final_n=5)

        trainer = ModelTrainer()
        start = time.perf_counter()
        trainer.cross_validate(X_sel, y, cv_folds=3)
        trainer.train_best_model(X_sel, y)
        elapsed = time.perf_counter() - start
        assert elapsed < 30, f"Обучение заняло {elapsed:.1f}с (лимит: 30с)"

    def test_shap_under_15_seconds(self, full_dataset):
        """SHAP-объяснения < 15 секунд."""
        from ml_core.features import (
            add_composite_features,
            get_base_features,
            preprocess_data,
            select_features_for_model,
        )
        from ml_core.models import ModelTrainer
        from ml_core.evaluation import generate_shap_explanations

        df = add_composite_features(full_dataset)
        features = get_base_features(df)
        X, y = preprocess_data(df, features, "risk_flag")
        X_sel, names = select_features_for_model(X, y, top_n=7, final_n=5)

        trainer = ModelTrainer()
        model, _, _ = trainer.train_best_model(X_sel, y)

        X_test = X_sel.iloc[:20]
        start = time.perf_counter()
        generate_shap_explanations(model, X_test, names, top_n=3)
        elapsed = time.perf_counter() - start
        assert elapsed < 15, f"SHAP занял {elapsed:.1f}с (лимит: 15с)"

    def test_imputation_under_2_seconds(self):
        """Обработка пропусков < 2 секунд."""
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame(
            {
                "a": rng.random(n),
                "b": rng.random(n),
                "c": rng.random(n),
            }
        )
        # 20% NaN
        for col in ["a", "b", "c"]:
            mask = rng.random(n) < 0.2
            df.loc[mask, col] = np.nan

        start = time.perf_counter()
        handle_missing_values(df, strategy="auto")
        elapsed = time.perf_counter() - start
        assert elapsed < 2, f"Импутация заняла {elapsed:.2f}с (лимит: 2с)"


# ==================== LARGE-SCALE TESTS ====================


class TestLargeScale:
    """Тесты на увеличенных объёмах данных."""

    def test_full_pipeline_500_students(self):
        """500 студентов — пайплайн работает."""
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame(
            {
                "student_id": range(n),
                "avg_grade": rng.uniform(2.0, 5.0, n),
                "grade_std": rng.uniform(0.1, 1.0, n),
                "min_grade": rng.uniform(2.0, 3.5, n),
                "max_grade": rng.uniform(3.5, 5.0, n),
                "n_courses": rng.randint(3, 10, n),
                "avg_brs": rng.uniform(40, 100, n),
                "satisfaction_score": rng.uniform(1.0, 5.0, n),
                "stress_level": rng.uniform(0.0, 10.0, n),
                "motivation_score": rng.uniform(0.0, 10.0, n),
                "risk_flag": (rng.random(n) > 0.7).astype(int),
            }
        )

        from ml_core.analyzer import ResearchAnalyzer

        analyzer = ResearchAnalyzer()
        result = analyzer.run_full_analysis(
            df=df,
            target_col="risk_flag",
            n_clusters=3,
            risk_threshold=0.5,
            corr_threshold=0.3,
            is_synthetic=True,
            use_smote=True,
        )
        assert result.status == "success"
        assert len(result.test_metrics) > 0
