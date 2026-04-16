"""Тесты для функционала анализатора (ml_core/analyzer.py)"""

import pandas as pd
import numpy as np
from ml_core.analyzer import ResearchAnalyzer


class TestResearchAnalyzer:
    """Проверка работы основного анализатора: выборки, композиты, полный пайплайн."""

    def setup_method(self):
        self.analyzer = ResearchAnalyzer()
        # Создаем стабильный мини-датасет
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "student_id": range(n),
                "f1": np.random.rand(n) * 10,
                "f2": np.random.rand(n) * 10,
                "target": np.random.randint(0, 2, n),
                "cluster": np.random.randint(0, 3, n),
            }
        )
        # Делаем target бинарным явно
        self.df.loc[: n // 2 - 1, "target"] = 0
        self.df.loc[n // 2 :, "target"] = 1

    def test_select_subset_by_condition(self):
        """Выборка по условию (query)."""
        subset = self.analyzer.select_subset(self.df, condition="f1 > 5.0")
        assert len(subset) > 0
        assert (subset["f1"] > 5.0).all()

    def test_select_subset_by_random(self):
        """Случайная выборка (n_samples)."""
        subset = self.analyzer.select_subset(self.df, n_samples=10, random_seed=42)
        assert len(subset) == 10

    def test_select_subset_by_cluster(self):
        """Выборка по кластеру."""
        subset = self.analyzer.select_subset(self.df, by_cluster=1)
        assert len(subset) > 0
        assert (subset["cluster"] == 1).all()

    def test_select_subset_fallback(self):
        """Если ничего не передано, возвращаем весь DF."""
        subset = self.analyzer.select_subset(self.df)
        assert len(subset) == len(self.df)

    def test_create_composite_score(self):
        """Создание композитной оценки."""
        weights = {"f1": 1.0, "f2": 0.5}
        df_new, score_name = self.analyzer.create_composite_score(self.df, weights, "my_score")
        assert "my_score" in df_new.columns
        assert pd.api.types.is_numeric_dtype(df_new["my_score"])

    def test_run_full_analysis_minimal(self):
        """Минимальный запуск полного анализа для покрытия ветвей инициализации."""
        result = self.analyzer.run_full_analysis(
            df=self.df,
            target_col="target",
            n_clusters=2,
            use_smote=False,  # Отключаем для скорости
            use_lr=True,
            use_rf=False,
            use_xgb=False,
        )
        assert result.status == "success"
        assert result.fig_cm is not None
        assert len(result.selected_features) > 0

    def test_run_full_analysis_smote(self):
        """Покрывает ветку с SMOTE и проверку метрик теста."""
        result = self.analyzer.run_full_analysis(
            df=self.df,
            target_col="target",
            n_clusters=2,
            use_smote=True,  # <-- Ключевая ветка
            use_lr=True,
            use_rf=True,
            use_xgb=True,
        )
        assert result.status == "success"
        # Исправлено: метрики теста находятся прямо в test_metrics
        assert "f1" in result.test_metrics
        assert "roc_auc" in result.test_metrics
