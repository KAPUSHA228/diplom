"""Тесты инструментов анализатора: выборки, композиты (ml_core/analyzer.py)."""

import pandas as pd
from ml_core.analyzer import ResearchAnalyzer


class TestAnalyzerTools:
    def setup_method(self):
        self.analyzer = ResearchAnalyzer()
        self.df = pd.DataFrame(
            {
                "user": [1, 2, 3, 4, 5],
                "feature_a": [10, 20, 30, 40, 50],
                "feature_b": [5, 4, 3, 2, 1],
                "cluster": [0, 0, 1, 1, 2],
            }
        )

    def test_select_subset_by_condition(self):
        """Выборка по условию (query)."""
        subset = self.analyzer.select_subset(self.df, condition="feature_a > 25")
        assert len(subset) == 3
        assert all(subset["feature_a"] > 25)

    def test_select_subset_by_random(self):
        """Случайная выборка (n_samples)."""
        subset = self.analyzer.select_subset(self.df, n_samples=2, random_seed=42)
        assert len(subset) == 2
        # Индексы должны сохраниться
        assert len(subset.index) == 2

    def test_select_subset_by_cluster(self):
        """Выборка по кластеру."""
        subset = self.analyzer.select_subset(self.df, by_cluster=1)
        assert len(subset) == 2
        assert all(subset["cluster"] == 1)

    def test_create_composite_score(self):
        """Создание композитной оценки."""
        weights = {"feature_a": 1.0, "feature_b": 0.5}
        df_new, score_name = self.analyzer.create_composite_score(self.df, weights, "my_score")

        assert "my_score" in df_new.columns
        # Проверяем, что тип данных числовой
        assert pd.api.types.is_numeric_dtype(df_new["my_score"])
        # Проверяем, что значения не все нули (значит формула сработала)
        assert df_new["my_score"].sum() != 0
