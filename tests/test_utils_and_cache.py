"""Тесты для ml_core/utils.py и ml_core/cache.py"""

from unittest.mock import MagicMock, patch
import pandas as pd
from ml_core.utils import save_plotly_fig
from ml_core.cache import _make_key, cache_result, redis_client, REDIS_AVAILABLE


class TestSavePlotlyFig:
    """Тесты функции save_plotly_fig"""

    def test_save_png(self):
        """PNG: к filename добавляется .png"""
        fig = MagicMock()
        result = save_plotly_fig(fig, filename="test_plot", format="png")
        fig.write_image.assert_called_once_with("test_plot.png", scale=2)
        assert result == "test_plot.png"

    def test_save_svg(self):
        """SVG: к filename добавляется .svg"""
        fig = MagicMock()
        result = save_plotly_fig(fig, filename="test_plot", format="svg")
        fig.write_image.assert_called_once_with("test_plot.svg")
        assert result == "test_plot.svg"

    def test_default_filename(self):
        """По умолчанию filename='plot', format='png'"""
        fig = MagicMock()
        result = save_plotly_fig(fig)
        fig.write_image.assert_called_once_with("plot.png", scale=2)
        assert result == "plot.png"


class TestMakeKey:
    """Тесты генерации ключей кеширования"""

    def test_key_has_prefix(self):
        key = _make_key("test_func", 1, "hello", verbose=True)
        assert key.startswith("ml_cache:test_func:")

    def test_different_args_different_keys(self):
        key1 = _make_key("func", 1)
        key2 = _make_key("func", 2)
        assert key1 != key2

    def test_dataframe_arg_creates_key(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        key = _make_key("cluster", df, n_clusters=3)
        assert key.startswith("ml_cache:cluster:")

    def test_same_args_same_key(self):
        df = pd.DataFrame({"x": [10, 20]})
        key1 = _make_key("my_func", df, k=5)
        key2 = _make_key("my_func", df, k=5)
        assert key1 == key2


class TestCacheResultDecorator:
    """Тесты декоратора cache_result"""

    def test_no_redis_fallback(self):
        """Если redis_client is None — функция вызывается без кеширования."""
        if REDIS_AVAILABLE and redis_client is not None:
            return  # Тест только для случая без Redis

        call_count = 0

        @cache_result
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        result = my_func(10)
        assert result == 11
        assert call_count == 1

        # Повторный вызов — снова вызывает функцию (нет кеша)
        result = my_func(10)
        assert result == 11
        assert call_count == 2

    @patch("ml_core.cache.redis_client")
    def test_redis_ping_fails(self, mock_redis):
        """Если ping падает — функция вызывается без кеширования."""
        mock_redis.ping.side_effect = Exception("Connection refused")

        call_count = 0

        @cache_result
        def fail_func(x):
            nonlocal call_count
            call_count += 1
            return x * 3

        result = fail_func(7)
        assert result == 21
        assert call_count == 1
