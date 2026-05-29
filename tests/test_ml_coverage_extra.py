"""Дополнительные тесты для модулей с низким покрытием."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml_core.error_handler import safe_execute, JSONFormatter
from ml_core.llm_interface import LLMInterface
from ml_core.utils import save_plotly_fig
from ml_core.crosstab import export_crosstab


class TestErrorHandler:
    def test_safe_execute_returns_result(self):
        assert safe_execute(lambda x: x + 1, 2) == 3

    def test_safe_execute_reraises_with_log(self):
        def boom():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            safe_execute(boom, error_msg="test error")

    def test_json_formatter_includes_message(self):
        import logging

        record = logging.LogRecord(
            name="ml_core",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        payload = JSONFormatter().format(record)
        assert "hello" in payload
        assert "ml_core" in payload


class TestLLMInterface:
    def test_gigachat_stub(self):
        llm = LLMInterface(provider="gigachat")
        assert "GigaChat" in llm.complete("test prompt")

    def test_unknown_provider(self):
        llm = LLMInterface(provider="unknown")
        assert llm.complete("test") == "LLM не настроен"

    @patch("ml_core.llm_interface.requests.post")
    def test_yandex_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": {"alternatives": [{"message": {"text": "ok"}}]}},
        )
        llm = LLMInterface(provider="yandex", api_key="k", folder_id="f")
        assert llm.complete("prompt") == "ok"

    @patch("ml_core.llm_interface.requests.post")
    def test_yandex_error_status(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        llm = LLMInterface(provider="yandex", api_key="k", folder_id="f")
        assert "Ошибка LLM" in llm.complete("prompt")

    def test_interpret_clusters_builds_prompt(self):
        llm = LLMInterface(provider="gigachat")
        profiles = pd.DataFrame({"a": [1.0, 2.0]}, index=[0, 1])
        text = llm.interpret_clusters(profiles, n_clusters=2)
        assert isinstance(text, str)


class TestUtilsExtra:
    def test_save_plotly_pdf(self):
        fig = MagicMock()
        result = save_plotly_fig(fig, filename="plot", format="pdf")
        fig.write_image.assert_called_once_with("plot.pdf")
        assert result == "plot.pdf"


class TestCrosstabExport:
    def test_export_crosstab_csv(self, tmp_path, monkeypatch):
        from ml_core.config import config as ml_config

        out_dir = tmp_path / "processed"
        out_dir.mkdir(parents=True)
        monkeypatch.setattr(ml_config, "ANALYSIS_DATA_DIR", tmp_path)

        table = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        result = export_crosstab({"table": table}, filename="test_ct", format="csv")
        assert result.endswith("test_ct.csv")
        assert (out_dir / "test_ct.csv").exists()
