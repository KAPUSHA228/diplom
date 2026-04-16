"""
Контракт-тесты (Consumer-Driven Contracts) для API.
Гарантируют, что бэкенд возвращает данные в формате, ожидаемом фронтендом.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Минимальный валидный запрос для теста
SAMPLE_REQUEST = {
    "df": [
        {"f1": 1.0, "f2": 2.0, "target": 0},
        {"f1": 2.0, "f2": 1.0, "target": 1},
        {"f1": 1.5, "f2": 1.5, "target": 0},
        {"f1": 2.5, "f2": 0.5, "target": 1},
        {"f1": 0.5, "f2": 2.5, "target": 0},
        {"f1": 3.0, "f2": 3.0, "target": 1},
    ],
    "target_col": "target",
    "n_clusters": 2,
    "use_smote": False,
}

REQUIRED_FIELDS = [
    "status",
    "metrics",
    "test_metrics",
    "cv_results",
    "selected_features",
    "cluster_profiles",
    "explanations",
    "fig_cm",
    "fig_roc",
    "fig_fi",
]


class TestAPIContract:
    """
    Проверяет, что ответ API соответствует контракту,
    зашитому во фронтенде (AnalysisResults.jsx).
    """

    def test_full_analysis_response_contract(self):
        # Отправляем запрос
        response = client.post("/api/v1/analyze/full", json=SAMPLE_REQUEST)

        # Проверяем статус
        assert response.status_code == 200, f"API Error: {response.text}"

        data = response.json()

        # Проверяем наличие всех обязательных полей
        for field in REQUIRED_FIELDS:
            assert field in data, f"Missing required field in response: {field}"

        # Проверяем типы данных (контракт типов)
        assert isinstance(data["status"], str)
        assert isinstance(data["metrics"], dict)
        assert isinstance(data["test_metrics"], dict)
        assert isinstance(data["cv_results"], dict)
        assert isinstance(data["selected_features"], list)
        assert isinstance(data["cluster_profiles"], dict)
        assert isinstance(data["explanations"], list)

        # Проверяем вложенность графиков (Plotly JSON structure)
        if data.get("fig_cm"):
            assert "data" in data["fig_cm"], "fig_cm missing 'data' key"
            assert "layout" in data["fig_cm"], "fig_cm missing 'layout' key"
