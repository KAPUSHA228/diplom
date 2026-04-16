"""Тестирование текстовых SHAP-объяснений."""

from ml_core.evaluation import generate_text_explanation


class TestSHAPTextExplanation:
    """Генерация текстовых SHAP-объяснений."""

    def test_text_with_mixed_factors(self):
        """Текст со смешанными ↑↓ факторами."""
        exp = {
            "risk_probability": 0.75,
            "risk_level": "высокое",
            "top_features": [
                {"feature": "stress", "shap_value": 0.3},
                {"feature": "satisfaction", "shap_value": -0.2},
            ],
        }
        text = generate_text_explanation(exp, target_name="риск")
        assert "75.0%" in text
        assert "↑" in text
        assert "↓" in text

    def test_text_with_no_factors(self):
        """Текст без факторов — только вероятность."""
        exp = {
            "risk_probability": 0.50,
            "risk_level": "низкое",
            "top_features": [],
        }
        text = generate_text_explanation(exp, target_name="риск")
        assert "50.0%" in text
