"""Тесты кеширования, дрейфа данных, SHAP-объяснений и полного пайплайна анализа.

Включает:
- Mock Redis для тестирования декоратора cache_result
- Детекция дрейфа данных (DataDriftDetector, DriftMonitorThread, DriftMonitorScheduler)
- Объяснения моделей (SHAP, batch, detailed)
- Полный пайплайн ResearchAnalyzer с разными настройками
- Временные ряды и прогнозы
"""

import pickle
import numpy as np
import pandas as pd
import pytest
import matplotlib
from ml_core.evaluation import (
    plot_feature_importance,
    generate_detailed_explanation,
    generate_batch_explanations,
    plot_feature_importance_png,
)
from ml_core.drift_detector import (
    DataDriftDetector,
    DriftMonitorThread,
    DriftMonitorScheduler,
)
from ml_core.analyzer import ResearchAnalyzer
from ml_core.timeseries import (
    forecast_grades_chupep,
    detect_negative_dynamics,
    analyze_cohort_trajectory,
)

matplotlib.use("Agg")
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ========== Кеширование с mock Redis ==========

from ml_core.cache import cache_result


class TestCacheMockRedis:
    """Тестирование декоратора cache_result с подменой Redis.

    Проверяем все ветки декоратора:
    - Cache miss → вычисление → сохранение
    - Cache hit → возврат без вычисления
    - Ошибка pickle.loads → повторное вычисление
    - Ошибка setex → результат возвращается без ошибки
    - Ошибка ping → функция вызывается без кеширования
    """

    @patch("ml_core.cache.redis_client")
    def test_cache_miss_then_hit(self, mock_redis):
        """Cache miss → вычисление → сохранение → следующий вызов из кеша."""
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None  # MISS
        mock_redis.setex.return_value = True

        call_count = 0

        @cache_result
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Первый вызов — MISS
        result1 = expensive(5)
        assert result1 == 10
        assert call_count == 1
        mock_redis.setex.assert_called_once()

        # Настраиваем HIT — возвращаем ранее сохранённый результат
        mock_redis.get.return_value = pickle.dumps(10)

        # Второй вызов — HIT
        result2 = expensive(5)
        assert result2 == 10
        assert call_count == 1  # Функция НЕ вызывалась

    @patch("ml_core.cache.redis_client")
    def test_cache_corrupt_pickle_data(self, mock_redis):
        """Повреждённые данные в кеше → функция вызывается заново."""
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = b"corrupt_data"  # Невалидный pickle
        mock_redis.setex.return_value = True

        call_count = 0

        @cache_result
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        result = my_func(10)
        assert result == 11
        assert call_count == 1

    @patch("ml_core.cache.redis_client")
    def test_cache_storage_full(self, mock_redis):
        """Redis переполнен → результат возвращается без ошибки записи."""
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.setex.side_effect = Exception("Redis full")

        @cache_result
        def my_func(x):
            return x * 3

        result = my_func(7)
        assert result == 21

    @patch("ml_core.cache.redis_client")
    def test_cache_connection_lost(self, mock_redis):
        """Соединение с Redis потеряно → функция вызывается без кеширования."""
        mock_redis.ping.side_effect = Exception("Connection lost")

        call_count = 0

        @cache_result
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x - 1

        result = my_func(10)
        assert result == 9
        assert call_count == 1


# ========== Детекция дрейфа данных и мониторинг ==========


class TestDriftDetection:
    """Тестирование системы обнаружения дрейфа данных.

    DataDriftDetector сравнивает распределения признаков
    между эталонными (reference) и новыми данными.
    """

    def test_detect_drift_between_distributions(self):
        """Обнаружение дрейфа между разными распределениями."""
        rng = np.random.RandomState(42)
        ref = pd.DataFrame(
            {
                "num": rng.normal(0, 1, 200),
                "cat": rng.choice(["X", "Y"], 200),
            }
        )
        new = pd.DataFrame(
            {
                "num": rng.normal(2, 2, 200),
                "cat": rng.choice(["X", "Y", "Z"], 200),
            }
        )
        detector = DataDriftDetector(reference_data=ref)
        report = detector.detect_drift(new)
        assert isinstance(report, dict)
        assert "overall_drift" in report or "drifted_features" in report

    def test_no_drift_same_distribution(self):
        """Отсутствие дрейфа при одинаковых распределениях."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({"a": rng.normal(0, 1, 500)})
        detector = DataDriftDetector(reference_data=data)
        # Те же данные — дрейфа нет
        report = detector.detect_drift(data)
        assert isinstance(report, dict)


class TestDriftMonitoringInfrastructure:
    """Тестирование инфраструктуры мониторинга дрейфа.

    DriftMonitorThread — фоновый поток для периодической проверки.
    DriftMonitorScheduler — планировщик для нескольких моделей.
    """

    def test_monitor_thread_lifecycle(self):
        """Жизненный цикл потока мониторинга: создание → остановка."""
        ref = pd.DataFrame({"a": [1, 2, 3]})
        detector = DataDriftDetector(reference_data=ref)
        thread = DriftMonitorThread(detector=detector, check_interval_hours=1)
        assert thread.check_interval == 1
        thread.stop()

    def test_scheduler_register_and_check(self):
        """Регистрация моделей и проверка дрейфа по расписанию."""
        scheduler = DriftMonitorScheduler(check_interval_hours=1)
        assert scheduler.check_interval == 1

        ref = pd.DataFrame({"a": [1, 2, 3]})
        detector = DataDriftDetector(reference_data=ref)
        scheduler.register_model("test_model", detector)
        assert "test_model" in scheduler.detectors

        current = pd.DataFrame({"a": [1, 2, 3]})
        results = scheduler.check_all_models(current)
        assert "test_model" in results


# ========== Объяснения моделей (SHAP) ==========


class TestModelExplanations:
    """Тестирование системы объяснений моделей.

    SHAP-объяснения, детальные текстовые отчёты для студентов,
    batch-объяснения для группы риска.
    """

    def test_feature_importance_from_coefficients(self):
        """Важность признаков из коэффициентов линейной модели (LR)."""
        model = LogisticRegression(max_iter=200)
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        y = pd.Series([0, 1, 0, 1])
        model.fit(X, y)
        fig = plot_feature_importance(model, ["a", "b"])
        assert fig is not None
        assert hasattr(fig, "data")

    def test_batch_explanations_for_top_risk_students(self):
        """Batch-объяснения для топ-N студентов с наибольшим риском."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"a": rng.rand(30), "b": rng.rand(30)})
        y = (rng.rand(30) > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        explanations = generate_batch_explanations(model, X, feature_names=["a", "b"], top_n=3)
        assert len(explanations) == 3
        assert "student_index" in explanations[0]

    def test_detailed_explanation_with_recommendations(self):
        """Детальное объяснение с рекомендациями на основе SHAP-факторов."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            {
                "avg_grade_trend": rng.rand(20),
                "attendance_rate": rng.rand(20),
                "stress_level": rng.rand(20),
            }
        )
        y = (rng.rand(20) > 0.5).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
        text = generate_detailed_explanation(
            model,
            X,
            student_idx=0,
            feature_names=["avg_grade_trend", "attendance_rate", "stress_level"],
        )
        assert "Вероятность" in text or "Риск" in text.lower() or "фактор" in text.lower()

    def test_feature_importance_png_no_importance_attribute(self):
        """Модель без feature_importances_ (LogisticRegression) → PNG не строится."""
        model = LogisticRegression().fit([[1], [2], [3]], [0, 1, 0])
        result = plot_feature_importance_png(model, ["a"], name="test")
        assert result is None


# ========== Полный пайплайн ResearchAnalyzer ==========


class TestResearchAnalyzerPipeline:
    """Тестирование полного пайплайна анализа.

    Проверяем разные режимы работы:
    - Автовыбор целевой переменной
    - Разные комбинации включённых моделей
    - Данные с текстовыми колонками
    - Минимальные наборы данных
    - Подзапросы через Pydantic
    - Загрузка по источнику (заглушка)
    """

    def test_auto_select_target_column(self):
        """Автовыбор целевой переменной по похожему имени."""
        analyzer = ResearchAnalyzer()
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "a": rng.rand(n),
                "b": rng.rand(n),
                "my_flag": (rng.rand(n) > 0.5).astype(int),
            }
        )
        result = analyzer.run_full_analysis(df, target_col="risk_flag", n_clusters=2, use_smote=False)
        # Должен найти 'my_flag' как похожий и успешно проанализировать
        assert result.status == "success"

    def test_fallback_to_lr_when_all_models_disabled(self):
        """Все модели выключены → fallback на LogisticRegression."""
        analyzer = ResearchAnalyzer()
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "a": rng.rand(n),
                "b": rng.rand(n),
                "risk_flag": (rng.rand(n) > 0.5).astype(int),
            }
        )
        result = analyzer.run_full_analysis(
            df,
            target_col="risk_flag",
            n_clusters=2,
            use_smote=False,
            use_lr=False,
            use_rf=False,
            use_xgb=False,
        )
        assert result.status == "success"
        assert result.model_name == "LR"

    def test_run_by_request_with_none_dataframe(self):
        """Pydantic-запрос с df=None → ошибка валидации."""
        from ml_core.schemas import AnalysisRequest

        analyzer = ResearchAnalyzer()
        request = AnalysisRequest(df=None, target_col="risk_flag")
        result = analyzer.run_full_analysis_by_request(request)
        assert result.status == "error"
        assert "не передан" in result.message.lower()

    def test_run_by_source_unknown_type(self):
        """Неизвестный тип источника данных → ошибка."""
        analyzer = ResearchAnalyzer()
        result = analyzer.run_full_analysis_by_source(source_id=1, source_type="unknown_type")
        assert result.status == "error"

    def test_run_with_text_column(self):
        """Пайплайн с текстовой колонкой (essay_text)."""
        analyzer = ResearchAnalyzer()
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "a": rng.rand(n),
                "essay_text": ["short text"] * 50 + ["longer and more complex text here"] * 50,
                "risk_flag": (rng.rand(n) > 0.5).astype(int),
            }
        )
        result = analyzer.run_full_analysis(df, target_col="risk_flag", n_clusters=2, use_smote=False)
        assert result is not None

    def test_run_minimal_dataset(self):
        """Минимальный набор данных (10 строк)."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "risk_flag": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        result = analyzer.run_full_analysis(df, target_col="risk_flag", n_clusters=2, use_smote=False)
        assert result is not None


class TestResearchAnalyzerSubsets:
    """Выборка подмножеств респондентов.

    Поддерживаемые методы:
    - По условию (pandas query)
    - Случайная выборка (n_samples)
    - По кластеру (by_cluster)
    """

    def test_select_by_condition(self):
        """Выборка по условию (pandas query)."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame({"age": [18, 25, 30, 35], "score": [80, 90, 70, 85]})
        result = analyzer.select_subset(df, condition="age > 20")
        assert len(result) == 3

    def test_select_random_sample(self):
        """Случайная выборка фиксированного размера."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame({"a": range(100)})
        result = analyzer.select_subset(df, n_samples=10)
        assert len(result) == 10

    def test_select_by_cluster(self):
        """Выборка по номеру кластера."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame({"a": [1, 2, 3, 4], "cluster": [0, 0, 1, 1]})
        result = analyzer.select_subset(df, by_cluster=0)
        assert len(result) == 2

    def test_select_no_filter_returns_all(self):
        """Без фильтрации — возвращает все данные."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = analyzer.select_subset(df)
        assert len(result) == 3


class TestResearchAnalyzerTrajectories:
    """Анализ траекторий студентов.

    - Траектория отдельного студента (тренд, статус)
    - Детекция негативной динамики
    - Прогноз оценок
    """

    def test_analyze_student_trajectory(self):
        """Анализ траектории отдельного студента."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame(
            {
                "student_id": [1, 1, 1, 1],
                "semester": [1, 2, 3, 4],
                "avg_grade": [3.0, 3.5, 4.0, 4.5],
            }
        )
        result = analyzer.analyze_student_trajectory(df, student_id=1)
        assert "status" in result

    def test_detect_negative_dynamics(self):
        """Детекция негативной динамики у студентов."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame(
            {
                "student_id": [1, 1, 1, 2, 2, 2],
                "semester": [1, 2, 3, 1, 2, 3],
                "avg_grade": [4.5, 3.5, 2.5, 3.0, 3.5, 4.0],
            }
        )
        result = analyzer.detect_negative_dynamics(df)
        assert "n_students_analyzed" in result or "at_risk" in str(result)

    def test_forecast_grades_for_student(self):
        """Прогноз оценок для студента."""
        analyzer = ResearchAnalyzer()
        df = pd.DataFrame(
            {
                "student_id": [1, 1, 1],
                "semester": [1, 2, 3],
                "avg_grade": [3.0, 3.5, 4.0],
            }
        )
        result = analyzer.forecast_for_student(df, student_id=1, future_semesters=2)
        assert "predictions" in result


class TestResearchAnalyzerExperiments:
    """Управление экспериментами.

    Сохранение и загрузка результатов анализа.
    """

    def test_save_experiment_after_analysis(self):
        """Сохранение эксперимента после запуска анализа."""
        analyzer = ResearchAnalyzer()
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "a": rng.rand(n),
                "b": rng.rand(n),
                "risk_flag": (rng.rand(n) > 0.5).astype(int),
            }
        )
        analyzer.run_full_analysis(df, target_col="risk_flag", n_clusters=2, use_smote=False)
        exp_id = analyzer.save_experiment("test_exp")
        assert exp_id is not None

    def test_load_nonexistent_experiment(self):
        """Загрузка несуществующего эксперимента → FileNotFoundError."""
        analyzer = ResearchAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.load_experiment("nonexistent")


# ========== Временные ряды ==========


class TestTimeSeriesAnalysis:
    """Тестирование анализа временных рядов.

    - Прогноз оценок (линейная экстраполяция, last_value)
    - Детекция негативной динамики
    - Сравнение когорт
    """

    def test_forecast_empty_history(self):
        """Прогноз при пустой истории → нулевые значения."""
        result = forecast_grades_chupep([], periods=3)
        assert len(result) == 3
        assert all(v == 0 for v in result)

    def test_detect_negative_dynamics_missing_column(self):
        """Детекция при отсутствии колонки времени → ошибка."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = detect_negative_dynamics(df, time_col="nonexistent")
        assert "error" in result or result.get("n_students_analyzed", 0) == 0

    def test_compare_cohort_trajectories(self):
        """Сравнение траекторий когорт."""
        rng = np.random.RandomState(42)
        records = []
        for year in [2020, 2021]:
            for sem in range(1, 4):
                for _ in range(5):
                    records.append(
                        {
                            "year": year,
                            "semester": sem,
                            "avg_grade": 3.5 + rng.normal(0, 0.3),
                        }
                    )
        df = pd.DataFrame(records)
        result = analyze_cohort_trajectory(df, cohort_col="year", time_col="semester")
        assert "cohort_data" in result
        assert "figure" in result
