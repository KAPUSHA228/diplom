from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from ml_core.analyzer import ResearchAnalyzer
from ml_core.error_handler import logger
from ml_core.imputation import handle_missing_values, detect_outliers
from ml_core.crosstab import create_crosstab, simple_crosstab
from ml_core.features import create_feature_combinations
from ml_core.timeseries import (
    analyze_student_trajectory,
    detect_negative_dynamics,
    forecast_grades,
)
from ml_core.drift_detector import DataDriftDetector
from ml_core.experiment_tracker import ExperimentTracker
from ml_core.loader import get_sheet_preview
from ..schemas import AnalysisRequest, CompositeRequest, SubsetRequest, FeatureCombinationRequest
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/analyze")

# Один экземпляр анализатора на всё приложение
analyzer = ResearchAnalyzer()


def _safe_json_serializable(obj):
    """Рекурсивная очистка данных от nan/inf/numpy-типов для JSON."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return _safe_json_serializable(obj.tolist())
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _safe_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json_serializable(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if hasattr(obj, "to_dict"):
        return _safe_json_serializable(obj.to_dict())
    return obj


# ==================== Excel Processing ====================


class ExcelPreviewRequest(BaseModel):
    sheet_name: str


class ExcelProcessRequest(BaseModel):
    sheet_name: str
    mapping_config: Optional[Dict[str, Any]] = None


@router.post("/excel/preview")
async def excel_preview(file: UploadFile = File(...), sheet_name: str = Form("0")):
    """
    Возвращает метаданные листа: типы колонок и уникальные значения для настройки маппинга.
    """
    import tempfile
    import os

    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # === РЕШЕНИЕ ПРОБЛЕМЫ "Worksheet named '0' not found" ===
        # Если передан индекс (цифра), находим реальное имя листа
        xl = pd.ExcelFile(tmp_path)
        all_sheets = xl.sheet_names
        target_name = sheet_name

        print(f"DEBUG excel_preview: получен sheet_name='{sheet_name}'")
        print(f"DEBUG excel_preview: все листы в файле: {all_sheets}")

        if sheet_name.isdigit():
            idx = int(sheet_name)
            if 0 <= idx < len(all_sheets):
                target_name = all_sheets[idx]
            else:
                raise ValueError(f"Лист с индексом {idx} не найден")
        else:
            # Проверяем точное совпадение
            if sheet_name not in all_sheets:
                # Пробуем найти по частичному совпадению (без учёта регистра и пробелов)
                normalized = sheet_name.strip().lower()
                found = [s for s in all_sheets if s.strip().lower() == normalized]
                if found:
                    target_name = found[0]
                    print(f"DEBUG excel_preview: найдено частичное совпадение: '{sheet_name}' → '{target_name}'")
                else:
                    # Если не нашли — ищем по началу имени (первые символы)
                    found_prefix = [s for s in all_sheets if s.lower().startswith(normalized[:6])]
                    if found_prefix:
                        target_name = found_prefix[0]
                        print(f"DEBUG excel_preview: найдено по префиксу: '{sheet_name}' → '{target_name}'")
                    else:
                        print(
                            f"DEBUG excel_preview: лист '{sheet_name}' НЕ НАЙДЕН! Доступны: {all_sheets}. Использую первый лист '{all_sheets[0]}'"
                        )
                        target_name = all_sheets[0]
            else:
                print(f"DEBUG excel_preview: точное совпадение '{sheet_name}'")

        print(f"DEBUG excel_preview: ФИНАЛЬНОЕ target_name='{target_name}'")

        preview = get_sheet_preview(tmp_path, target_name)
        xl.close()  # Обязательно закрываем файл перед удалением!
        return _safe_json_serializable(preview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка превью: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass  # На Windows иногда глючит блокировка


@router.post("/excel/process")
async def excel_process(
    file: UploadFile = File(...),
    sheet_name: str = Form("0"),
    sheet_group: str = Form("numeric"),
    mapping_config: Optional[str] = Form(None),
):
    """
    Обрабатывает лист Excel с учетом пользовательских настроек маппинга.
    """
    import tempfile
    import os
    import json

    config = None
    if mapping_config:
        try:
            config = json.loads(mapping_config)
        except Exception:
            pass

    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # === РЕШЕНИЕ ПРОБЛЕМЫ "Worksheet named '0' not found" ===
        xl = pd.ExcelFile(tmp_path)
        target_name = sheet_name
        if sheet_name.isdigit():
            idx = int(sheet_name)
            if 0 <= idx < len(xl.sheet_names):
                target_name = xl.sheet_names[idx]
            else:
                raise ValueError(f"Лист с индексом {idx} не найден")
        xl.close()

        from ml_core.loader import preprocess_sheet

        # Читаем файл уже по правильному имени
        df = pd.read_excel(tmp_path, sheet_name=target_name)

        # Передаем реальное имя листа в процессор
        df_processed, msg = preprocess_sheet(df, sheet_group, target_name, mapping_config=config)

        # Заменяем NaN и inf перед сериализацией
        df_clean = df_processed.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.where(df_clean.notna(), None)

        return {"message": msg, "data": _safe_json_serializable(df_clean.to_dict("records")), "rows": len(df_processed)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ==================== Дополнительные Pydantic-схемы ====================


class ImputationRequest(BaseModel):
    df: List[Dict[str, Any]]
    strategy: str = "auto"
    threshold: float = 30.0


class CrosstabRequest(BaseModel):
    df: List[Dict[str, Any]]
    row_var: str
    col_var: str
    values: Optional[str] = None
    aggfunc: str = "count"
    normalize: bool = False
    n_bins: int = Field(4, ge=2, le=10, description="Количество групп при автоматическом биннинге")
    bin_method: str = Field(
        "cut",
        pattern="^(cut|qcut)$",
        description="Метод разбиения: 'cut' - равные интервалы, 'qcut' - равное количество наблюдений",
    )


class TrajectoryRequest(BaseModel):
    df: List[Dict[str, Any]]
    student_id: Optional[Any] = None  # Опционально: для негативной динамики не нужен
    value_col: str = "avg_grade"
    time_col: str = "semester"


class ForecastRequest(BaseModel):
    df: List[Dict[str, Any]]
    student_id: Any
    value_col: str = "avg_grade"
    time_col: str = "semester"
    future_semesters: int = 2


class DriftCheckRequest(BaseModel):
    reference_data: List[Dict[str, Any]]
    current_data: List[Dict[str, Any]]
    model_name: str = "unknown"


class ExperimentSaveRequest(BaseModel):
    name: str
    metrics: Dict[str, Any] = {}
    features: List[str] = []
    description: str = ""
    config: Dict[str, Any] = {}  # <--- Добавили поле конфигурации


@router.post("/full")
async def full_analysis(request: AnalysisRequest):
    try:
        df = pd.DataFrame(request.df)

        target = request.target_col
        if not target or target not in df.columns:
            # Ищем первую числовую колонку (исключая ID)
            num_cols = df.select_dtypes(include="number").columns
            exclude = ["user", "_id", "vk"]
            possible = [c for c in num_cols if not any(x in c.lower() for x in exclude)]
            target = possible[0] if possible else num_cols[0]
        # Передаем все параметры, которые пришли от фронтенда
        print(f"DEBUG full_analysis: n_clusters={request.n_clusters}, target={target}")
        result = analyzer.run_full_analysis(
            df=df,
            target_col=target,
            n_clusters=request.n_clusters,
            corr_threshold=request.corr_threshold,
            use_smote=request.use_smote,
            # Передаем настройки моделей из сайдбара
            use_lr=request.use_lr,
            use_rf=request.use_rf,
            use_xgb=request.use_xgb,
            optimization_metric=request.optimization_metric,
        )

        # === АГРЕССИВНАЯ ОЧИСТКА ДАННЫХ ===
        import numpy as np

        def scrub(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return scrub(obj.tolist())
            if isinstance(obj, np.generic):
                return obj.item()
            if hasattr(obj, "to_dict"):
                return scrub(obj.to_dict())  # DataFrame -> Dict
            if isinstance(obj, dict):
                return {k: scrub(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [scrub(v) for v in obj]
            return obj

        # Сериализуем графики Plotly в JSON
        def plot_to_json(fig):
            if fig is None:
                return None
            try:
                return scrub(fig.to_plotly_json())  # Чистим данные графика!
            except Exception:
                return None

        # Генерируем predictions
        predictions = []
        model_name = result.model_name or "RF"
        if result.last_y_pred is not None and analyzer.last_X_test is not None:
            model = analyzer.trainer.models.get(model_name)
            if model and hasattr(model, "predict_proba"):
                # Обернём X_test в DataFrame с именами колонок
                X_test_df = pd.DataFrame(analyzer.last_X_test, columns=result.selected_features or None)
                proba = model.predict_proba(X_test_df)[:, 1].tolist()
                y_pred_list = (
                    result.last_y_pred if isinstance(result.last_y_pred, list) else result.last_y_pred.tolist()
                )
                for i in range(len(y_pred_list)):
                    predictions.append(
                        {
                            "student_index": i,
                            "prediction": int(y_pred_list[i]),
                            "probability": round(float(proba[i]), 4),
                        }
                    )

        # result — это объект Pydantic (AnalysisResult), обращаемся через точку
        # Возвращаем СЛОВАРЬ, а не объект модели, чтобы избежать валидации Pydantic

        # Добавляем конфигурацию для воспроизводимости
        analysis_config = {
            "model_name": result.model_name or "unknown",
            "target_col": target,
            "n_samples": len(df),
            "n_features": len(result.selected_features),
            "n_clusters": request.n_clusters,
            "use_smote": request.use_smote,
            "corr_threshold": request.corr_threshold,
            "optimization_metric": request.optimization_metric,
        }

        return {
            "status": result.status,
            "message": result.message,
            "config": analysis_config,
            "metrics": scrub(result.metrics),
            "test_metrics": scrub(result.test_metrics),
            "cv_results": scrub(result.cv_results),
            "selected_features": scrub(result.selected_features),
            "cluster_profiles": scrub(result.cluster_profiles),
            "explanations": scrub(result.explanations),
            "predictions": _safe_json_serializable(predictions),
            # Передаем графики
            "fig_cm": plot_to_json(result.fig_cm),
            "fig_roc": plot_to_json(result.fig_roc),
            "fig_fi": plot_to_json(result.fig_fi),
            "fig_clusters": plot_to_json(result.fig_clusters),
            "fig_corr": plot_to_json(result.fig_corr),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@router.post("/composite/create")
async def create_composite(request: CompositeRequest):
    try:
        df = pd.DataFrame(request.df)
        df_new, score_name = analyzer.create_composite_score(df, request.feature_weights, request.score_name)

        # 1. Статистика
        stats = df_new[score_name].describe()

        # 2. Корреляции с другими числовыми колонками
        numeric_cols = df_new.select_dtypes(include="number").columns
        correlations = {}
        if score_name in df_new.columns:
            for col in numeric_cols:
                if col != score_name:
                    corr = df_new[score_name].corr(df_new[col])
                    correlations[col] = round(corr, 4) if not pd.isna(corr) else 0

        # 3. Сортируем по абсолютному значению
        sorted_corrs = dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "score_name": score_name,
            "statistics": _safe_json_serializable(stats.to_dict()),
            "correlations": _safe_json_serializable(sorted_corrs),
            "values": _safe_json_serializable(df_new[score_name].tolist()),  # Для гистограммы
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subset/select")
async def select_subset(request: SubsetRequest):
    try:
        df = pd.DataFrame(request.df)

        print("Subset request columns:", df.columns.tolist())
        subset = analyzer.select_subset(
            df,
            condition=request.condition,
            n_samples=request.n_samples,
            by_cluster=request.by_cluster,
            random_seed=request.random_seed,
        )
        return {"count": len(subset), "data": _safe_json_serializable(subset.to_dict("records"))}
    except Exception as e:
        logger.error(f"Error in select_subset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Комбинации признаков ====================


@router.post("/combinations/create")
async def create_combinations(request: FeatureCombinationRequest):
    """Создание комбинированных признаков с оценкой важности."""
    try:
        df = pd.DataFrame(request.df)
        df_new = create_feature_combinations(
            df,
            numerical_cols=request.numerical_cols,
            text_cols=request.text_cols,
            max_pairs=request.max_pairs,
        )
        new_cols = [c for c in df_new.columns if c not in df.columns]

        # Умный отбор: корреляция с целевой переменной
        recommendations = []
        target = request.target_col
        if target and target in df_new.columns:
            for col in new_cols:
                # Считаем корреляцию Пирсона (автоматически игнорирует NaN)
                corr = df_new[col].corr(df_new[target])
                if not pd.isna(corr):
                    recommendations.append(
                        {"name": col, "correlation": round(float(corr), 4), "abs_corr": round(abs(float(corr)), 4)}
                    )
            # Сортируем по силе связи (по модулю)
            recommendations.sort(key=lambda x: x["abs_corr"], reverse=True)

        return {
            "data": _safe_json_serializable(df_new.to_dict("records")),
            "new_columns": new_cols,
            "recommendations": recommendations[:5],  # Топ-5 для пользователя
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Обработка пропусков ====================


@router.post("/imputation/handle")
async def handle_imputation(request: ImputationRequest):
    """Обработка пропусков и выбросов в данных."""
    try:
        df = pd.DataFrame(request.df)
        df_clean, report = handle_missing_values(df, strategy=request.strategy, threshold=request.threshold)
        outliers = detect_outliers(df_clean)

        return {
            "data": _safe_json_serializable(df_clean.to_dict("records")),
            "report": _safe_json_serializable(report),
            "outliers": _safe_json_serializable({k: v for k, v in outliers.items() if v.get("n_outliers", 0) > 0}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Кросс-таблицы ====================


@router.post("/crosstab")
async def build_crosstab(request: CrosstabRequest):
    """Построение кросс-таблицы с χ²-тестом."""
    try:
        df = pd.DataFrame(request.df)
        result = create_crosstab(
            df,
            request.row_var,
            request.col_var,
            values=request.values,
            aggfunc=request.aggfunc,
            normalize=request.normalize,
            auto_bin=True,
            n_bins=request.n_bins,
            bin_method=request.bin_method,
        )
        print(f"[Crosstab Backend] table shape: {result['table'].shape}, first 3 rows:\n{result['table'].head(3)}")
        # Транспонируем: строки должны быть внешними ключами, колонки — внутренними
        table_dict = result["table"].T.to_dict()
        print(f"[Crosstab Backend] serialized keys: {list(table_dict.keys())[:3]}...")
        return {
            "table": _safe_json_serializable(table_dict),
            "chi2_test": _safe_json_serializable(result.get("chi2_test")),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crosstab/simple")
async def build_simple_crosstab(request: CrosstabRequest):
    """Упрощённая кросс-таблица (только таблица + χ²)."""
    try:
        df = pd.DataFrame(request.df)
        result = simple_crosstab(df, request.row_var, request.col_var)
        table_dict = result["table"].T.to_dict()
        return {
            "table": _safe_json_serializable(table_dict),
            "chi2_test": _safe_json_serializable(result.get("chi2_test")),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Временные ряды ====================


@router.post("/timeseries/trajectory")
async def student_trajectory(request: TrajectoryRequest):
    """Анализ траектории конкретного студента."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        df = pd.DataFrame(request.df)
        result = analyze_student_trajectory(
            df, request.student_id, time_col=request.time_col, value_col=request.value_col
        )

        # Генерируем график траектории
        student_data = df[df["student_id"] == request.student_id] if "student_id" in df.columns else df
        fig = px.line(
            student_data,
            x=request.time_col,
            y=request.value_col,
            title=f"Траектория: студент {request.student_id}",
            markers=True,
        )
        fig.add_trace(
            go.Scatter(
                x=student_data[request.time_col],
                y=[result.get("first_value", 0)] * len(student_data),
                mode="lines",
                name="Начало",
                line=dict(dash="dash", color="red"),
            )
        )

        return {
            "trend": result.get("trend"),
            "status": result.get("status"),
            "first_value": _safe_json_serializable(result.get("first_value")),
            "last_value": _safe_json_serializable(result.get("last_value")),
            "chart": _safe_json_serializable(fig.to_plotly_json()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries/negative_dynamics")
async def find_negative_dynamics(request: TrajectoryRequest):
    """Поиск студентов с отрицательной динамикой."""
    try:
        df = pd.DataFrame(request.df)
        result = detect_negative_dynamics(
            df, student_id_col="student_id", time_col=request.time_col, value_col=request.value_col
        )
        return {
            "n_students_analyzed": _safe_json_serializable(result.get("n_students_analyzed", 0)),
            "at_risk_count": len(result.get("at_risk_students", [])),
            "risk_percentage": _safe_json_serializable(result.get("risk_percentage", 0)),
            "at_risk_students": _safe_json_serializable(
                result.get("at_risk_students", []).to_dict("records")
                if isinstance(result.get("at_risk_students"), pd.DataFrame)
                else result.get("at_risk_students", [])
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries/forecast")
async def forecast_student(request: ForecastRequest):
    """Прогноз оценок студента на будущие семестры."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        df = pd.DataFrame(request.df)
        result = forecast_grades(
            df,
            request.student_id,
            time_col=request.time_col,
            value_col=request.value_col,
            future_semesters=request.future_semesters,
        )

        # График: история + прогноз
        student_data = df[df["student_id"] == request.student_id] if "student_id" in df.columns else df
        fig = px.line(
            student_data,
            x=request.time_col,
            y=request.value_col,
            title=f"Прогноз: студент {request.student_id}",
            markers=True,
        )
        # Добавляем прогноз пунктиром
        future_x = result.get("future_semesters", [])
        future_y = result.get("predictions", [])
        if future_x and future_y:
            fig.add_trace(
                go.Scatter(
                    x=future_x,
                    y=future_y,
                    mode="lines+markers",
                    name="Прогноз",
                    line=dict(dash="dash", color="orange"),
                    marker=dict(size=10),
                )
            )

        return {
            "future_semesters": _safe_json_serializable(result.get("future_semesters", [])),
            "predictions": _safe_json_serializable(result.get("predictions", [])),
            "chart": _safe_json_serializable(fig.to_plotly_json()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Дрейф данных ====================


@router.post("/drift/check")
async def check_drift(request: DriftCheckRequest):
    """Проверка дрейфа распределений между reference и current данными."""
    try:
        ref_df = pd.DataFrame(request.reference_data)
        cur_df = pd.DataFrame(request.current_data)

        detector = DataDriftDetector(ref_df, model_name=request.model_name)
        report = detector.detect_drift(cur_df)

        return {
            "overall_drift": _safe_json_serializable(report["overall_drift"]),
            "drift_percentage": _safe_json_serializable(report["drift_percentage"]),
            "drifted_features": _safe_json_serializable(report["drifted_features"]),
            "recommendations": _safe_json_serializable(report["recommendations"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Эксперименты ====================


@router.get("/metrics/history")
async def metrics_history():
    """История метрик моделей из логов."""
    import os

    try:
        from ml_core.config import config as ml_config

        metrics_path = f"{ml_config.LOGS_DIR}/model_metrics.csv"
        if not os.path.exists(metrics_path):
            # Попробуем дефолтный путь
            metrics_path = "logs/model_metrics.csv"
        if not os.path.exists(metrics_path):
            return {"metrics": [], "message": "Файл model_metrics.csv не найден"}

        df = pd.read_csv(metrics_path)
        return {
            "metrics": _safe_json_serializable(df.to_dict("records")),
            "total": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/save")
async def save_experiment(request: ExperimentSaveRequest):
    """Сохранение эксперимента."""
    try:
        tracker = ExperimentTracker()

        # Собираем полные данные: метрики + признаки + описание + КОНФИГ
        exp_data = {
            "metrics": request.metrics,
            "features": request.features,
            "description": request.description,
            "config": request.config or {},  # <--- Сохраняем конфигурацию
        }

        exp_id = tracker.save_experiment(request.name, exp_data)
        return {"experiment_id": exp_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/list")
async def list_experiments(limit: int = 20):
    """Список сохранённых экспериментов."""
    try:
        tracker = ExperimentTracker()
        experiments = tracker.list_experiments(limit=limit)
        return {
            "experiments": _safe_json_serializable(experiments.to_dict("records")),
            "total": len(experiments),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Загрузка конкретного эксперимента по ID."""
    try:
        tracker = ExperimentTracker()
        data = tracker.load_experiment(experiment_id)
        return data
    except Exception:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
