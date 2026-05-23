"""
Analyze Service — корреляционный анализ, EDA.
"""

import io

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

from ml_core.analysis import correlation_analysis
from shared.utils import scrub
from fastapi import APIRouter, HTTPException
import plotly.express as px
import plotly.graph_objects as go

from ml_core.crosstab import create_crosstab, simple_crosstab
from ml_core.features import create_feature_combinations
from ml_core.timeseries import get_trajectory, find_negative_dynamics, forecast_student
from ml_core.drift_detector import DataDriftDetector
from .schemas import (
    FeatureCombinationRequest,
    CrosstabRequest,
    TrajectoryRequest,
    ForecastRequest,
    DriftCheckRequest,
)
from shared.utils import safe_json_serializable

app = FastAPI(title="Analyze Service", description="Аналитика данных", version="1.0.0")

router = APIRouter(prefix="/api/v1/analyze")


@router.post("/correlation")
async def correlation(file: UploadFile = File(...), sheet_name: Optional[str] = Form(None)):
    """Корреляционный анализ (поддерживает CSV и Excel)."""
    content = await file.read()
    filename = file.filename or ""
    is_excel = filename.lower().endswith((".xlsx", ".xls"))

    df = None

    if is_excel:
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name if sheet_name else 0)
        except Exception:
            try:
                df = pd.read_excel(io.BytesIO(content))
            except Exception as e2:
                raise HTTPException(400, f"Ошибка чтения Excel: {str(e2)}")
    else:
        for enc in ["utf-8", "cp1251", "latin1"]:
            try:
                text = content.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
                for sep in [None, ";", "\t", ","]:
                    try:
                        if sep is None:
                            df = pd.read_csv(io.StringIO(text), engine="python")
                        else:
                            df = pd.read_csv(io.StringIO(text), sep=sep)
                        if df is not None and len(df.columns) > 1:
                            break
                    except Exception:
                        df = None
                if df is not None:
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue

    if df is None:
        raise HTTPException(400, "Не удалось прочитать файл.")

    if len(df) > 10000:
        raise HTTPException(400, "Data too large, use async endpoint")

    # Ищем целевую колонку
    target_col = "risk_flag"
    if target_col not in df.columns:
        possible = [c for c in df.columns if "risk" in c.lower() or "target" in c.lower()]
        target_col = possible[0] if possible else None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    service_patterns = [
        "user",
        "_id",
        "vk",
        "фио",
        "фамилия",
        "имя",
        "отчество",
        "дата",
        "date",
        "группа",
        "group",
        "курс",
        "номер",
        "cluster",
        "risk_flag",
    ]
    numeric_cols = [c for c in numeric_cols if not any(p in c.lower() for p in service_patterns)]

    if not numeric_cols:
        raise HTTPException(400, f"Нет числовых колонок для корреляции. Доступные: {list(df.columns)}")

    final_target = target_col if target_col else numeric_cols[-1]
    corr_result = correlation_analysis(df, numeric_cols, final_target)

    if not corr_result:
        raise HTTPException(400, "Не удалось построить корреляционную матрицу")

    corr = corr_result["full_matrix"]

    import plotly.express as px

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Корреляционная матрица",
    )
    fig.update_layout(height=600, width=800)

    response = {
        "correlation_matrix": scrub(corr.to_dict()),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "heatmap": scrub(fig.to_plotly_json()),
    }
    return JSONResponse(content=response)


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

        recommendations = []
        target = request.target_col
        if target and target in df_new.columns:
            for col in new_cols:
                corr = df_new[col].corr(df_new[target])
                if not pd.isna(corr):
                    recommendations.append(
                        {"name": col, "correlation": round(float(corr), 4), "abs_corr": round(abs(float(corr)), 4)}
                    )
            recommendations.sort(key=lambda x: x["abs_corr"], reverse=True)

        return {
            "data": safe_json_serializable(df_new.to_dict("records")),
            "new_columns": new_cols,
            "recommendations": recommendations[:5],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        table_dict = result["table"].T.to_dict()
        return {
            "table": safe_json_serializable(table_dict),
            "chi2_test": safe_json_serializable(result.get("chi2_test")),
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
            "table": safe_json_serializable(table_dict),
            "chi2_test": safe_json_serializable(result.get("chi2_test")),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries/trajectory")
async def student_trajectory(request: TrajectoryRequest):
    """Анализ траектории конкретного студента."""
    try:
        df = pd.DataFrame(request.df)
        result = get_trajectory(df, request.student_id, value_col=request.value_col, time_col=request.time_col)

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
            "student_id": request.student_id,
            "trend": result.get("trend"),
            "r2": result.get("r2"),
            "status": result.get("status"),
            "first_value": safe_json_serializable(result.get("first_value")),
            "last_value": safe_json_serializable(result.get("last_value")),
            "n_points": result.get("n_points"),
            "chart": safe_json_serializable(result.get("figure").to_plotly_json()) if result.get("figure") else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plot/save")
async def save_plot(data: dict):
    """
    Сохраняет Plotly JSON в PNG/SVG|/PDF и возвращает файл.
    data: {"figure": {...}, "filename": "plot", "format": "png"}
    """

    import plotly.graph_objects as go
    import os
    from fastapi.responses import FileResponse
    import tempfile
    from starlette.background import BackgroundTask

    try:
        fig = go.Figure(data=data["figure"]["data"], layout=data["figure"]["layout"])
        filename = data.get("filename", "plot")
        format = data.get("format", "png")

        # Создаём временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp:
            tmp_path = tmp.name

        # Сохраняем изображение
        if format == "png":
            fig.write_image(tmp_path, scale=2, width=800, height=600)
        elif format == "svg":
            fig.write_image(tmp_path)
        elif format == "pdf":
            fig.write_image(tmp_path)
        else:
            os.unlink(tmp_path)
            raise HTTPException(400, f"Unsupported format: {format}")

        # Функция для удаления файла после отправки
        def delete_file():
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    print(f"[DEBUG] Deleted temp file: {tmp_path}")
            except Exception as e:
                print(f"[ERROR] Failed to delete temp file: {e}")

        # Возвращаем файл с фоновой задачей на удаление
        return FileResponse(
            path=tmp_path,
            media_type=f"image/{format}",
            filename=f"{filename}.{format}",
            background=BackgroundTask(delete_file),
        )

    except Exception as e:
        print(f"[ERROR] save_plot failed: {str(e)}")
        # Если ошибка, пытаемся удалить временный файл
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, detail=str(e))


@router.post("/timeseries/negative_dynamics")
async def negative_dynamics(request: TrajectoryRequest):
    """Поиск студентов с отрицательной динамикой."""
    try:
        df = pd.DataFrame(request.df)
        result = find_negative_dynamics(df, value_col=request.value_col, time_col=request.time_col)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries/forecast")
async def forecast_student_endpoint(request: ForecastRequest):
    """Прогноз оценок студента на будущие семестры."""
    try:
        df = pd.DataFrame(request.df)
        result = forecast_student(
            df,
            request.student_id,
            value_col=request.value_col,
            time_col=request.time_col,
            periods=request.future_semesters,
        )

        student_data = df[df["student_id"] == request.student_id] if "student_id" in df.columns else df
        fig = px.line(
            student_data,
            x=request.time_col,
            y=request.value_col,
            title=f"Прогноз: студент {request.student_id}",
            markers=True,
        )
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
            "student_id": request.student_id,
            "future_periods": result.get("future_periods"),
            "predictions": result.get("predictions"),
            "trend": result.get("trend"),
            # chart можно построить на фронте или здесь
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/check")
async def check_drift(request: DriftCheckRequest):
    """Проверка дрейфа распределений между reference и current данными."""
    try:
        ref_df = pd.DataFrame(request.reference_data)
        cur_df = pd.DataFrame(request.current_data)

        detector = DataDriftDetector(ref_df, model_name=request.model_name)
        report = detector.detect_drift(cur_df)

        return {
            "overall_drift": safe_json_serializable(report["overall_drift"]),
            "drift_percentage": safe_json_serializable(report["drift_percentage"]),
            "drifted_features": safe_json_serializable(report["drifted_features"]),
            "recommendations": safe_json_serializable(report["recommendations"]),
            "feature_reports": safe_json_serializable(report.get("feature_reports", {})),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
