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
from ml_core.timeseries import (
    analyze_student_trajectory,
    detect_negative_dynamics,
    forecast_grades,
)
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
        result = analyze_student_trajectory(
            df, request.student_id, time_col=request.time_col, value_col=request.value_col
        )

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
            "first_value": safe_json_serializable(result.get("first_value")),
            "last_value": safe_json_serializable(result.get("last_value")),
            "chart": safe_json_serializable(fig.to_plotly_json()),
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
            "n_students_analyzed": safe_json_serializable(result.get("n_students_analyzed", 0)),
            "at_risk_count": len(result.get("at_risk_students", [])),
            "risk_percentage": safe_json_serializable(result.get("risk_percentage", 0)),
            "at_risk_students": safe_json_serializable(
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
        df = pd.DataFrame(request.df)
        result = forecast_grades(
            df,
            request.student_id,
            time_col=request.time_col,
            value_col=request.value_col,
            future_semesters=request.future_semesters,
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
            "future_semesters": safe_json_serializable(result.get("future_semesters", [])),
            "predictions": safe_json_serializable(result.get("predictions", [])),
            "chart": safe_json_serializable(fig.to_plotly_json()),
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
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
