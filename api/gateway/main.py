"""
API Gateway — точка входа для фронтенда.
Маршрутизирует запросы к соответствующим микросервисам.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.excel_service.main import router as excel_router
from api.analyze_service.main import router as analyze_router
from api.ml_service.main import router as ml_router
from api.experiments_service.main import router as experiments_router

app = FastAPI(
    title="ML Analytics Gateway",
    description="API Gateway для ML-модуля системы мониторинга академических рисков",
    version="1.0.0",
)

# CORS для React (пока разрешаем всё, потом ограничим)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(excel_router)  # /excel/preview, /excel/process, /imputation/handle
app.include_router(analyze_router)  # /crosstab, /timeseries/*, /drift/check
app.include_router(ml_router)  # /full, /composite, /subset
app.include_router(experiments_router)  # /experiments/*, /metrics/history


@app.get("/", response_model=dict)
async def root():
    return {
        "service": "ML Analytics Gateway",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "gateway"}
