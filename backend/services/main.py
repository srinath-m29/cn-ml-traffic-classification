"""Versioned HTTP API for network-flow classification."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.services.model_service import model_service
from backend.services.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once while keeping the event loop responsive."""
    await asyncio.to_thread(model_service.load)
    yield


app = FastAPI(
    title="Network Traffic Classification API",
    description="Inference API for the CIC-IDS2017 network traffic classifier.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

configured_origins = os.getenv("CORS_ORIGINS")
allowed_origins: List[str] = (
    [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    if configured_origins
    else [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first.get("loc", []) if part != "body")
    message = first.get("msg", "Invalid request")
    detail = f"{location}: {message}" if location else message
    logger.info("Rejected request to %s: %s", request.url.path, detail)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": detail},
    )


@app.exception_handler(ValueError)
async def invalid_flow_error(request: Request, exc: ValueError) -> JSONResponse:
    logger.info("Invalid flow sent to %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "The server could not complete this request."},
    )


api = APIRouter(prefix="/api/v1")
legacy_api = APIRouter()


def require_model() -> None:
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The inference model is unavailable. Check the model files and server log.",
        )


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {"name": "Network Traffic Classification API", "docs": "/docs", "api": "/api/v1"}


@api.get("/health", response_model=HealthResponse, tags=["service"])
@legacy_api.get("/health", response_model=HealthResponse, tags=["service"], include_in_schema=False)
def health() -> HealthResponse:
    return HealthResponse(
        status="ready" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_type="XGBoost multiclass classifier" if model_service.is_loaded else None,
    )


@api.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
@legacy_api.get("/model-info", response_model=ModelInfoResponse, tags=["model"], include_in_schema=False)
def model_info() -> ModelInfoResponse:
    require_model()
    return ModelInfoResponse(**model_service.get_model_info())


@api.post("/predict", response_model=PredictionResponse, tags=["prediction"])
@legacy_api.post("/predict", response_model=PredictionResponse, tags=["prediction"], include_in_schema=False)
def predict(payload: PredictionRequest) -> PredictionResponse:
    require_model()
    return PredictionResponse(**model_service.predict_one(payload.features))


@api.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
@legacy_api.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"], include_in_schema=False)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    require_model()
    if not payload.samples:
        raise ValueError("Batch prediction request must contain at least one sample.")
    results = model_service.predict_batch(payload.samples)
    return BatchPredictionResponse(
        total_samples=len(results),
        predictions=[PredictionResponse(**result) for result in results],
    )


app.include_router(api)
app.include_router(legacy_api)
