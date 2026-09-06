"""
FastAPI Backend Application for Network Traffic Classification

Exposes REST API endpoints for single and batch network traffic flow
prediction, health monitoring, and model metadata inspection.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.services.api")

# FastAPI application instance
app = FastAPI(
    title="Network Traffic Classification API",
    description=(
        "REST API serving real-time and batch intrusion detection / network traffic "
        "classification using a GPU-trained 15-class XGBoost model on CIC-IDS2017 features."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Configuration for frontend development
ALLOWED_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# EXCEPTION HANDLERS
# ============================================================


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    """Catches feature validation errors and returns a clean HTTP 400 Bad Request."""
    logger.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(TypeError)
async def handle_type_error(request: Request, exc: TypeError) -> JSONResponse:
    """Catches data type errors and returns a clean HTTP 400 Bad Request."""
    logger.warning("Type error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Formats Pydantic schema validation errors cleanly without exposing internals."""
    errors = exc.errors()
    if errors:
        first_err = errors[0]
        field_loc = " -> ".join(str(loc) for loc in first_err.get("loc", []))
        message = first_err.get("msg", "Invalid request body")
        detail = f"{field_loc}: {message}" if field_loc else message
    else:
        detail = "Invalid request payload"

    logger.warning("Request schema error on %s: %s", request.url.path, detail)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": detail},
    )


@app.exception_handler(Exception)
async def handle_generic_exception(request: Request, exc: Exception) -> JSONResponse:
    """Catches any unexpected internal server error without leaking internal traces."""
    logger.error("Unhandled internal error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."},
    )


# ============================================================
# ENDPOINTS
# ============================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Reports the service status and whether the ML model is loaded and ready.",
    responses={
        200: {"model": HealthResponse},
        503: {"model": ErrorResponse, "description": "Model failed to load"},
    },
)
async def health() -> HealthResponse:
    """Checks the health of the service and model loading status."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {model_service.load_error}",
        )
    return HealthResponse(status="healthy", model_loaded=True)


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Returns metadata about the active model, expected feature count, and class names.",
    responses={
        200: {"model": ModelInfoResponse},
        503: {"model": ErrorResponse, "description": "Model failed to load"},
    },
)
async def model_info() -> ModelInfoResponse:
    """Returns model metadata, feature count, and class mapping."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {model_service.load_error}",
        )
    info = model_service.get_model_info()
    return ModelInfoResponse(**info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single flow prediction",
    description=(
        "Classifies a single network flow given all 61 required statistical features. "
        "Returns the predicted traffic category, winning confidence score, and full "
        "probability distribution across all 15 classes."
    ),
    responses={
        200: {"model": PredictionResponse},
        400: {"model": ErrorResponse, "description": "Feature validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    """Classifies a single network flow."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {model_service.load_error}",
        )

    # All validation, feature ordering, and probabilities are handled by predictor
    result = model_service.predict_one(payload.features)
    return PredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch flow prediction",
    description=(
        "Classifies multiple network flows in a single vectorized batch execution. "
        "Returns predictions, confidences, and probability distributions for each sample."
    ),
    responses={
        200: {"model": BatchPredictionResponse},
        400: {"model": ErrorResponse, "description": "Feature validation error in one or more samples"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Classifies a batch of network flows."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {model_service.load_error}",
        )

    if not payload.samples:
        return BatchPredictionResponse(total_samples=0, predictions=[])

    # Vectorized batch prediction handled by predictor
    results = model_service.predict_batch(payload.samples)
    return BatchPredictionResponse(
        total_samples=len(results),
        predictions=[PredictionResponse(**r) for r in results],
    )
