"""Request and response contracts for the public inference API."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SAMPLES = 10_000


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: str
    model_loaded: bool
    model_type: str | None = None


class ModelInfoResponse(ApiModel):
    model_type: str
    num_features: int
    num_classes: int
    classes: List[str]
    feature_names: List[str]


class PredictionRequest(ApiModel):
    features: Dict[str, float]


class BatchPredictionRequest(ApiModel):
    samples: List[Dict[str, float]] = Field(max_length=MAX_BATCH_SAMPLES)


class PredictionResponse(ApiModel):
    prediction: str
    class_id: int
    confidence: float
    is_attack: bool
    probabilities: Dict[str, float]


class BatchPredictionResponse(ApiModel):
    total_samples: int
    predictions: List[PredictionResponse]


class ErrorResponse(ApiModel):
    detail: str
