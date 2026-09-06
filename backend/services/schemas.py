"""
Pydantic Schemas for Network Traffic Classification API

Defines typed request and response data models for single and batch predictions,
health checks, and model metadata endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str = Field(..., description="Service status indicator (e.g. 'healthy')")
    model_loaded: bool = Field(..., description="True if the ML model is loaded and ready")


class ModelInfoResponse(BaseModel):
    """Response schema for the /model-info endpoint."""
    model_type: str = Field(..., description="Type and configuration of the ML model")
    num_features: int = Field(..., description="Exact number of features expected (61)")
    num_classes: int = Field(..., description="Number of target traffic classes (15)")
    classes: List[str] = Field(..., description="List of all 15 class labels in index order")
    feature_names: List[str] = Field(..., description="List of all 61 expected feature names in order")


class PredictionRequest(BaseModel):
    """Request schema for single-flow traffic prediction."""
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping exactly 61 network flow feature names to their numeric values.",
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch traffic prediction."""
    samples: List[Dict[str, Any]] = Field(
        ...,
        description="List of dictionaries, each mapping 61 network flow feature names to their numeric values.",
    )


class PredictionResponse(BaseModel):
    """Response schema for a single flow classification result."""
    prediction: str = Field(..., description="Human-readable predicted class name (e.g. 'BENIGN', 'DDoS')")
    class_id: int = Field(..., description="Integer class identifier (0 to 14)")
    confidence: float = Field(..., description="Confidence probability for the predicted class (0.0 to 1.0)")
    is_attack: bool = Field(..., description="True if traffic is classified as an attack, False if BENIGN")
    probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution across all 15 network traffic classes"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch flow classification results."""
    total_samples: int = Field(..., description="Total number of evaluated samples in the batch")
    predictions: List[PredictionResponse] = Field(
        ..., description="List of prediction results for each submitted sample"
    )


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    detail: str = Field(..., description="Concise description of the error")
