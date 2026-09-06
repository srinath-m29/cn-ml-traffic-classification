"""
Model Service Module

Manages the singleton NetworkTrafficPredictor instance and provides
a thread-safe service interface for FastAPI route handlers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.ml.predictor import NetworkTrafficPredictor

logger = logging.getLogger("backend.services.model_service")


class ModelService:
    """
    Service wrapper around NetworkTrafficPredictor.
    Ensures the model and label mappings are loaded only once upon initialization.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._predictor: Optional[NetworkTrafficPredictor] = None
        self._load_error: Optional[str] = None
        self._initialize_predictor()

    def _initialize_predictor(self) -> None:
        """Loads the predictor instance once at startup."""
        try:
            logger.info("Initializing NetworkTrafficPredictor (device=%s)...", self._device)
            self._predictor = NetworkTrafficPredictor(device=self._device)
            self._load_error = None
            logger.info(
                "NetworkTrafficPredictor loaded successfully with %d features and %d classes.",
                self._predictor.num_features,
                self._predictor.num_classes,
            )
        except Exception as exc:
            self._predictor = None
            self._load_error = str(exc)
            logger.error("Failed to initialize NetworkTrafficPredictor: %s", exc)

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying model predictor is loaded and ready."""
        return self._predictor is not None

    @property
    def load_error(self) -> Optional[str]:
        """Returns the initialization error message if model loading failed."""
        return self._load_error

    def get_predictor(self) -> NetworkTrafficPredictor:
        """Returns the underlying predictor instance or raises RuntimeError if not loaded."""
        if self._predictor is None:
            raise RuntimeError(
                f"Model predictor is not initialized. Error: {self._load_error}"
            )
        return self._predictor

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs single-flow prediction.
        Delegates all validation, feature ordering, and probability extraction
        directly to the reusable NetworkTrafficPredictor instance.
        """
        predictor = self.get_predictor()
        return predictor.predict_one(features)

    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs vectorized batch prediction.
        Delegates to the reusable NetworkTrafficPredictor instance.
        """
        predictor = self.get_predictor()
        return predictor.predict_batch(samples)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns public metadata about the loaded model without exposing internal paths."""
        predictor = self.get_predictor()
        return {
            "model_type": "XGBoost Multiclass Classifier (hist)",
            "num_features": predictor.num_features,
            "num_classes": predictor.num_classes,
            "classes": predictor.classes,
            "feature_names": predictor.feature_names,
        }


# Global singleton instance loaded once for the backend application
model_service = ModelService(device="cpu")
