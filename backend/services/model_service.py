"""Owns the single CPU inference model used by the API process."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from backend.ml.predictor import NetworkTrafficPredictor

logger = logging.getLogger(__name__)


class ModelService:
    """Loads model artifacts once during application startup and serves requests."""

    def __init__(self) -> None:
        self._predictor: Optional[NetworkTrafficPredictor] = None
        self._load_error: Optional[str] = None

    def load(self) -> None:
        """Load model files without preventing health and docs routes from starting."""
        try:
            self._predictor = NetworkTrafficPredictor(device="cpu")
            self._load_error = None
            logger.info(
                "Inference model loaded (%d features, %d classes)",
                self._predictor.num_features,
                self._predictor.num_classes,
            )
        except Exception as exc:
            self._predictor = None
            self._load_error = str(exc)
            logger.exception("Could not load inference model")

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _get_predictor(self) -> NetworkTrafficPredictor:
        if self._predictor is None:
            raise RuntimeError("The inference model is not available")
        return self._predictor

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in features.items():
            if isinstance(v, (float, int)):
                if math.isnan(v):
                    raise ValueError(f"Feature {k} contains NaN")
                if math.isinf(v):
                    raise ValueError(f"Feature {k} contains infinite value")
        return self._get_predictor().predict_one(features)

    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for sample in samples:
            for k, v in sample.items():
                if isinstance(v, (float, int)):
                    if math.isnan(v):
                        raise ValueError(f"Feature {k} contains NaN")
                    if math.isinf(v):
                        raise ValueError(f"Feature {k} contains infinite value")
        return self._get_predictor().predict_batch(samples)

    def get_model_info(self) -> Dict[str, Any]:
        predictor = self._get_predictor()
        return {
            "model_type": "XGBoost multiclass classifier",
            "num_features": predictor.num_features,
            "num_classes": predictor.num_classes,
            "classes": predictor.classes,
            "feature_names": predictor.feature_names,
        }


model_service = ModelService()
