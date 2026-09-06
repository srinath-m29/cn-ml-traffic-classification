"""
Network Traffic Predictor Module

Reusable inference engine for multi-class network traffic classification
and intrusion detection using the pre-trained XGBoost model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb


def normalize_display_label(raw_label: str) -> str:
    """
    Normalizes label strings by removing Unicode corruption artifacts (such as \\ufffd)
    while preserving the true semantic label name.

    Example:
        'Web Attack \ufffd Brute Force' -> 'Web Attack - Brute Force'
    """
    if not isinstance(raw_label, str):
        raw_label = str(raw_label)

    # Replace unicode replacement characters, non-breaking spaces, em-dashes
    cleaned = raw_label.replace("\ufffd", "-").replace("\u2013", "-").replace("\u2014", "-")
    cleaned = " ".join(cleaned.split())
    if "Web Attack" in cleaned:
        parts = [p.strip() for p in cleaned.split("-") if p.strip()]
        if len(parts) == 2:
            return f"{parts[0]} - {parts[1]}"
    return cleaned


class NetworkTrafficPredictor:
    """
    Production-ready inference class for network traffic classification.

    Loads the pre-trained 15-class XGBoost model and associated label mappings once,
    validates incoming network flow features against the exact 61 features expected by
    the model in strict order, and outputs predictions, confidences, and full probability
    distributions.
    """

    DEFAULT_BENIGN_LABEL = "BENIGN"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        mapping_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        strict_features: bool = True,
        normalize_labels: bool = True,
    ) -> None:
        """
        Initialize the predictor and load model and label mappings into memory.

        Args:
            model_path: Optional custom path to the XGBoost model JSON.
                        Defaults to <project_root>/models/xgboost_multiclass_gpu.json.
            mapping_path: Optional custom path to the multiclass_label_mapping.json.
                          Defaults to <project_root>/models/multiclass_label_mapping.json.
            device: Computing device for XGBoost inference ('cpu' or 'cuda').
                    Defaults to 'cpu' for universal cross-platform portability.
            strict_features: If True, rejects feature dictionaries containing unexpected extra keys.
            normalize_labels: If True, uses clean normalized display labels (e.g. replacing \\ufffd with '-').
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.strict_features = strict_features
        self.normalize_labels = normalize_labels
        self.device = device.lower()

        # Resolve paths relative to project root
        self.model_path = (
            Path(model_path).resolve()
            if model_path is not None
            else (self.project_root / "models" / "xgboost_multiclass_gpu.json")
        )
        self.mapping_path = (
            Path(mapping_path).resolve()
            if mapping_path is not None
            else (self.project_root / "models" / "multiclass_label_mapping.json")
        )

        # Validate existence of files
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained model file not found: {self.model_path}")
        if not self.mapping_path.is_file():
            raise FileNotFoundError(f"Label mapping file not found: {self.mapping_path}")

        # Load label mapping
        self._load_label_mapping()

        # Load XGBoost model
        self._load_model()

    def _load_label_mapping(self) -> None:
        """Loads and parses the official multiclass label mapping JSON."""
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        if "id_to_label" in mapping_data:
            raw_id_to_label = {int(k): str(v) for k, v in mapping_data["id_to_label"].items()}
        elif all(str(k).isdigit() for k in mapping_data.keys()):
            raw_id_to_label = {int(k): str(v) for k, v in mapping_data.items()}
        else:
            raise ValueError(f"Unable to parse id-to-label structure from {self.mapping_path}")

        self._raw_id_to_label: Dict[int, str] = dict(sorted(raw_id_to_label.items()))
        self._num_classes = len(self._raw_id_to_label)

        # Build normalized display labels
        self._display_id_to_label: Dict[int, str] = {
            cid: normalize_display_label(lbl) if self.normalize_labels else lbl
            for cid, lbl in self._raw_id_to_label.items()
        }
        self._display_label_to_id: Dict[str, int] = {
            lbl: cid for cid, lbl in self._display_id_to_label.items()
        }

    def _load_model(self) -> None:
        """Loads the XGBoost classifier model and extracts feature metadata."""
        try:
            self.model = xgb.XGBClassifier(device=self.device)
            self.model.load_model(str(self.model_path))
        except Exception as e:
            # If cuda was requested but failed, fallback to cpu with a clear message
            if self.device != "cpu":
                self.device = "cpu"
                self.model = xgb.XGBClassifier(device="cpu")
                self.model.load_model(str(self.model_path))
            else:
                raise RuntimeError(f"Failed to load XGBoost model from {self.model_path}: {e}") from e

        booster = self.model.get_booster()
        feature_names = booster.feature_names
        if not feature_names:
            raise RuntimeError(f"Loaded XGBoost model from {self.model_path} does not contain feature names.")

        self._feature_names: List[str] = list(feature_names)
        self._feature_set = set(self._feature_names)
        self._num_features = len(self._feature_names)

        if self._num_features != 61:
            raise ValueError(f"Expected model to contain 61 features, but found {self._num_features}.")

    @property
    def feature_names(self) -> List[str]:
        """Returns a copy of the 61 feature names in the exact order expected by the model."""
        return list(self._feature_names)

    @property
    def num_features(self) -> int:
        """Returns the number of features expected by the model (61)."""
        return self._num_features

    @property
    def num_classes(self) -> int:
        """Returns the number of target classes (15)."""
        return self._num_classes

    @property
    def classes(self) -> List[str]:
        """Returns the list of class names in index order (0 to 14)."""
        return [self._display_id_to_label[cid] for cid in sorted(self._display_id_to_label.keys())]

    @property
    def id_to_label(self) -> Dict[int, str]:
        """Returns a mapping from integer class ID to human-readable class name."""
        return dict(self._display_id_to_label)

    @property
    def label_to_id(self) -> Dict[str, int]:
        """Returns a mapping from human-readable class name to integer class ID."""
        return dict(self._display_label_to_id)

    @property
    def raw_id_to_label(self) -> Dict[int, str]:
        """Returns the exact raw label mapping from the source JSON file without normalization."""
        return dict(self._raw_id_to_label)

    def validate_features(self, features: Dict[str, Any]) -> List[float]:
        """
        Validates a single flow feature dictionary against model requirements.

        Performs:
        1. Type checking (must be a dictionary).
        2. Unexpected feature detection.
        3. Missing feature detection.
        4. Count validation (must have 61 features).
        5. Numeric conversion, NaN checks, and Infinity checks.

        Returns:
            A list of 61 float values in the exact order expected by the model.
        """
        if not isinstance(features, dict):
            raise TypeError(f"Expected a feature dictionary, but received {type(features).__name__}")

        # Check for unexpected extra features if strict mode is enabled
        if self.strict_features:
            unexpected = [k for k in features.keys() if k not in self._feature_set]
            if unexpected:
                if len(unexpected) == 1:
                    raise ValueError(f"Unexpected feature: {unexpected[0]}")
                raise ValueError(f"Unexpected features ({len(unexpected)}): {', '.join(unexpected)}")

        # Check for missing required features
        missing = [f for f in self._feature_names if f not in features]
        if missing:
            if len(missing) == 1:
                raise ValueError(f"Missing required feature: {missing[0]}")
            raise ValueError(f"Missing required features ({len(missing)}): {', '.join(missing)}")

        if len(features) != self._num_features and self.strict_features:
            raise ValueError(f"Expected {self._num_features} features, but received {len(features)}")

        # Validate numeric values and extract in strict feature order
        ordered_values: List[float] = []
        for feature in self._feature_names:
            val = features[feature]
            try:
                num_val = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Feature '{feature}' value '{val}' cannot be converted to numeric"
                ) from exc

            if math.isnan(num_val):
                raise ValueError(f"Feature {feature} contains NaN")
            if math.isinf(num_val):
                raise ValueError(f"Feature {feature} contains infinite value")

            ordered_values.append(num_val)

        return ordered_values

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs network traffic classification on a single flow dictionary.

        Args:
            features: Dictionary containing all 61 network flow features.

        Returns:
            API-friendly dictionary with:
                - prediction: Human-readable class name (e.g. 'DDoS', 'BENIGN')
                - class_id: Integer class ID (0 to 14)
                - confidence: Probability of the predicted class (0.0 to 1.0)
                - is_attack: True if prediction != 'BENIGN', False otherwise
                - probabilities: Dictionary of probabilities for all 15 classes
        """
        row = self.validate_features(features)
        X = np.array([row], dtype=np.float32)

        probabilities = self.model.predict_proba(X)[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        prediction_label = self._display_id_to_label[class_id]
        is_attack = bool(prediction_label != self.DEFAULT_BENIGN_LABEL)

        all_probabilities: Dict[str, float] = {
            self._display_id_to_label[cid]: float(prob)
            for cid, prob in enumerate(probabilities)
        }

        return {
            "prediction": prediction_label,
            "class_id": class_id,
            "confidence": confidence,
            "is_attack": is_attack,
            "probabilities": all_probabilities,
        }

    def predict_batch(
        self, samples: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Performs high-throughput batch classification.

        Args:
            samples: A list of feature dictionaries OR a pandas DataFrame.

        Returns:
            List of prediction dictionaries corresponding to each input sample.
        """
        if isinstance(samples, pd.DataFrame):
            if len(samples) == 0:
                return []

            # Check columns
            sample_cols = set(samples.columns)
            missing = [f for f in self._feature_names if f not in sample_cols]
            if missing:
                if len(missing) == 1:
                    raise ValueError(f"Missing required feature: {missing[0]}")
                raise ValueError(f"Missing required features ({len(missing)}): {', '.join(missing)}")

            if self.strict_features:
                unexpected = [c for c in samples.columns if c not in self._feature_set]
                if unexpected:
                    if len(unexpected) == 1:
                        raise ValueError(f"Unexpected feature: {unexpected[0]}")
                    raise ValueError(f"Unexpected features ({len(unexpected)}): {', '.join(unexpected)}")

            # Extract in exact feature order
            sub_df = samples[self._feature_names].copy()

            # Check for non-numeric, NaNs, and Infs
            numeric_df = sub_df.apply(pd.to_numeric, errors="coerce")
            nan_cols = numeric_df.columns[numeric_df.isna().any()].tolist()
            if nan_cols:
                raise ValueError(f"Feature {nan_cols[0]} contains NaN")

            inf_mask = np.isinf(numeric_df.to_numpy())
            if inf_mask.any():
                col_idx = np.where(inf_mask)[1][0]
                raise ValueError(f"Feature {self._feature_names[col_idx]} contains infinite value")

            X = numeric_df.to_numpy(dtype=np.float32)

        elif isinstance(samples, list):
            if len(samples) == 0:
                return []
            rows = [self.validate_features(sample) for sample in samples]
            X = np.array(rows, dtype=np.float32)

        else:
            raise TypeError(f"Expected list of dicts or pandas DataFrame, got {type(samples).__name__}")

        # Vectorized batch prediction
        all_probabilities = self.model.predict_proba(X)

        results: List[Dict[str, Any]] = []
        for probs in all_probabilities:
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            prediction_label = self._display_id_to_label[class_id]
            is_attack = bool(prediction_label != self.DEFAULT_BENIGN_LABEL)

            class_probs = {
                self._display_id_to_label[cid]: float(p)
                for cid, p in enumerate(probs)
            }

            results.append({
                "prediction": prediction_label,
                "class_id": class_id,
                "confidence": confidence,
                "is_attack": is_attack,
                "probabilities": class_probs,
            })

        return results
