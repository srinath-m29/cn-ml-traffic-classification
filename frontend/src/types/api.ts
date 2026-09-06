/**
 * API Type Definitions for Network Traffic Classification
 * Matches the FastAPI Pydantic backend models exactly.
 */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ModelInfoResponse {
  model_type: string;
  num_features: number;
  num_classes: number;
  classes: string[];
  feature_names: string[];
}

export interface PredictionRequest {
  features: Record<string, number>;
}

export interface PredictionResponse {
  prediction: string;
  class_id: number;
  confidence: number;
  is_attack: boolean;
  probabilities: Record<string, number>;
}

export interface BatchPredictionRequest {
  samples: Record<string, number>[];
}

export interface BatchPredictionResponse {
  total_samples: number;
  predictions: PredictionResponse[];
}

export interface ApiError {
  detail: string;
  status?: number;
}
