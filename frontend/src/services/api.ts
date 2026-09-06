/**
 * Frontend API Service
 * Encapsulates all HTTP communication with the FastAPI backend.
 */

import type {
  BatchPredictionRequest,
  BatchPredictionResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionRequest,
  PredictionResponse,
} from '../types/api';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errorBody = await response.json();
      if (errorBody && errorBody.detail) {
        errorMessage = typeof errorBody.detail === 'string'
          ? errorBody.detail
          : JSON.stringify(errorBody.detail);
      }
    } catch {
      // If response body is not JSON, fallback to status text
    }
    throw new Error(errorMessage);
  }
  return (await response.json()) as T;
}

export const apiService = {
  /**
   * Checks backend and model status.
   */
  async getHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    return handleResponse<HealthResponse>(response);
  },

  /**
   * Retrieves model metadata, feature count, and class names.
   */
  async getModelInfo(): Promise<ModelInfoResponse> {
    const response = await fetch(`${API_BASE_URL}/model-info`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    return handleResponse<ModelInfoResponse>(response);
  },

  /**
   * Performs single-flow classification given 61 features.
   */
  async predict(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return handleResponse<PredictionResponse>(response);
  },

  /**
   * Performs vectorized batch flow classification.
   */
  async predictBatch(
    request: BatchPredictionRequest
  ): Promise<BatchPredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return handleResponse<BatchPredictionResponse>(response);
  },
};
