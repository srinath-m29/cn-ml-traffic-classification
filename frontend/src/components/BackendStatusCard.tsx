import React from 'react';
import type { HealthResponse } from '../types/api';

interface BackendStatusCardProps {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

export const BackendStatusCard: React.FC<BackendStatusCardProps> = ({
  health,
  loading,
  error,
  onRetry,
}) => {
  const isConnected = !loading && !error && health?.status === 'healthy';
  const isModelLoaded = isConnected && health?.model_loaded === true;

  return (
    <div className="card status-card">
      <div className="card-header">
        <h2 className="card-title">1. Backend Status</h2>
        <button
          className="btn btn-secondary btn-sm"
          onClick={onRetry}
          disabled={loading}
          title="Refresh server status"
        >
          {loading ? 'Checking...' : 'Refresh Status'}
        </button>
      </div>

      <div className="card-body">
        <div className="status-grid">
          {/* Server Connection Indicator */}
          <div className="status-item">
            <span className="status-label">FastAPI Backend:</span>
            <div className="status-value-row">
              <span
                className={`status-indicator ${
                  loading
                    ? 'status-loading'
                    : isConnected
                    ? 'status-online'
                    : 'status-offline'
                }`}
              >
                ●
              </span>
              <span className="status-text">
                {loading
                  ? 'Connecting...'
                  : isConnected
                  ? 'Connected (HTTP 200)'
                  : 'Disconnected'}
              </span>
            </div>
          </div>

          {/* Model Status Indicator */}
          <div className="status-item">
            <span className="status-label">XGBoost Multiclass Model:</span>
            <div className="status-value-row">
              <span
                className={`status-indicator ${
                  loading
                    ? 'status-loading'
                    : isModelLoaded
                    ? 'status-online'
                    : 'status-offline'
                }`}
              >
                ●
              </span>
              <span className="status-text">
                {loading
                  ? 'Verifying...'
                  : isModelLoaded
                  ? 'Loaded in Memory (Ready)'
                  : 'Unavailable'}
              </span>
            </div>
          </div>
        </div>

        {/* Error Notice */}
        {error && (
          <div className="alert alert-error">
            <div className="alert-title">Connection Failed</div>
            <div className="alert-message">{error}</div>
            <div className="alert-hint">
              Make sure the FastAPI backend is running on <code>http://127.0.0.1:8000</code>:
              <br />
              <code>.\.venv\Scripts\python.exe -m uvicorn backend.services.main:app --host 127.0.0.1 --port 8000</code>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
