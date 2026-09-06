import React, { useState } from 'react';
import type { ModelInfoResponse } from '../types/api';

interface ModelInfoCardProps {
  modelInfo: ModelInfoResponse | null;
  loading: boolean;
  error: string | null;
}

export const ModelInfoCard: React.FC<ModelInfoCardProps> = ({
  modelInfo,
  loading,
  error,
}) => {
  const [showFeatures, setShowFeatures] = useState(false);

  if (loading) {
    return (
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">2. Model Information</h2>
        </div>
        <div className="card-body">
          <div className="loading-spinner-row">
            <span className="spinner"></span>
            <span>Loading model metadata from backend...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">2. Model Information</h2>
        </div>
        <div className="card-body">
          <p className="text-muted">
            Model metadata unavailable. Connect to the backend to inspect model specifications.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">2. Model Information</h2>
        <span className="badge badge-primary">{modelInfo.model_type}</span>
      </div>

      <div className="card-body">
        {/* Metric stats row */}
        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-label">Features Required</div>
            <div className="metric-value">{modelInfo.num_features}</div>
            <div className="metric-subtext">Statistical Flow Metrics</div>
          </div>

          <div className="metric-box">
            <div className="metric-label">Target Classes</div>
            <div className="metric-value">{modelInfo.num_classes}</div>
            <div className="metric-subtext">1 Benign + 14 Attack Types</div>
          </div>

          <div className="metric-box">
            <div className="metric-label">Framework &amp; Engine</div>
            <div className="metric-value font-sm">XGBoost (hist)</div>
            <div className="metric-subtext">CUDA GPU Accelerated</div>
          </div>
        </div>

        {/* Supported Class Labels */}
        <div className="class-labels-section">
          <div className="section-subtitle">Supported Traffic Categories (15 Classes):</div>
          <div className="labels-flex">
            {modelInfo.classes.map((cls, idx) => {
              const isBenign = cls === 'BENIGN';
              return (
                <span
                  key={cls}
                  className={`class-tag ${isBenign ? 'class-benign' : 'class-attack'}`}
                  title={`Class ID: ${idx}`}
                >
                  <span className="class-id-num">{idx}</span>
                  {cls}
                </span>
              );
            })}
          </div>
        </div>

        {/* Expandable Feature List */}
        <div className="features-toggle-section">
          <button
            type="button"
            className="btn btn-text"
            onClick={() => setShowFeatures(!showFeatures)}
          >
            {showFeatures ? '▲ Hide Expected Feature Names (61)' : '▼ View All Expected Feature Names (61)'}
          </button>

          {showFeatures && (
            <div className="features-table-wrapper">
              <table className="features-table">
                <thead>
                  <tr>
                    <th style={{ width: '60px' }}>Index</th>
                    <th>Feature Name (Exact Booster Order)</th>
                  </tr>
                </thead>
                <tbody>
                  {modelInfo.feature_names.map((name, index) => (
                    <tr key={name}>
                      <td className="text-muted">{index + 1}</td>
                      <td><code>{name}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
