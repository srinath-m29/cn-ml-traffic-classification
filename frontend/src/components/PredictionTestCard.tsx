import React, { useState } from 'react';
import { SAMPLE_FLOWS, type SampleFlowOption } from '../data/sampleFlows';
import { apiService } from '../services/api';
import type { PredictionResponse } from '../types/api';

interface PredictionTestCardProps {
  backendOnline: boolean;
}

export const PredictionTestCard: React.FC<PredictionTestCardProps> = ({
  backendOnline,
}) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>(SAMPLE_FLOWS[0].id);
  const [loadedFeatures, setLoadedFeatures] = useState<Record<string, number> | null>(
    SAMPLE_FLOWS[0].features
  );
  const [loadedSampleMeta, setLoadedSampleMeta] = useState<SampleFlowOption>(SAMPLE_FLOWS[0]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showJson, setShowJson] = useState<boolean>(false);
  const [showFeaturePreview, setShowFeaturePreview] = useState<boolean>(false);

  const handleSelectSample = (sampleId: string) => {
    const found = SAMPLE_FLOWS.find((s) => s.id === sampleId);
    if (found) {
      setSelectedSampleId(sampleId);
      setLoadedFeatures(found.features);
      setLoadedSampleMeta(found);
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!loadedFeatures) {
      setError('Please load a sample flow first.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await apiService.predict({ features: loadedFeatures });
      setPrediction(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown prediction error occurred.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  // Sort probabilities from highest to lowest
  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">3. Prediction Test (API Verification)</h2>
        <span className="badge badge-secondary">POST /predict</span>
      </div>

      <div className="card-body">
        <p className="card-description">
          Test live model inference by loading authentic network flows with all 61 pre-calculated
          statistical flow features extracted from the CIC-IDS2017 benchmark.
        </p>

        {/* Sample Selection Toolbar */}
        <div className="sample-selection-box">
          <div className="sample-control-group">
            <label htmlFor="sample-selector" className="control-label">
              Select Sample Flow:
            </label>
            <select
              id="sample-selector"
              className="select-input"
              value={selectedSampleId}
              onChange={(e) => handleSelectSample(e.target.value)}
              disabled={loading}
            >
              {SAMPLE_FLOWS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          <div className="sample-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => handleSelectSample(selectedSampleId)}
              disabled={loading}
            >
              Reset Sample
            </button>
            <button
              type="button"
              className="btn btn-primary btn-lg"
              onClick={handlePredict}
              disabled={loading || !backendOnline || !loadedFeatures}
            >
              {loading ? (
                <>
                  <span className="spinner spinner-sm"></span>
                  <span>Predicting...</span>
                </>
              ) : (
                'Run Predict'
              )}
            </button>
          </div>
        </div>

        {/* Sample Description */}
        {loadedSampleMeta && (
          <div className="sample-info-banner">
            <strong>Loaded Flow:</strong> {loadedSampleMeta.name} —{' '}
            <span className="text-muted">{loadedSampleMeta.description}</span>
            <div className="sample-meta-tags">
              <span className="mini-tag">
                Features Loaded: {loadedFeatures ? Object.keys(loadedFeatures).length : 0} / 61
              </span>
              <span className="mini-tag">Target: {loadedSampleMeta.label}</span>
              <button
                type="button"
                className="btn-link"
                onClick={() => setShowFeaturePreview(!showFeaturePreview)}
              >
                {showFeaturePreview ? 'Hide 61 Values' : 'Inspect 61 Values'}
              </button>
            </div>
          </div>
        )}

        {/* Feature Preview Table */}
        {showFeaturePreview && loadedFeatures && (
          <div className="feature-preview-drawer">
            <table className="features-table font-sm">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Feature Name</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(loadedFeatures).map(([k, v], idx) => (
                  <tr key={k}>
                    <td className="text-muted">{idx + 1}</td>
                    <td><code>{k}</code></td>
                    <td className="text-numeric">{v.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Error Notice */}
        {error && (
          <div className="alert alert-error mt-md">
            <div className="alert-title">Prediction Failed</div>
            <div className="alert-message">{error}</div>
          </div>
        )}

        {/* Prediction Results Display */}
        {prediction && (
          <div className="result-container mt-lg">
            <div className="result-header">
              <h3 className="result-heading">Inference Result</h3>
              <span
                className={`badge-status ${
                  prediction.is_attack ? 'badge-attack' : 'badge-benign'
                }`}
              >
                {prediction.is_attack ? 'ATTACK DETECTED' : 'NORMAL TRAFFIC (BENIGN)'}
              </span>
            </div>

            <div className="result-summary-grid">
              <div className="result-metric-card">
                <div className="result-metric-label">Predicted Traffic Class</div>
                <div className="result-metric-value">{prediction.prediction}</div>
              </div>

              <div className="result-metric-card">
                <div className="result-metric-label">Model Class ID</div>
                <div className="result-metric-value">#{prediction.class_id}</div>
              </div>

              <div className="result-metric-card">
                <div className="result-metric-label">Winning Confidence</div>
                <div className="result-metric-value">
                  {(prediction.confidence * 100).toFixed(2)}%
                </div>
              </div>

              <div className="result-metric-card">
                <div className="result-metric-label">Security Classification</div>
                <div className="result-metric-value">
                  {prediction.is_attack ? 'MALICIOUS' : 'CLEAN'}
                </div>
              </div>
            </div>

            {/* Top Class Probabilities breakdown */}
            <div className="probabilities-section mt-md">
              <h4 className="probabilities-title">Class Probability Distribution:</h4>
              <div className="probabilities-list">
                {sortedProbabilities.map(([className, prob]) => {
                  const percentage = prob * 100;
                  const isWinner = className === prediction.prediction;
                  return (
                    <div
                      key={className}
                      className={`prob-row ${isWinner ? 'prob-winner' : ''}`}
                    >
                      <div className="prob-info">
                        <span className="prob-name">
                          {isWinner && <span className="winner-mark">★ </span>}
                          {className}
                        </span>
                        <span className="prob-pct">
                          {percentage > 0.01
                            ? `${percentage.toFixed(2)}%`
                            : prob > 0
                            ? '<0.01%'
                            : '0.00%'}
                        </span>
                      </div>
                      <div className="prob-bar-track">
                        <div
                          className={`prob-bar-fill ${
                            isWinner
                              ? prediction.is_attack
                                ? 'fill-attack'
                                : 'fill-benign'
                              : 'fill-neutral'
                          }`}
                          style={{ width: `${Math.max(percentage, prob > 0 ? 1 : 0)}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Toggle Raw JSON */}
            <div className="json-toggle-row mt-md">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setShowJson(!showJson)}
              >
                {showJson ? 'Hide Raw API Response' : 'View Raw API Response (JSON)'}
              </button>
            </div>

            {showJson && (
              <pre className="json-display mt-sm">
                {JSON.stringify(prediction, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
