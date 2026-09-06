import React from 'react';
import { Activity, AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';
import type { PredictionResponse } from '../types/api';

interface TrafficSummaryCardsProps {
  predictions: PredictionResponse[];
  fileName?: string | null;
}

export const TrafficSummaryCards: React.FC<TrafficSummaryCardsProps> = ({
  predictions,
  fileName,
}) => {
  const totalFlows = predictions.length;
  const benignFlows = predictions.filter((p) => !p.is_attack).length;
  const attackFlows = predictions.filter((p) => p.is_attack).length;
  const attackPercentage = totalFlows > 0 ? (attackFlows / totalFlows) * 100 : 0;
  const benignPercentage = totalFlows > 0 ? (benignFlows / totalFlows) * 100 : 0;

  const avgConfidence =
    totalFlows > 0
      ? (predictions.reduce((acc, p) => acc + p.confidence, 0) / totalFlows) * 100
      : 0;

  return (
    <div className="traffic-summary-section">
      <div className="summary-section-header">
        <div>
          <h2 className="section-title">Traffic Analysis Overview</h2>
          <p className="section-subtitle">
            Calculated from {totalFlows} processed network flows {fileName ? `(${fileName})` : ''}
          </p>
        </div>
        <div className="summary-status-pill">
          {attackFlows > 0 ? (
            <span className="badge badge-attack">
              <AlertTriangle size={14} /> Threat Detected ({attackFlows} Flows)
            </span>
          ) : (
            <span className="badge badge-benign">
              <CheckCircle size={14} /> All Traffic Benign
            </span>
          )}
        </div>
      </div>

      <div className="metrics-grid">
        {/* 1. Total Flows */}
        <div className="metric-card metric-total">
          <div className="metric-card-header">
            <span className="metric-label">Total Flows Evaluated</span>
            <Activity size={18} className="metric-icon total-icon" />
          </div>
          <div className="metric-value">{totalFlows.toLocaleString()}</div>
          <div className="metric-subtext">Avg Confidence: {avgConfidence.toFixed(2)}%</div>
        </div>

        {/* 2. Benign Flows */}
        <div className="metric-card metric-benign">
          <div className="metric-card-header">
            <span className="metric-label">Benign Flows</span>
            <CheckCircle size={18} className="metric-icon benign-icon" />
          </div>
          <div className="metric-value text-emerald">{benignFlows.toLocaleString()}</div>
          <div className="metric-subtext">{benignPercentage.toFixed(2)}% of total traffic</div>
        </div>

        {/* 3. Attack Flows */}
        <div className="metric-card metric-attack">
          <div className="metric-card-header">
            <span className="metric-label">Malicious / Attack Flows</span>
            <ShieldAlert size={18} className="metric-icon attack-icon" />
          </div>
          <div className="metric-value text-rose">{attackFlows.toLocaleString()}</div>
          <div className="metric-subtext">
            {attackFlows > 0 ? `${attackFlows} anomalies flagged` : 'Zero malicious flows detected'}
          </div>
        </div>

        {/* 4. Attack Percentage */}
        <div className="metric-card metric-percentage">
          <div className="metric-card-header">
            <span className="metric-label">Attack Percentage</span>
            <AlertTriangle size={18} className="metric-icon percent-icon" />
          </div>
          <div className="metric-value text-amber">{attackPercentage.toFixed(2)}%</div>
          <div className="metric-subtext">
            Formula: (attack_flows / total_flows) × 100
          </div>
        </div>
      </div>
    </div>
  );
};
