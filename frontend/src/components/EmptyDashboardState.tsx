import React from 'react';
import { BarChart3, FileSpreadsheet, ShieldCheck } from 'lucide-react';

export const EmptyDashboardState: React.FC = () => {
  return (
    <div className="empty-dashboard-card">
      <div className="empty-icon-wrapper">
        <BarChart3 size={48} className="empty-main-icon" />
      </div>
      <h3 className="empty-dashboard-title">No traffic data analyzed yet.</h3>
      <p className="empty-dashboard-desc">
        Upload a network traffic CSV file above, or click <strong>&quot;Load Sample Demo CSV&quot;</strong> to
        execute batch intrusion classification across 61 statistical features and generate full telemetry.
      </p>

      <div className="empty-features-grid">
        <div className="empty-feature-item">
          <FileSpreadsheet size={20} className="feature-item-icon" />
          <div>
            <div className="feature-item-title">61-Feature Validation</div>
            <div className="feature-item-desc">
              Strict schema verification ensures exact alignment with CIC-IDS2017 training features.
            </div>
          </div>
        </div>

        <div className="empty-feature-item">
          <BarChart3 size={20} className="feature-item-icon" />
          <div>
            <div className="feature-item-title">Real-Time Visualizations</div>
            <div className="feature-item-desc">
              Automatic generation of traffic distributions, threat donuts, and attack breakdowns.
            </div>
          </div>
        </div>

        <div className="empty-feature-item">
          <ShieldCheck size={20} className="feature-item-icon" />
          <div>
            <div className="feature-item-title">Vectorized Batch Inference</div>
            <div className="feature-item-desc">
              High-throughput predictions with winning class IDs, confidence scores, and probability vectors.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
