import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="app-header">
      <div className="header-container">
        <div className="header-title-group">
          <div className="header-badge">CN-ML INTRUSION DETECTION SYSTEM</div>
          <h1 className="header-title">Network Traffic Classification</h1>
          <p className="header-subtitle">
            Machine Learning Based Network Traffic Analysis &amp; Flow Categorization
          </p>
        </div>
        <div className="header-meta">
          <span className="dataset-tag">Dataset: CIC-IDS2017</span>
          <span className="model-tag">Model: XGBoost (15-Class)</span>
        </div>
      </div>
    </header>
  );
};
