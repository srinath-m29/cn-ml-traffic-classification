import { useCallback, useEffect, useState } from 'react';
import {
  AlertCircle,
  BarChart3,
  Sliders,
} from 'lucide-react';
import './App.css';
import { BackendStatusCard } from './components/BackendStatusCard';
import { ClassDistributionTable } from './components/ClassDistributionTable';
import { CsvUploadCard } from './components/CsvUploadCard';
import { EmptyDashboardState } from './components/EmptyDashboardState';
import { Header } from './components/Header';
import { ModelInfoCard } from './components/ModelInfoCard';
import { PredictionTestCard } from './components/PredictionTestCard';
import { ResultsTable } from './components/ResultsTable';
import { TrafficCharts } from './components/TrafficCharts';
import { TrafficSummaryCards } from './components/TrafficSummaryCards';
import { apiService } from './services/api';
import type {
  BatchPredictionResponse,
  HealthResponse,
  ModelInfoResponse,
} from './types/api';

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [modelInfoLoading, setModelInfoLoading] = useState<boolean>(true);
  const [modelInfoError, setModelInfoError] = useState<string | null>(null);

  // Tab State: 'dashboard' (Batch CSV) | 'single' (Phase 3A Single-Flow Test)
  const [activeTab, setActiveTab] = useState<'dashboard' | 'single'>('dashboard');

  // Batch Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<BatchPredictionResponse | null>(null);
  const [analyzedFileName, setAnalyzedFileName] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const fetchStatusAndModelInfo = useCallback(async () => {
    // 1. Fetch Health
    setHealthLoading(true);
    setHealthError(null);
    try {
      const healthData = await apiService.getHealth();
      setHealth(healthData);
      setHealthError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unable to connect to backend service';
      setHealth(null);
      setHealthError(msg);
    } finally {
      setHealthLoading(false);
    }

    // 2. Fetch Model Info
    setModelInfoLoading(true);
    setModelInfoError(null);
    try {
      const infoData = await apiService.getModelInfo();
      setModelInfo(infoData);
      setModelInfoError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unable to load model metadata';
      setModelInfo(null);
      setModelInfoError(msg);
    } finally {
      setModelInfoLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatusAndModelInfo();
  }, [fetchStatusAndModelInfo]);

  const backendOnline = !healthLoading && !healthError && health?.status === 'healthy';

  const handleStartBatchAnalysis = async (
    samples: Record<string, number>[],
    fileName: string
  ) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalyzedFileName(fileName);

    try {
      const result = await apiService.predictBatch({ samples });
      setAnalysisResult(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Batch prediction request failed';
      setAnalysisError(msg);
      setAnalysisResult(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClearAnalysis = () => {
    setAnalysisResult(null);
    setAnalyzedFileName(null);
    setAnalysisError(null);
  };

  return (
    <div className="app-layout">
      <Header />

      <main className="app-main">
        {/* Navigation Tabs Bar */}
        <div className="dashboard-nav-tabs">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <BarChart3 size={18} />
            <span>Traffic Analysis Dashboard (Batch CSV)</span>
            {analysisResult && (
              <span className="tab-pill">{analysisResult.total_samples} Flows</span>
            )}
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => setActiveTab('single')}
          >
            <Sliders size={18} />
            <span>Single Flow Predictor (Phase 3A)</span>
          </button>
        </div>

        {/* Global System Health & Metadata Strip */}
        <div className="system-status-grid">
          <BackendStatusCard
            health={health}
            loading={healthLoading}
            error={healthError}
            onRetry={fetchStatusAndModelInfo}
          />
          <ModelInfoCard
            modelInfo={modelInfo}
            loading={modelInfoLoading}
            error={modelInfoError}
          />
        </div>

        {/* Tab 1: Traffic Analysis Dashboard */}
        {activeTab === 'dashboard' && (
          <div className="traffic-dashboard-flow">
            {/* Step 1: CSV Ingestion Card */}
            <CsvUploadCard
              expectedFeatures={modelInfo?.feature_names || []}
              backendOnline={backendOnline}
              isAnalyzing={isAnalyzing}
              onStartAnalysis={handleStartBatchAnalysis}
              onClear={handleClearAnalysis}
            />

            {/* Error Notification Banner */}
            {analysisError && (
              <div className="dashboard-alert-banner alert-error">
                <AlertCircle size={20} className="alert-icon" />
                <div>
                  <strong>Batch Prediction Failed:</strong> {analysisError}
                </div>
              </div>
            )}

            {/* Step 2: Empty State vs Analysis Telemetry */}
            {analysisResult ? (
              <div className="analysis-results-container">
                {/* 1. Summary Metrics */}
                <TrafficSummaryCards
                  predictions={analysisResult.predictions}
                  fileName={analyzedFileName}
                />

                {/* 2. Visualizations (Recharts) */}
                <TrafficCharts
                  predictions={analysisResult.predictions}
                  classes={modelInfo?.classes || []}
                />

                {/* 3. Class Distribution & Attack Breakdown Tables */}
                <ClassDistributionTable
                  predictions={analysisResult.predictions}
                  classes={modelInfo?.classes || []}
                />

                {/* 4. Paginated Results Table */}
                <ResultsTable
                  predictions={analysisResult.predictions}
                  classes={modelInfo?.classes || []}
                />
              </div>
            ) : (
              !isAnalyzing && <EmptyDashboardState />
            )}
          </div>
        )}

        {/* Tab 2: Single Flow Predictor (Phase 3A) */}
        {activeTab === 'single' && (
          <div className="single-predictor-container">
            <PredictionTestCard backendOnline={backendOnline} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-container">
          <span>College Project: ML-Based Network Traffic Classification &amp; Intrusion Detection</span>
          <span>FastAPI + XGBoost + React Dashboard</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
