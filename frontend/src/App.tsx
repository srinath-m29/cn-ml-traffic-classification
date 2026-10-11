import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Activity, ArrowDownToLine, ArrowRight, BarChart3, Check, ChevronDown,
  CircleAlert, Clock3, CloudUpload, FileSpreadsheet, LayoutDashboard,
  Network, RefreshCw, Search, Settings2, Shield, ShieldAlert, ShieldCheck,
  SlidersHorizontal, Sparkles, X,
} from 'lucide-react';
import './App.css';
import { apiService } from './services/api';
import { SAMPLE_FLOWS } from './data/sampleFlows';
import { parseAndValidateCsv } from './utils/csvValidator';
import type { HealthResponse, ModelInfoResponse, PredictionResponse } from './types/api';

type Page = 'Overview' | 'Analyze traffic' | 'Single flow' | 'Model details';

const NAV_ITEMS: { label: Page; icon: typeof LayoutDashboard }[] = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Analyze traffic', icon: BarChart3 },
  { label: 'Single flow', icon: SlidersHorizontal },
  { label: 'Model details', icon: Settings2 },
];

const formatNumber = (value: number) => new Intl.NumberFormat('en-US').format(value);
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function App() {
  const [page, setPage] = useState<Page>('Overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [statusError, setStatusError] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [fileName, setFileName] = useState('');
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [selectedSample, setSelectedSample] = useState(SAMPLE_FLOWS[0].id);
  const [features, setFeatures] = useState<Record<string, number>>(SAMPLE_FLOWS[0].features);
  const [singleResult, setSingleResult] = useState<PredictionResponse | null>(null);
  const [featureSearch, setFeatureSearch] = useState('');
  const [resultSearch, setResultSearch] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);

  const refreshStatus = useCallback(async () => {
    const [healthResult, infoResult] = await Promise.allSettled([
      apiService.getHealth(), apiService.getModelInfo(),
    ]);
    if (healthResult.status === 'fulfilled') setHealth(healthResult.value);
    else {
      setHealth(null);
      setStatusError(healthResult.reason instanceof Error ? healthResult.reason.message : 'API is unavailable');
    }
    if (infoResult.status === 'fulfilled') setModelInfo(infoResult.value);
    else setModelInfo(null);
    setStatusLoading(false);
  }, []);
  const handleRefreshStatus = () => {
    setStatusLoading(true);
    setStatusError('');
    void refreshStatus();
  };

  useEffect(() => {
    let active = true;
    Promise.allSettled([apiService.getHealth(), apiService.getModelInfo()]).then(([healthResult, infoResult]) => {
      if (!active) return;
      if (healthResult.status === 'fulfilled') setHealth(healthResult.value);
      else {
        setHealth(null);
        setStatusError(healthResult.reason instanceof Error ? healthResult.reason.message : 'API is unavailable');
      }
      if (infoResult.status === 'fulfilled') setModelInfo(infoResult.value);
      else setModelInfo(null);
      setStatusLoading(false);
    });
    return () => { active = false; };
  }, []);

  const modelReady = health?.model_loaded === true;
  const classCounts = useMemo(() => {
    const counts = new Map<string, number>();
    predictions.forEach((item) => counts.set(item.prediction, (counts.get(item.prediction) ?? 0) + 1));
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [predictions]);
  const filteredPredictions = useMemo(() => predictions.filter((item) =>
    item.prediction.toLowerCase().includes(resultSearch.toLowerCase()) ||
    String(item.class_id).includes(resultSearch),
  ), [predictions, resultSearch]);
  const filteredFeatures = useMemo(() => Object.entries(features).filter(([name]) =>
    name.toLowerCase().includes(featureSearch.toLowerCase()),
  ), [features, featureSearch]);

  const analyzeFile = async (file?: File) => {
    if (!file) return;
    setError('');
    setPredictions([]);
    setFileName(file.name);
    if (!modelReady || !modelInfo) {
      setError('The API and model must be ready before you can analyze a file.');
      return;
    }
    setBusy(true);
    try {
      const parsed = await parseAndValidateCsv(file, modelInfo.feature_names);
      if (!parsed.isValid) throw new Error(parsed.errors.slice(0, 4).join(' · '));
      const result = await apiService.predictBatch({ samples: parsed.parsedSamples });
      setPredictions(result.predictions);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not analyze this CSV file.');
    } finally {
      setBusy(false);
    }
  };

  const chooseSample = (id: string) => {
    const sample = SAMPLE_FLOWS.find((item) => item.id === id);
    if (!sample) return;
    setSelectedSample(id);
    setFeatures(sample.features);
    setSingleResult(null);
    setError('');
  };

  const runSinglePrediction = async () => {
    setBusy(true);
    setError('');
    try {
      setSingleResult(await apiService.predict({ features }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Prediction failed.');
    } finally {
      setBusy(false);
    }
  };

  const exportResults = () => {
    const header = ['prediction', 'class_id', 'confidence', 'is_attack'];
    const rows = predictions.map((item) => [item.prediction, item.class_id, item.confidence, item.is_attack]);
    const csv = [header, ...rows].map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${fileName.replace(/\.csv$/i, '') || 'traffic-results'}-predictions.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" onClick={() => setPage('Overview')}>
          <span className="brand-mark"><Activity size={20} /></span>
          <span><strong>Sentinel</strong><small>NETWORK INTELLIGENCE</small></span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {NAV_ITEMS.map(({ label, icon: Icon }) => (
            <button key={label} type="button" aria-label={label} className={`nav-item ${page === label ? 'selected' : ''}`} onClick={() => { setPage(label); setError(''); }}>
              <Icon size={17} /><span>{label}</span>{page === label && <span className="nav-indicator" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <div className="note-icon"><Shield size={16} /></div>
          <strong>Local inference</strong>
          <p>Traffic data stays in your browser and is sent only to your configured API.</p>
        </div>
        <div className="profile"><div className="avatar">N</div><div><strong>NIDS Project</strong><small>Local environment</small></div><ChevronDown size={15} /></div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-sep">/</span><strong>{page}</strong></div>
          <div className="topbar-actions">
            <div className={`connection ${modelReady ? 'online' : statusLoading ? 'checking' : 'offline'}`}>
              <span className="connection-dot" />{statusLoading ? 'Connecting' : modelReady ? 'Model online' : 'Model offline'}
            </div>
            <button className="icon-button" type="button" onClick={handleRefreshStatus} aria-label="Refresh API status" title="Refresh status"><RefreshCw size={16} /></button>
          </div>
        </header>

        <div className="page-content">
          {statusError && <div className="notice error"><CircleAlert size={17} /><span>API connection failed: {statusError}</span></div>}
          {error && <div className="notice error"><CircleAlert size={17} /><span>{error}</span><button type="button" className="notice-close" onClick={() => setError('')} aria-label="Dismiss"><X size={16} /></button></div>}

          {page === 'Overview' && (
            <>
              <section className="welcome-row"><div><div className="eyebrow"><Sparkles size={14} /> NETWORK THREAT ANALYSIS</div><h1>See your network<br /><span>with clarity.</span></h1><p>Classify traffic flows and surface suspicious activity with your CIC-IDS2017 model.</p><div className="welcome-actions"><button className="button primary" type="button" onClick={() => setPage('Analyze traffic')}><CloudUpload size={17} /> Analyze a CSV <ArrowRight size={16} /></button><button className="button quiet" type="button" onClick={() => setPage('Single flow')}>Try a sample flow</button></div></div><div className="hero-visual"><div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" /><div className="hero-core"><Network size={38} /></div><span className="orbit-node node-a" /><span className="orbit-node node-b" /><span className="orbit-node node-c" /><div className="hero-caption"><span className="live-dot" /> FLOW MONITOR READY</div></div></section>

              <section className="metric-grid">
                <MetricCard icon={<ShieldCheck size={17} />} label="MODEL STATUS" value={statusLoading ? 'Checking' : modelReady ? 'Ready' : 'Unavailable'} detail={modelInfo?.model_type ?? 'Waiting for API'} tone={modelReady ? 'green' : 'amber'} />
                <MetricCard icon={<Network size={17} />} label="FEATURE INPUTS" value={modelInfo ? formatNumber(modelInfo.num_features) : '—'} detail="Required per network flow" tone="blue" />
                <MetricCard icon={<BarChart3 size={17} />} label="TRAFFIC CLASSES" value={modelInfo ? formatNumber(modelInfo.num_classes) : '—'} detail="Output categories" tone="violet" />
                <MetricCard icon={<Activity size={17} />} label="FLOWS ANALYZED" value={formatNumber(predictions.length)} detail={fileName || 'No analysis yet'} tone="cyan" />
              </section>

              <section className="overview-grid">
                <div className="panel quick-panel"><div className="panel-heading"><div><div className="eyebrow">GET STARTED</div><h2>Choose an analysis</h2></div><span className="soft-tag">2 workflows</span></div><button className="workflow-row" type="button" onClick={() => setPage('Analyze traffic')}><span className="workflow-icon purple"><FileSpreadsheet size={19} /></span><span className="workflow-copy"><strong>Batch traffic analysis</strong><small>Upload a feature CSV and classify up to 10,000 flows.</small></span><ArrowRight size={17} /></button><button className="workflow-row" type="button" onClick={() => setPage('Single flow')}><span className="workflow-icon teal"><SlidersHorizontal size={19} /></span><span className="workflow-copy"><strong>Inspect a single flow</strong><small>Explore a sample or edit its feature values.</small></span><ArrowRight size={17} /></button><div className="workflow-foot"><span className="tiny-dot" /> Files are validated before any prediction request.</div></div>
                <div className="panel readiness-panel"><div className="panel-heading"><div><div className="eyebrow">SYSTEM READINESS</div><h2>Inference service</h2></div><button className="text-button" type="button" onClick={handleRefreshStatus}>Refresh</button></div><div className="readiness-state"><span className={`readiness-icon ${modelReady ? 'ready' : 'waiting'}`}>{modelReady ? <Check size={18} /> : <Clock3 size={18} />}</span><div><strong>{modelReady ? 'Ready to classify' : statusLoading ? 'Checking connection' : 'Waiting for the API'}</strong><small>{modelReady ? 'The model and API are responding.' : 'Start the backend to enable predictions.'}</small></div></div><div className="readiness-divider" /><div className="readiness-item"><span>API base URL</span><code>{import.meta.env.VITE_API_BASE_URL?.trim() || 'http://127.0.0.1:8000'}</code></div><div className="readiness-item"><span>Model input</span><strong>{modelInfo ? `${modelInfo.num_features} numeric features` : 'Not detected'}</strong></div></div>
              </section>
            </>
          )}

          {page === 'Analyze traffic' && (
            <>
              <PageHeading eyebrow="BATCH CLASSIFICATION" title="Analyze traffic" subtitle="Upload a flow feature CSV to classify network activity in one batch." />
              <section className="panel upload-panel"><div className="upload-symbol"><CloudUpload size={24} /></div><h2>Drop your CSV here</h2><p>Choose a file with the model’s {modelInfo?.num_features ?? 61} feature columns. Maximum 25 MB and 10,000 flows.</p><input ref={fileInput} className="visually-hidden" type="file" accept=".csv,text/csv" onChange={(event) => { void analyzeFile(event.target.files?.[0]); event.currentTarget.value = ''; }} /><button className="button primary" type="button" onClick={() => fileInput.current?.click()} disabled={busy || !modelReady}>{busy ? <><span className="spinner" /> Analyzing…</> : <><FileSpreadsheet size={17} /> Choose CSV file</>}</button><div className="upload-foot"><span>CSV only</span><span>·</span><span>Validated in browser</span><span>·</span><span>Max 10,000 rows</span></div></section>
              {predictions.length > 0 && <section className="results-area"><div className="results-title"><div><div className="eyebrow">ANALYSIS COMPLETE</div><h2>{fileName}</h2><p>{formatNumber(predictions.length)} flows classified</p></div><button className="button secondary" type="button" onClick={exportResults}><ArrowDownToLine size={16} /> Export results</button></div><div className="analysis-summary"><div className="panel distribution-panel"><div className="panel-heading"><div><div className="eyebrow">PREDICTION MIX</div><h2>Class distribution</h2></div></div>{classCounts.slice(0, 7).map(([name, count]) => <div className="distribution-row" key={name}><div><span>{name}</span><strong>{formatNumber(count)}</strong></div><div className="bar-track"><span style={{ width: `${Math.max(2, count / predictions.length * 100)}%` }} /></div></div>)}</div><div className="panel result-table-panel"><div className="panel-heading"><div><div className="eyebrow">FLOW RESULTS</div><h2>Predictions</h2></div><label className="search-box"><Search size={15} /><input value={resultSearch} onChange={(event) => setResultSearch(event.target.value)} placeholder="Search class…" /></label></div><div className="table-wrap"><table><thead><tr><th>FLOW</th><th>CLASSIFICATION</th><th>CONFIDENCE</th><th>TYPE</th></tr></thead><tbody>{filteredPredictions.slice(0, 100).map((item, index) => <tr key={`${item.class_id}-${index}`}><td className="muted-cell">#{String(index + 1).padStart(4, '0')}</td><td><span className="class-cell"><span className={`class-dot ${item.is_attack ? 'attack' : 'benign'}`} />{item.prediction}</span></td><td>{percent(item.confidence)}</td><td><span className={`badge ${item.is_attack ? 'attack-badge' : 'benign-badge'}`}>{item.is_attack ? 'Attack' : 'Benign'}</span></td></tr>)}</tbody></table>{filteredPredictions.length > 100 && <div className="table-more">Showing first 100 of {formatNumber(filteredPredictions.length)} flows. Export to save the complete results.</div>}</div></div></div></section>}
            </>
          )}

          {page === 'Single flow' && (
            <>
              <PageHeading eyebrow="SINGLE FLOW CLASSIFICATION" title="Inspect a flow" subtitle="Start from a sample, review its feature values, then request a prediction." />
              <section className="single-layout"><div className="panel feature-panel"><div className="panel-heading"><div><div className="eyebrow">FLOW FEATURES</div><h2>Input values <span className="count-pill">{Object.keys(features).length}</span></h2></div><label className="sample-select"><span>Sample flow</span><select value={selectedSample} onChange={(event) => chooseSample(event.target.value)}>{SAMPLE_FLOWS.map((sample) => <option key={sample.id} value={sample.id}>{sample.name}</option>)}</select></label></div><div className="feature-toolbar"><span>Numeric features required by the model</span><label className="search-box"><Search size={15} /><input value={featureSearch} onChange={(event) => setFeatureSearch(event.target.value)} placeholder="Find a feature…" /></label></div><div className="feature-grid">{filteredFeatures.map(([name, value]) => <label className="feature-field" key={name}><span title={name}>{name}</span><input type="number" step="any" value={value} onChange={(event) => { const next = Number(event.target.value); setFeatures((current) => ({ ...current, [name]: Number.isFinite(next) ? next : 0 })); setSingleResult(null); }} /></label>)}</div><div className="feature-actions"><span>Values are sent to the configured local API for inference.</span><button className="button primary" type="button" disabled={!modelReady || busy} onClick={() => void runSinglePrediction()}>{busy ? <><span className="spinner" /> Predicting…</> : <><Sparkles size={16} /> Classify flow</>}</button></div></div><div className="panel single-result-panel"><div className="eyebrow">MODEL OUTPUT</div><h2>Prediction</h2>{singleResult ? <><div className={`single-verdict ${singleResult.is_attack ? 'verdict-attack' : 'verdict-benign'}`}>{singleResult.is_attack ? <ShieldAlert size={24} /> : <ShieldCheck size={24} />}<div><small>{singleResult.is_attack ? 'POTENTIAL THREAT' : 'NORMAL TRAFFIC'}</small><strong>{singleResult.prediction}</strong></div></div><div className="confidence-block"><div><span>Confidence</span><strong>{percent(singleResult.confidence)}</strong></div><div className="bar-track"><span style={{ width: `${singleResult.confidence * 100}%` }} /></div></div><div className="probability-list">{Object.entries(singleResult.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([label, value]) => <div key={label}><span>{label}</span><strong>{percent(value)}</strong></div>)}</div></> : <div className="result-empty"><span><Sparkles size={22} /></span><strong>Ready when you are</strong><p>Run a classification to see the predicted traffic class and confidence.</p></div>}</div></section>
            </>
          )}

          {page === 'Model details' && (
            <>
              <PageHeading eyebrow="MODEL REFERENCE" title="Model details" subtitle="Feature and class metadata provided by the running inference API." />
              <section className="details-grid"><div className="panel details-card"><div className="eyebrow">MODEL</div><h2>{modelInfo?.model_type ?? 'Model metadata unavailable'}</h2><p>The model is loaded by the backend for CPU inference. Training and model updates are handled separately.</p><div className="detail-pair"><span>Status</span><strong className={modelReady ? 'text-green' : 'text-amber'}>{modelReady ? 'Ready' : 'Unavailable'}</strong></div><div className="detail-pair"><span>Features</span><strong>{modelInfo?.num_features ?? '—'}</strong></div><div className="detail-pair"><span>Classes</span><strong>{modelInfo?.num_classes ?? '—'}</strong></div><div className="detail-pair"><span>Batch limit</span><strong>10,000 flows</strong></div></div><div className="panel metadata-card"><div className="panel-heading"><div><div className="eyebrow">OUTPUT LABELS</div><h2>Traffic classes</h2></div><span className="soft-tag">{modelInfo?.classes.length ?? 0} labels</span></div><div className="label-list">{modelInfo?.classes.map((label, index) => <div key={label}><span className="label-index">{String(index).padStart(2, '0')}</span><span>{label}</span><span className="class-dot" /></div>)}</div></div><div className="panel metadata-card feature-meta-card"><div className="panel-heading"><div><div className="eyebrow">INPUT SCHEMA</div><h2>Required features</h2></div><span className="soft-tag">{modelInfo?.feature_names.length ?? 0} columns</span></div><div className="schema-list">{modelInfo?.feature_names.map((name, index) => <div key={name}><span>{String(index + 1).padStart(2, '0')}</span><code>{name}</code></div>)}</div></div></section>
            </>
          )}

          <footer className="page-footer"><span>Sentinel · Network Traffic Classification</span><span><span className={`tiny-dot ${modelReady ? '' : 'muted-dot'}`} /> {modelReady ? 'Inference service ready' : 'Waiting for backend'}</span></footer>
        </div>
      </main>
    </div>
  );
}

function MetricCard({ icon, label, value, detail, tone }: { icon: ReactNode; label: string; value: string; detail: string; tone: string }) {
  return <div className="metric-card"><div className={`metric-icon ${tone}`}>{icon}</div><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-detail" title={detail}>{detail}</div></div>;
}

function PageHeading({ eyebrow, title, subtitle }: { eyebrow: string; title: string; subtitle: string }) {
  return <div className="page-heading"><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>;
}

export default App;
