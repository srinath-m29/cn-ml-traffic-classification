import React, { useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  FileSpreadsheet,
  Loader2,
  Play,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { generateDemoCsvContent } from '../data/demoCsvData';
import { parseAndValidateCsv, type CsvValidationResult } from '../utils/csvValidator';

interface CsvUploadCardProps {
  expectedFeatures: string[];
  backendOnline: boolean;
  isAnalyzing: boolean;
  onStartAnalysis: (samples: Record<string, number>[], fileName: string) => Promise<void>;
  onClear: () => void;
}

export const CsvUploadCard: React.FC<CsvUploadCardProps> = ({
  expectedFeatures,
  backendOnline,
  isAnalyzing,
  onStartAnalysis,
  onClear,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [selectedFileSize, setSelectedFileSize] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<CsvValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const processFile = async (file: File) => {
    setSelectedFileName(file.name);
    setSelectedFileSize(formatFileSize(file.size));
    setIsValidating(true);
    setValidationResult(null);

    try {
      const result = await parseAndValidateCsv(file, expectedFeatures);
      setValidationResult(result);
    } catch (err: unknown) {
      setValidationResult({
        isValid: false,
        errors: [err instanceof Error ? err.message : 'Unknown validation error occurred'],
        warnings: [],
        rowCount: 0,
        parsedSamples: [],
        headers: [],
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleLoadDemo = async () => {
    setSelectedFileName('demo_cic_ids2017_sample_20flows.csv');
    setSelectedFileSize('12.4 KB');
    setIsValidating(true);
    setValidationResult(null);

    try {
      const demoCsvText = generateDemoCsvContent();
      const result = await parseAndValidateCsv(demoCsvText, expectedFeatures);
      setValidationResult(result);
    } catch (err: unknown) {
      setValidationResult({
        isValid: false,
        errors: [err instanceof Error ? err.message : 'Failed to load demo CSV'],
        warnings: [],
        rowCount: 0,
        parsedSamples: [],
        headers: [],
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleClear = () => {
    setSelectedFileName(null);
    setSelectedFileSize(null);
    setValidationResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onClear();
  };

  const handleAnalyzeClick = async () => {
    if (!validationResult || !validationResult.isValid || validationResult.parsedSamples.length === 0) {
      return;
    }
    await onStartAnalysis(
      validationResult.parsedSamples,
      selectedFileName || 'traffic_sample.csv'
    );
  };

  return (
    <section className="dashboard-card csv-upload-card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Network Traffic CSV Ingestion</h2>
          <p className="card-subtitle">
            Upload network flow records for batch intrusion classification (Max 25 MB)
          </p>
        </div>
        <div className="card-actions">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={handleLoadDemo}
            disabled={isAnalyzing || isValidating}
          >
            <FileSpreadsheet size={15} />
            Load Sample Demo CSV (20 Flows)
          </button>
        </div>
      </div>

      <div className="card-body">
        <input
          type="file"
          ref={fileInputRef}
          accept=".csv,text/csv"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        {/* Dropzone Area */}
        {!selectedFileName ? (
          <div
            className={`upload-dropzone ${isDragging ? 'is-dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                fileInputRef.current?.click();
              }
            }}
          >
            <UploadCloud size={40} className="dropzone-icon" />
            <div className="dropzone-text">
              <span className="dropzone-primary">Click to upload CSV</span> or drag and drop
            </div>
            <div className="dropzone-hint">
              Requires all 61 CIC-IDS2017 flow features. Supported format: standard comma-separated .csv
            </div>
          </div>
        ) : (
          /* File Selected & Validation Summary Box */
          <div className="selected-file-container">
            <div className="selected-file-header">
              <div className="file-info-group">
                <FileSpreadsheet size={24} className="file-icon" />
                <div>
                  <div className="file-name">{selectedFileName}</div>
                  <div className="file-meta">
                    <span>Size: {selectedFileSize}</span>
                    {validationResult && (
                      <>
                        <span className="dot-sep">•</span>
                        <span>Detected Rows: {validationResult.rowCount}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <button
                type="button"
                className="btn btn-icon-danger"
                onClick={handleClear}
                disabled={isAnalyzing || isValidating}
                title="Remove file"
              >
                <Trash2 size={16} />
              </button>
            </div>

            {/* Validation State Display */}
            {isValidating && (
              <div className="validation-box validation-loading">
                <Loader2 size={16} className="spinner" />
                <span>Validating 61 model features and data integrity...</span>
              </div>
            )}

            {!isValidating && validationResult && validationResult.isValid && (
              <div className="validation-box validation-success">
                <CheckCircle2 size={18} className="success-icon" />
                <div>
                  <strong>Validation Succeeded:</strong> Found all {expectedFeatures.length} required features across{' '}
                  {validationResult.rowCount} flow rows. Ready for batch analysis.
                </div>
              </div>
            )}

            {!isValidating && validationResult && !validationResult.isValid && (
              <div className="validation-box validation-error">
                <div className="error-header">
                  <AlertCircle size={18} className="error-icon" />
                  <strong>Validation Failed ({validationResult.errors.length} issue{validationResult.errors.length > 1 ? 's' : ''}):</strong>
                </div>
                <ul className="error-list">
                  {validationResult.errors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Action Controls */}
        <div className="upload-actions-bar">
          <div className="features-badge-info">
            Required Schema: <strong>{expectedFeatures.length || 61} features</strong> (validated against GET /model-info)
          </div>

          <button
            type="button"
            className="btn btn-primary"
            onClick={handleAnalyzeClick}
            disabled={
              !backendOnline ||
              isAnalyzing ||
              isValidating ||
              !validationResult ||
              !validationResult.isValid ||
              validationResult.parsedSamples.length === 0
            }
          >
            {isAnalyzing ? (
              <>
                <Loader2 size={16} className="spinner" />
                Analyzing network traffic...
              </>
            ) : (
              <>
                <Play size={16} />
                Analyze Traffic ({validationResult?.parsedSamples.length ?? 0} Flows)
              </>
            )}
          </button>
        </div>
      </div>
    </section>
  );
};
