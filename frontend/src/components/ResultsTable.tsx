import React, { useMemo, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Filter,
  Search,
} from 'lucide-react';
import type { PredictionResponse } from '../types/api';

interface ResultsTableProps {
  predictions: PredictionResponse[];
  classes: string[];
}

export const ResultsTable: React.FC<ResultsTableProps> = ({ predictions, classes }) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'benign' | 'attack'>('all');
  const [classFilter, setClassFilter] = useState<string>('all');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [rowsPerPage, setRowsPerPage] = useState<number>(10);

  // Map predictions to include original row indices (1-indexed)
  const indexedPredictions = useMemo(() => {
    return predictions.map((p, idx) => ({
      ...p,
      rowNumber: idx + 1,
    }));
  }, [predictions]);

  // Filter logic
  const filteredPredictions = useMemo(() => {
    return indexedPredictions.filter((item) => {
      // Status filter
      if (statusFilter === 'benign' && item.is_attack) return false;
      if (statusFilter === 'attack' && !item.is_attack) return false;

      // Class filter
      if (classFilter !== 'all' && item.prediction !== classFilter) return false;

      // Search query (case-insensitive search against prediction name, class ID, row #)
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase().trim();
        const matchesClass = item.prediction.toLowerCase().includes(query);
        const matchesId = String(item.class_id) === query;
        const matchesRow = String(item.rowNumber) === query;
        if (!matchesClass && !matchesId && !matchesRow) return false;
      }

      return true;
    });
  }, [indexedPredictions, statusFilter, classFilter, searchQuery]);

  // Reset page when filters change
  const totalFilteredRows = filteredPredictions.length;
  const totalPages = Math.max(1, Math.ceil(totalFilteredRows / rowsPerPage));

  // Current page slice
  const safeCurrentPage = Math.min(currentPage, totalPages);
  const startIndex = (safeCurrentPage - 1) * rowsPerPage;
  const currentRows = filteredPredictions.slice(startIndex, startIndex + rowsPerPage);

  const handleStatusFilterChange = (filter: 'all' | 'benign' | 'attack') => {
    setStatusFilter(filter);
    setCurrentPage(1);
  };

  const handleClassFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setClassFilter(e.target.value);
    setCurrentPage(1);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setCurrentPage(1);
  };

  const handleRowsPerPageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setRowsPerPage(Number(e.target.value));
    setCurrentPage(1);
  };

  return (
    <section className="dashboard-card results-table-card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Flow Classification Results</h2>
          <p className="card-subtitle">
            Granular inference telemetry per individual flow record
          </p>
        </div>
        <div className="results-count-badge">
          Showing <strong>{currentRows.length}</strong> of <strong>{totalFilteredRows}</strong> flows{' '}
          {totalFilteredRows !== predictions.length && `(filtered from ${predictions.length} total)`}
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="results-controls-bar">
        {/* Search */}
        <div className="search-input-wrapper">
          <Search size={16} className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Search prediction (e.g. DDoS, PortScan, BENIGN)..."
            value={searchQuery}
            onChange={handleSearchChange}
          />
          {searchQuery && (
            <button
              type="button"
              className="clear-search-btn"
              onClick={() => {
                setSearchQuery('');
                setCurrentPage(1);
              }}
            >
              ×
            </button>
          )}
        </div>

        <div className="filters-group">
          {/* Status Filter Buttons */}
          <div className="status-filter-pills">
            <button
              type="button"
              className={`pill-btn ${statusFilter === 'all' ? 'active' : ''}`}
              onClick={() => handleStatusFilterChange('all')}
            >
              All ({predictions.length})
            </button>
            <button
              type="button"
              className={`pill-btn pill-benign ${statusFilter === 'benign' ? 'active' : ''}`}
              onClick={() => handleStatusFilterChange('benign')}
            >
              Benign ({predictions.filter((p) => !p.is_attack).length})
            </button>
            <button
              type="button"
              className={`pill-btn pill-attack ${statusFilter === 'attack' ? 'active' : ''}`}
              onClick={() => handleStatusFilterChange('attack')}
            >
              Attack ({predictions.filter((p) => p.is_attack).length})
            </button>
          </div>

          {/* Class Filter Dropdown */}
          <div className="class-filter-wrapper">
            <Filter size={14} className="filter-icon" />
            <select
              className="class-select"
              value={classFilter}
              onChange={handleClassFilterChange}
            >
              <option value="all">All Classes ({classes.length})</option>
              {classes.map((cls) => (
                <option key={cls} value={cls}>
                  {cls}
                </option>
              ))}
            </select>
          </div>

          {/* Rows per page */}
          <div className="rows-per-page-wrapper">
            <span className="control-label">Rows:</span>
            <select
              className="rows-select"
              value={rowsPerPage}
              onChange={handleRowsPerPageChange}
            >
              <option value={10}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table Content */}
      <div className="card-body no-padding">
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '80px' }}>Row</th>
                <th>Prediction</th>
                <th style={{ width: '100px' }} className="text-center">Class ID</th>
                <th style={{ width: '160px' }} className="text-right">Confidence</th>
                <th style={{ width: '140px' }} className="text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {currentRows.length > 0 ? (
                currentRows.map((row) => {
                  const confidencePercent = (row.confidence * 100).toFixed(2);
                  return (
                    <tr key={row.rowNumber} className={row.is_attack ? 'row-attack-highlight' : ''}>
                      <td className="font-mono text-muted">#{row.rowNumber}</td>
                      <td className="font-medium">
                        <span className={`prediction-cell-name ${row.is_attack ? 'text-rose' : 'text-emerald'}`}>
                          {row.prediction}
                        </span>
                      </td>
                      <td className="text-center font-mono">{row.class_id}</td>
                      <td className="text-right font-mono font-medium">
                        <div className="confidence-display-cell">
                          <span>{confidencePercent}%</span>
                          <div className="confidence-micro-bar">
                            <div
                              className={`confidence-micro-fill ${
                                row.is_attack ? 'fill-attack' : 'fill-benign'
                              }`}
                              style={{ width: `${Math.min(100, row.confidence * 100)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="text-center">
                        {row.is_attack ? (
                          <span className="badge badge-attack">ATTACK</span>
                        ) : (
                          <span className="badge badge-benign">BENIGN</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={5} className="table-empty-row">
                    No flow records match the current filter or search criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        {totalPages > 1 && (
          <div className="pagination-bar">
            <div className="pagination-info">
              Showing flows <strong>{startIndex + 1}</strong> to{' '}
              <strong>{Math.min(startIndex + rowsPerPage, totalFilteredRows)}</strong> of{' '}
              <strong>{totalFilteredRows}</strong>
            </div>

            <div className="pagination-controls">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCurrentPage(1)}
                disabled={safeCurrentPage === 1}
                title="First Page"
              >
                <ChevronsLeft size={16} />
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={safeCurrentPage === 1}
                title="Previous Page"
              >
                <ChevronLeft size={16} />
                Prev
              </button>

              <span className="page-indicator">
                Page <strong>{safeCurrentPage}</strong> of <strong>{totalPages}</strong>
              </span>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={safeCurrentPage === totalPages}
                title="Next Page"
              >
                Next
                <ChevronRight size={16} />
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCurrentPage(totalPages)}
                disabled={safeCurrentPage === totalPages}
                title="Last Page"
              >
                <ChevronsRight size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
