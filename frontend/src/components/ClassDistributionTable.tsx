import React, { useState } from 'react';
import { ChevronDown, ChevronUp, ShieldAlert, ShieldCheck } from 'lucide-react';
import type { PredictionResponse } from '../types/api';

interface ClassDistributionTableProps {
  predictions: PredictionResponse[];
  classes: string[];
}

export const ClassDistributionTable: React.FC<ClassDistributionTableProps> = ({
  predictions,
  classes,
}) => {
  const [showZeroClasses, setShowZeroClasses] = useState(true);
  const totalFlows = predictions.length;
  const attackFlows = predictions.filter((p) => p.is_attack).length;

  // Compute counts for all 15 classes
  const countMap: Record<string, number> = {};
  classes.forEach((cls) => {
    countMap[cls] = 0;
  });
  predictions.forEach((p) => {
    countMap[p.prediction] = (countMap[p.prediction] || 0) + 1;
  });

  // Full 15-class list, sorted descending by count
  const allClassRows = classes
    .map((className) => {
      const count = countMap[className] || 0;
      const percentage = totalFlows > 0 ? (count / totalFlows) * 100 : 0;
      return {
        className,
        count,
        percentage,
        isAttack: className !== 'BENIGN',
      };
    })
    .sort((a, b) => b.count - a.count);

  const displayedAllRows = showZeroClasses
    ? allClassRows
    : allClassRows.filter((r) => r.count > 0);

  // Attack-only breakdown (excluding BENIGN)
  const attackRows = allClassRows
    .filter((r) => r.isAttack && r.count > 0)
    .map((r) => ({
      ...r,
      attackShare: attackFlows > 0 ? (r.count / attackFlows) * 100 : 0,
    }));

  return (
    <div className="distribution-tables-section">
      <div className="section-grid-2col">
        {/* Section 8: All 15 Classes Distribution */}
        <section className="dashboard-card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Class Distribution (All 15 Classes)</h3>
              <p className="card-subtitle">
                Comprehensive frequency across the 15 supported traffic classifications
              </p>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setShowZeroClasses(!showZeroClasses)}
            >
              {showZeroClasses ? (
                <>
                  <ChevronUp size={14} /> Hide Zero Counts
                </>
              ) : (
                <>
                  <ChevronDown size={14} /> Show All 15 Classes
                </>
              )}
            </button>
          </div>

          <div className="card-body no-padding">
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Category</th>
                    <th className="text-right">Count</th>
                    <th className="text-right">Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {displayedAllRows.map((row) => (
                    <tr key={row.className} className={row.count > 0 ? 'row-active' : 'row-zero'}>
                      <td className="font-medium">
                        <span className="class-name-cell">{row.className}</span>
                      </td>
                      <td>
                        {row.isAttack ? (
                          <span className="badge badge-attack-sm">Attack</span>
                        ) : (
                          <span className="badge badge-benign-sm">Benign</span>
                        )}
                      </td>
                      <td className="text-right font-mono">{row.count.toLocaleString()}</td>
                      <td className="text-right font-mono">
                        <div className="table-percentage-cell">
                          <span>{row.percentage.toFixed(2)}%</span>
                          <div className="mini-progress-bar">
                            <div
                              className={`mini-progress-fill ${row.isAttack ? 'fill-attack' : 'fill-benign'}`}
                              style={{ width: `${Math.min(100, row.percentage)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Section 9: Attack Breakdown */}
        <section className="dashboard-card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Attack Breakdown</h3>
              <p className="card-subtitle">
                Specific intrusion categories identified (excluding BENIGN)
              </p>
            </div>
            <div className="card-tag">
              <ShieldAlert size={14} />
              {attackFlows} Attack Flows
            </div>
          </div>

          <div className="card-body no-padding">
            {attackRows.length > 0 ? (
              <div className="table-responsive">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Attack Type</th>
                      <th className="text-right">Count</th>
                      <th className="text-right">% of All Traffic</th>
                      <th className="text-right">% of Attacks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {attackRows.map((row) => (
                      <tr key={row.className}>
                        <td className="font-medium text-rose">{row.className}</td>
                        <td className="text-right font-mono font-bold text-rose">
                          {row.count.toLocaleString()}
                        </td>
                        <td className="text-right font-mono">{row.percentage.toFixed(2)}%</td>
                        <td className="text-right font-mono font-medium">
                          {row.attackShare.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-attack-state">
                <ShieldCheck size={36} className="text-emerald" />
                <div className="empty-attack-title">No Intrusions Detected</div>
                <div className="empty-attack-desc">
                  All {totalFlows} analyzed flows were classified as legitimate BENIGN traffic.
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};
