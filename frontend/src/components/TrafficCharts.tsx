import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { PredictionResponse } from '../types/api';

interface TrafficChartsProps {
  predictions: PredictionResponse[];
  classes: string[];
}

const ATTACK_COLORS = [
  '#ef4444', // Red / Rose
  '#f97316', // Orange
  '#f59e0b', // Amber
  '#8b5cf6', // Violet
  '#ec4899', // Pink
  '#06b6d4', // Cyan
  '#14b8a6', // Teal
  '#6366f1', // Indigo
  '#d946ef', // Fuchsia
  '#84cc16', // Lime
  '#e11d48', // Crimson
];

const DONUT_COLORS = {
  Benign: '#10b981', // Emerald
  Attack: '#ef4444', // Red
};

export const TrafficCharts: React.FC<TrafficChartsProps> = ({ predictions, classes }) => {
  const totalFlows = predictions.length;

  // 1. Calculate distribution across all classes
  const classCountMap: Record<string, number> = {};
  classes.forEach((cls) => {
    classCountMap[cls] = 0;
  });
  predictions.forEach((p) => {
    classCountMap[p.prediction] = (classCountMap[p.prediction] || 0) + 1;
  });

  const allClassDistributionData = Object.entries(classCountMap)
    .map(([name, count]) => ({
      name,
      count,
      percentage: totalFlows > 0 ? Number(((count / totalFlows) * 100).toFixed(2)) : 0,
    }))
    .filter((d) => d.count > 0) // Only plot classes with >0 occurrences for chart clarity
    .sort((a, b) => b.count - a.count);

  // 2. Benign vs Attack Data
  const benignCount = predictions.filter((p) => !p.is_attack).length;
  const attackCount = predictions.filter((p) => p.is_attack).length;

  const donutData = [
    {
      name: 'Benign',
      value: benignCount,
      percentage: totalFlows > 0 ? ((benignCount / totalFlows) * 100).toFixed(1) : '0',
    },
    {
      name: 'Attack',
      value: attackCount,
      percentage: totalFlows > 0 ? ((attackCount / totalFlows) * 100).toFixed(1) : '0',
    },
  ].filter((d) => d.value > 0);

  // 3. Attack Type Distribution Data (Excluding BENIGN)
  const attackClassDistributionData = Object.entries(classCountMap)
    .filter(([name, count]) => name !== 'BENIGN' && count > 0)
    .map(([name, count], index) => ({
      name,
      count,
      percentage: totalFlows > 0 ? Number(((count / totalFlows) * 100).toFixed(2)) : 0,
      color: ATTACK_COLORS[index % ATTACK_COLORS.length],
    }))
    .sort((a, b) => b.count - a.count);

  return (
    <section className="traffic-charts-section">
      <div className="section-heading">
        <h2 className="section-title">Traffic Visualizations</h2>
        <p className="section-subtitle">Graphical telemetry derived directly from model predictions</p>
      </div>

      <div className="charts-grid">
        {/* Chart 1: Traffic Classification Distribution */}
        <div className="dashboard-card chart-card full-width">
          <div className="card-header">
            <div>
              <h3 className="card-title">1. Traffic Classification Distribution</h3>
              <p className="card-subtitle">Detected volume and distribution across predicted classes</p>
            </div>
          </div>
          <div className="card-body chart-container" style={{ height: 320 }}>
            {allClassDistributionData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={allClassDistributionData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 45 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis
                    dataKey="name"
                    stroke="#a3a3a3"
                    angle={-25}
                    textAnchor="end"
                    interval={0}
                    height={60}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis stroke="#a3a3a3" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#171717',
                      borderColor: '#404040',
                      borderRadius: 6,
                      color: '#f5f5f5',
                    }}
                    formatter={(value: any) => [
                      `${value ?? 0} flows (${totalFlows > 0 ? (((Number(value) || 0) / totalFlows) * 100).toFixed(1) : 0}%)`,
                      'Count',
                    ]}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {allClassDistributionData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.name === 'BENIGN' ? '#10b981' : ATTACK_COLORS[index % ATTACK_COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty-placeholder">No class distribution data</div>
            )}
          </div>
        </div>

        {/* Chart 2: Benign vs Attack */}
        <div className="dashboard-card chart-card">
          <div className="card-header">
            <div>
              <h3 className="card-title">2. Benign vs Attack</h3>
              <p className="card-subtitle">Global proportion of legitimate vs malicious traffic</p>
            </div>
          </div>
          <div className="card-body chart-container" style={{ height: 280 }}>
            {donutData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={donutData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={95}
                    paddingAngle={3}
                    dataKey="value"
                    label={({ name, percent }: { name?: string; percent?: number }) =>
                      `${name ?? ''}: ${((percent ?? 0) * 100).toFixed(1)}%`
                    }
                    labelLine={{ stroke: '#525252' }}
                  >
                    {donutData.map((entry) => (
                      <Cell
                        key={`donut-cell-${entry.name}`}
                        fill={DONUT_COLORS[entry.name as keyof typeof DONUT_COLORS] || '#737373'}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#171717',
                      borderColor: '#404040',
                      borderRadius: 6,
                      color: '#f5f5f5',
                    }}
                    formatter={(value: any, name: any) => [
                      `${value ?? 0} flows (${totalFlows > 0 ? (((Number(value) || 0) / totalFlows) * 100).toFixed(1) : 0}%)`,
                      String(name ?? 'Value'),
                    ]}
                  />
                  <Legend verticalAlign="bottom" height={36} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty-placeholder">No traffic data</div>
            )}
          </div>
        </div>

        {/* Chart 3: Attack Type Distribution */}
        <div className="dashboard-card chart-card">
          <div className="card-header">
            <div>
              <h3 className="card-title">3. Attack Type Distribution</h3>
              <p className="card-subtitle">Breakdown of specific attack vectors identified</p>
            </div>
          </div>
          <div className="card-body chart-container" style={{ height: 280 }}>
            {attackClassDistributionData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={attackClassDistributionData}
                  layout="vertical"
                  margin={{ top: 10, right: 30, left: 60, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis type="number" stroke="#a3a3a3" />
                  <YAxis
                    type="category"
                    dataKey="name"
                    stroke="#a3a3a3"
                    width={90}
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#171717',
                      borderColor: '#404040',
                      borderRadius: 6,
                      color: '#f5f5f5',
                    }}
                    formatter={(value: any) => [
                      `${value ?? 0} flows (${totalFlows > 0 ? (((Number(value) || 0) / totalFlows) * 100).toFixed(1) : 0}%)`,
                      'Attack Count',
                    ]}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {attackClassDistributionData.map((entry, index) => (
                      <Cell key={`attack-cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty-placeholder text-emerald">
                ✓ No attack traffic identified in the analyzed dataset.
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
