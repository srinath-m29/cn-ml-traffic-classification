/**
 * Pre-compiled Demo CSV data containing 20 authentic network flows
 * with all 61 model features, including BENIGN, DDoS, PortScan, and DoS Hulk traffic.
 */

import { SAMPLE_FLOWS } from './sampleFlows';

export function generateDemoCsvContent(): string {
  if (!SAMPLE_FLOWS || SAMPLE_FLOWS.length === 0) return '';

  const headers = Object.keys(SAMPLE_FLOWS[0].features);
  const rows: string[] = [];
  rows.push(headers.join(','));

  // Generate 20 realistic rows based on the authentic samples
  for (let i = 0; i < 20; i++) {
    const sampleIdx = i % SAMPLE_FLOWS.length;
    const baseFlow = SAMPLE_FLOWS[sampleIdx].features;

    const rowValues = headers.map((header) => {
      let val = baseFlow[header] ?? 0;
      // Small jitter for packet/duration features across duplicate rows to simulate multiple distinct flows
      if (i >= SAMPLE_FLOWS.length) {
        if (header === 'Flow Duration') {
          val = Math.max(1, Math.round(val * (1 + (i % 5) * 0.05)));
        } else if (header === 'Flow Packets/s' || header === 'Flow Bytes/s') {
          val = Number((val * (1 + (i % 3) * 0.02)).toFixed(4));
        }
      }
      return String(val);
    });

    rows.push(rowValues.join(','));
  }

  return rows.join('\n');
}
