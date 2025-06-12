import React from 'react';
import { colors, spacing } from '../design-system/tokens';
import { MetricsSnapshot } from '../hooks/useMetrics';

interface Props {
  metrics: MetricsSnapshot;
}

export default function MetricsChart({ metrics }: Props) {
  const entries = Object.entries(metrics);
  const max = Math.max(...entries.map(e => e[1]), 1);

  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: spacing.sm, height: '6rem' }}>
      {entries.map(([key, value]) => (
        <div key={key} style={{ flex: 1, textAlign: 'center' }}>
          <div
            style={{
              height: `${(value / max) * 100}%`,
              backgroundColor: colors.primary,
              borderRadius: spacing.xs,
              transition: 'height 0.3s',
            }}
          />
          <div style={{ fontSize: '0.75rem', marginTop: spacing.xs }}>{key}</div>
        </div>
      ))}
    </div>
  );
}
