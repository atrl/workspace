import { describe, expect, test } from 'vitest';
import {
  buildActionMarkers,
  clampComparisonProfiles,
  clipRowsByDate,
  computeCommonWindow,
  computeInspectorRows,
  computeVisibleComparison,
} from './domain';
import { comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';

describe('dashboard v2 domain helpers', () => {
  test('clamps comparison profiles to two through four unique models', () => {
    expect(clampComparisonProfiles(['a'], ['a', 'b', 'c'])).toEqual(['a', 'b']);
    expect(clampComparisonProfiles(['a', 'a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e'])).toEqual([
      'a',
      'b',
      'c',
      'd',
    ]);
  });

  test('computes common overlap window across selected models and market rows', () => {
    expect(computeCommonWindow(marketFixture.rows, [modelFixture.rows])).toEqual({
      start: '2026-01-02',
      end: '2026-01-06',
    });
  });

  test('builds action markers from event rows instead of price movement', () => {
    expect(buildActionMarkers(modelFixture.events).map((marker) => marker.text)).toEqual(['BUY', 'SELL']);
  });

  test('computes visible comparison deltas against baseline', () => {
    const rows = computeVisibleComparison(comparisonFixture, '2026-01-02', '2026-01-06');
    expect(rows[1].profiles.smoke.nav_delta_vs_baseline).toBe(5);
  });

  test('clips rows to selected date inputs', () => {
    expect(clipRowsByDate(marketFixture.rows, '2026-01-05', '2026-01-06').map((row) => row.date)).toEqual([
      '2026-01-05',
      '2026-01-06',
    ]);
  });

  test('computes inspector rows for hovered date', () => {
    const rows = computeInspectorRows({
      date: '2026-01-05',
      manifest: manifestFixture,
      market: marketFixture,
      models: { smoke: modelFixture },
      comparison: comparisonFixture,
    });
    expect(rows.some((row) => row.label === 'SOXL close' && row.value === '11.00')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke action' && row.value === 'BUY')).toBe(true);
    expect(rows.some((row) => row.label === 'action mismatches' && row.value === '1')).toBe(true);
  });
});
