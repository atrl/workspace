import { describe, expect, test } from 'vitest';
import {
  buildActionMarkers,
  clampComparisonProfiles,
  clipRowsByDate,
  computeCommonWindow,
  computeDateRangePreset,
  computeInspectorRows,
  computeKeyboardDateRange,
  computeVisibleComparison,
} from './domain';
import { comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';
import { viewUpdateRequiresChartRender } from './viewState';

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
    expect(rows.some((row) => row.label === 'smoke BuyHold NAV' && row.value === '110.00')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke vs BuyHold' && row.value === '-5.00%')).toBe(true);
    expect(rows.some((row) => row.label === 'action mismatches' && row.value === '1')).toBe(true);
  });

  test('treats hover-only updates as inspector updates instead of chart renders', () => {
    expect(viewUpdateRequiresChartRender({ hoveredDate: '2026-01-06' })).toBe(false);
    expect(viewUpdateRequiresChartRender({ start: '2026-01-05' })).toBe(true);
    expect(viewUpdateRequiresChartRender({ selectedProfiles: ['smoke', 'balanced_rule_residual_v22'] })).toBe(true);
    expect(viewUpdateRequiresChartRender({ baselineProfile: 'balanced_rule_residual_v22' })).toBe(true);
  });

  test('computes trading range presets from available dates', () => {
    const rows = [
      { date: '2025-12-31' },
      { date: '2026-01-02' },
      { date: '2026-03-06' },
      { date: '2026-05-08' },
      { date: '2026-06-09' },
    ];

    expect(computeDateRangePreset(rows, 'YTD')).toEqual({ start: '2026-01-02', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, '1M')).toEqual({ start: '2026-05-08', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, '3M')).toEqual({ start: '2026-03-06', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, 'ALL')).toEqual({ start: '2025-12-31', end: '2026-06-09' });
  });

  test('computes keyboard zoom, pan, and reset ranges', () => {
    const rows = [
      { date: '2026-01-02' },
      { date: '2026-01-05' },
      { date: '2026-01-06' },
      { date: '2026-01-07' },
      { date: '2026-01-08' },
      { date: '2026-01-09' },
    ];

    expect(computeKeyboardDateRange(rows, '2026-01-02', '2026-01-09', 'zoom_in')).toEqual({
      start: '2026-01-06',
      end: '2026-01-08',
    });
    expect(computeKeyboardDateRange(rows, '2026-01-05', '2026-01-08', 'pan_right')).toEqual({
      start: '2026-01-06',
      end: '2026-01-09',
    });
    expect(computeKeyboardDateRange(rows, '2026-01-05', '2026-01-08', 'reset')).toEqual({
      start: '2026-01-02',
      end: '2026-01-09',
    });
  });
});
