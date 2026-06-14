import { describe, expect, test } from 'vitest';
import {
  buildActionFeatureRows,
  buildBaselineFeatureContrastRows,
  buildActionMarkers,
  clampComparisonProfiles,
  clipRowsByDate,
  computeCommonWindow,
  computeDateRangePreset,
  computeInspectorRows,
  computeKeyboardDateRange,
  computeVisibleComparison,
  chooseDefaultChartProfile,
  runnableProfileReason,
} from './domain';
import { betaModelFixture, comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';
import { viewUpdateRequiresChartRender, viewUpdateRequiresFeatureRender } from './viewState';

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

  test('uses a reference baseline as the default chart profile when no runnable model remains', () => {
    const reference = {
      ...modelFixture,
      profile: 'soxl_backtest_policy_reimplementation_v1',
      runnable: false,
      runnable_reason: 'reference_baseline',
    };

    expect(
      chooseDefaultChartProfile(
        {
          ...manifestFixture,
          available_profiles: ['soxl_backtest_policy_reimplementation_v1'],
          default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
          default_chart_profile: 'soxl_backtest_policy_reimplementation_v1',
          default_comparison_profiles: ['soxl_backtest_policy_reimplementation_v1'],
        },
        { soxl_backtest_policy_reimplementation_v1: reference },
      ),
    ).toBe('soxl_backtest_policy_reimplementation_v1');
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

  test('builds buy/sell feature rows from each model-specific snapshot', () => {
    const rows = buildActionFeatureRows({
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
    });

    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      profile: 'smoke',
      action: 'BUY',
      features: { alpha_signal: 1.25, shared_feature: 10 },
    });
    expect(rows[1]).toMatchObject({
      profile: 'balanced_rule_residual_v22',
      action: 'SELL',
      features: { beta_signal: -2.5, shared_feature: 20 },
    });
  });

  test('builds buy/sell feature contrasts against the selected baseline', () => {
    const rows = buildBaselineFeatureContrastRows({
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
      baselineProfile: 'balanced_rule_residual_v22',
      limitPerEvent: 10,
    });

    const sharedFeature = rows.find((row) => row.date === '2026-01-05' && row.feature === 'shared_feature');
    expect(sharedFeature).toMatchObject({
      profile: 'smoke',
      action: 'BUY',
      baseline_action: 'SELL',
      value: 10,
      baseline_value: 20,
      delta: -10,
    });
    const modelSpecificFeature = rows.find((row) => row.date === '2026-01-05' && row.feature === 'alpha_signal');
    expect(modelSpecificFeature).toMatchObject({
      profile: 'smoke',
      baseline_value: null,
      delta: null,
    });
  });

  test('treats hover-only updates as inspector updates instead of chart renders', () => {
    expect(viewUpdateRequiresChartRender({ hoveredDate: '2026-01-06' })).toBe(false);
    expect(viewUpdateRequiresChartRender({ selectedDate: '2026-01-06' })).toBe(false);
    expect(viewUpdateRequiresFeatureRender({ hoveredDate: '2026-01-06' })).toBe(false);
    expect(viewUpdateRequiresFeatureRender({ selectedDate: '2026-01-06' })).toBe(true);
    expect(viewUpdateRequiresChartRender({ start: '2026-01-05' })).toBe(true);
    expect(viewUpdateRequiresFeatureRender({ start: '2026-01-05' })).toBe(true);
    expect(viewUpdateRequiresChartRender({ selectedProfiles: ['smoke', 'balanced_rule_residual_v22'] })).toBe(true);
    expect(viewUpdateRequiresChartRender({ baselineProfile: 'balanced_rule_residual_v22' })).toBe(true);
  });

  test('computes trading range presets from available dates', () => {
    const rows = [
      { date: '2025-06-09' },
      { date: '2025-12-31' },
      { date: '2026-01-02' },
      { date: '2026-03-06' },
      { date: '2026-05-08' },
      { date: '2026-06-09' },
    ];

    expect(computeDateRangePreset(rows, 'YTD')).toEqual({ start: '2026-01-02', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, '1M')).toEqual({ start: '2026-05-08', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, '3M')).toEqual({ start: '2026-03-06', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, '6M')).toEqual({ start: '2025-12-31', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, '1Y')).toEqual({ start: '2025-06-09', end: '2026-06-09' });
    expect(computeDateRangePreset(rows, 'ALL')).toEqual({ start: '2025-06-09', end: '2026-06-09' });
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

  test('explains why interactive rerun is unavailable for non-runnable profiles', () => {
    expect(runnableProfileReason(modelFixture)).toBe('ready for selected-window inference');
    expect(
      runnableProfileReason({
        ...betaModelFixture,
        runnable: false,
        runnable_reason: 'reference_baseline',
        source_kind: 'soxlbacktest_qlib_buysell',
      }),
    ).toBe('reference baseline only');
    expect(
      runnableProfileReason({
        ...betaModelFixture,
        runnable: false,
        runnable_reason: 'missing_actor_artifact',
      }),
    ).toBe('missing actor artifact');
  });
});
