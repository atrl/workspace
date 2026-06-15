import { describe, expect, test } from 'vitest';
import {
  buildActionMarkers,
  buildModelDifferenceMarkers,
  clampComparisonProfiles,
  clipRowsByDate,
  computeActionMismatchRows,
  computeCommonWindow,
  computeDateRangePreset,
  computeDefaultDateRange,
  computeDateRangeFromLogicalRange,
  computeInspectorRows,
  computeKeyboardDateRange,
  computeVisibleComparison,
  formatProfileLabel,
  selectDefaultChartProfile,
} from './domain';
import { comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';
import { viewUpdateRequiresChartRender } from './viewState';

describe('dashboard v2 domain helpers', () => {
  test('formats long profile ids into readable display labels', () => {
    expect(formatProfileLabel('soxl_backtest_policy_reimplementation_v1')).toBe('SOXLBacktest');
    expect(formatProfileLabel('balanced_rule_residual_v22')).toBe('BRR v22');
    expect(formatProfileLabel('mode_conditioned_rebound_reward_train_v2')).toBe('Mode rebound v2');
    expect(formatProfileLabel('state_conditioned_drawdown_rebound_reward_train_v1')).toBe('State drawdown v1');
    expect(formatProfileLabel('smoke')).toBe('smoke');
  });

  test('clamps comparison profiles to two through six unique models', () => {
    expect(clampComparisonProfiles(['a'], ['a', 'b', 'c'])).toEqual(['a', 'b']);
    expect(clampComparisonProfiles(['a', 'a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e'])).toEqual([
      'a',
      'b',
      'c',
      'd',
      'e',
    ]);
  });

  test('computes common overlap window across selected models and market rows', () => {
    expect(computeCommonWindow(marketFixture.rows, [modelFixture.rows])).toEqual({
      start: '2026-01-02',
      end: '2026-01-06',
    });
  });

  test('defaults visible range to fresh market and replayable actor rows instead of short reference baseline', () => {
    const referenceModel = {
      ...modelFixture,
      profile: 'soxl_backtest_policy_reimplementation_v1',
      status: 'reference',
      replay_end: '2026-01-05',
      rows: modelFixture.rows.slice(0, 2),
    };
    const actorModel = {
      ...modelFixture,
      profile: 'balanced_rule_residual_v22',
      status: 'canonical',
      replay_end: '2026-01-06',
    };

    expect(
      computeDefaultDateRange({
        manifest: manifestFixture,
        marketRows: marketFixture.rows,
        models: {
          soxl_backtest_policy_reimplementation_v1: referenceModel,
          balanced_rule_residual_v22: actorModel,
        },
        comparison: {
          ...comparisonFixture,
          common_end: '2026-01-05',
          rows: comparisonFixture.rows.slice(0, 2),
        },
      }),
    ).toEqual({
      start: '2026-01-02',
      end: '2026-01-06',
    });
  });

  test('defaults visible range to YTD instead of all replay history', () => {
    const marketRows = [
      { date: '2025-12-31', open: 9, high: 10, low: 8, close: 9.5, volume: 900 },
      ...marketFixture.rows,
    ];
    const longActor = {
      ...modelFixture,
      rows: [
        { date: '2025-12-31', strategy_nav: 99, buyhold_nav: 99, position: 0, action: 'HOLD' as const, price: 9.5, excess: 0, excess_pct: 0 },
        ...modelFixture.rows,
      ],
    };

    expect(
      computeDefaultDateRange({
        manifest: manifestFixture,
        marketRows,
        models: { smoke: longActor },
        comparison: comparisonFixture,
      }),
    ).toEqual({
      start: '2026-01-02',
      end: '2026-01-06',
    });
  });

  test('selects a replayable actor as the default chart model ahead of reference baseline', () => {
    expect(
      selectDefaultChartProfile(
        ['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v22'],
        {
          soxl_backtest_policy_reimplementation_v1: {
            ...modelFixture,
            profile: 'soxl_backtest_policy_reimplementation_v1',
            status: 'reference',
          },
          balanced_rule_residual_v22: {
            ...modelFixture,
            profile: 'balanced_rule_residual_v22',
            status: 'canonical',
          },
        },
        'soxl_backtest_policy_reimplementation_v1',
      ),
    ).toBe('balanced_rule_residual_v22');
  });

  test('builds action markers from event rows instead of price movement', () => {
    expect(buildActionMarkers(modelFixture.events).map((marker) => marker.text)).toEqual(['BUY', 'SELL']);
  });

  test('builds visible model-difference markers for buy/sell action mismatches against baseline', () => {
    expect(
      buildModelDifferenceMarkers(comparisonFixture, ['smoke', 'balanced_rule_residual_v22'], '2026-01-02', '2026-01-06'),
    ).toEqual([
      {
        time: '2026-01-05',
        position: 'belowBar',
        color: '#39b7c5',
        shape: 'arrowUp',
        text: '',
      },
      {
        time: '2026-01-06',
        position: 'aboveBar',
        color: '#39b7c5',
        shape: 'arrowDown',
        text: '',
      },
    ]);
  });

  test('computes visible comparison deltas against baseline', () => {
    const rows = computeVisibleComparison(comparisonFixture, '2026-01-02', '2026-01-06');
    expect(rows[1].profiles.smoke.nav_delta_vs_baseline).toBe(5);
  });

  test('computes action mismatch rows with market feature snapshots', () => {
    const rows = computeActionMismatchRows(comparisonFixture, marketFixture, '2026-01-02', '2026-01-06');

    expect(rows).toEqual([
      {
        date: '2026-01-05',
        profile: 'smoke',
        model_action: 'BUY',
        baseline_action: 'HOLD',
        model_position: 0.7,
        baseline_position: 0,
        position_delta_vs_baseline: 0.7,
        nav_delta_vs_baseline: 5,
        close: 11,
        features: { rsi6: '60', spread: '0.4', vix: '18' },
      },
      {
        date: '2026-01-06',
        profile: 'smoke',
        model_action: 'SELL',
        baseline_action: 'HOLD',
        model_position: 0.2,
        baseline_position: 0,
        position_delta_vs_baseline: 0.2,
        nav_delta_vs_baseline: 2,
        close: 10.8,
        features: { rsi6: '58', vix: '18.5' },
      },
    ]);
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
    expect(rows.some((row) => row.label === 'SOXL open' && row.value === '10.20')).toBe(true);
    expect(rows.some((row) => row.label === 'SOXL high' && row.value === '11.50')).toBe(true);
    expect(rows.some((row) => row.label === 'SOXL low' && row.value === '10.10')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke action' && row.value === 'BUY')).toBe(true);
    expect(rows.some((row) => row.label === 'action mismatches' && row.value === '1')).toBe(true);
  });

  test('computes inspector model state from the latest row at or before the hovered market date', () => {
    const rows = computeInspectorRows({
      date: '2026-01-07',
      manifest: manifestFixture,
      market: {
        ...marketFixture,
        rows: [
          ...marketFixture.rows,
          { date: '2026-01-07', open: 10.7, high: 10.9, low: 10.2, close: 10.4, volume: 1400 },
        ],
      },
      models: { smoke: modelFixture },
      comparison: comparisonFixture,
    });

    expect(rows.some((row) => row.label === 'smoke as-of' && row.value === '2026-01-06')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke position' && row.value === '0.200')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke NAV' && row.value === '103.00')).toBe(true);
  });

  test('computes inspector model state from latest signal without replay NAV', () => {
    const rows = computeInspectorRows({
      date: '2026-01-07',
      manifest: manifestFixture,
      market: {
        ...marketFixture,
        rows: [
          ...marketFixture.rows,
          { date: '2026-01-07', open: 10.7, high: 10.9, low: 10.2, close: 10.4, volume: 1400 },
        ],
      },
      models: {
        smoke: {
          ...modelFixture,
          latest_signal: {
            date: '2026-01-07',
            price: 10.4,
            position: 0.55,
            action: 'BUY',
            replay_asof: '2026-01-06',
            source: 'latest_signal_stateless',
          },
        },
      },
      comparison: comparisonFixture,
    });

    expect(rows.some((row) => row.label === 'smoke signal' && row.value === 'latest_signal_stateless')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke replay as-of' && row.value === '2026-01-06')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke position' && row.value === '0.550')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke NAV')).toBe(false);
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

  test('maps chart logical ranges back to snapped market dates', () => {
    const rows = [
      { date: '2026-01-02' },
      { date: '2026-01-05' },
      { date: '2026-01-06' },
      { date: '2026-01-07' },
      { date: '2026-01-08' },
    ];

    expect(computeDateRangeFromLogicalRange(rows, { from: 1.2, to: 3.7 })).toEqual({
      start: '2026-01-05',
      end: '2026-01-08',
    });
    expect(computeDateRangeFromLogicalRange(rows, { from: -10, to: 99 })).toEqual({
      start: '2026-01-02',
      end: '2026-01-08',
    });
  });
});
