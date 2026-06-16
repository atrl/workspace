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
  computeFocusedDateRange,
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

  test('defaults visible range through latest market date when price extends past replay actors', () => {
    const marketRows = [
      ...marketFixture.rows,
      { date: '2026-01-07', open: 10.7, high: 10.9, low: 10.2, close: 10.4, volume: 1400 },
    ];
    const referenceModel = {
      ...modelFixture,
      profile: 'soxl_backtest_policy_reimplementation_v1',
      status: 'reference',
      replay_end: '2026-01-07',
      rows: [
        ...modelFixture.rows,
        {
          date: '2026-01-07',
          strategy_nav: 104,
          buyhold_nav: 104,
          position: 0.2,
          action: 'HOLD' as const,
          price: 10.4,
          excess: 0,
          excess_pct: 0.04,
        },
      ],
    };
    const actorModel = {
      ...modelFixture,
      profile: 'balanced_rule_residual_v22',
      status: 'canonical',
      replay_end: '2026-01-06',
    };

    expect(
      computeDefaultDateRange({
        manifest: { ...manifestFixture, latest_market_date: '2026-01-07', safe_backtest_end: '2026-01-06' },
        marketRows,
        models: {
          soxl_backtest_policy_reimplementation_v1: referenceModel,
          balanced_rule_residual_v22: actorModel,
        },
        comparison: comparisonFixture,
      }),
    ).toEqual({
      start: '2026-01-02',
      end: '2026-01-07',
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

  test('caps the default K-line window to recent rows when YTD has too many candles', () => {
    const marketRows = Array.from({ length: 120 }, (_, index) => {
      const day = index + 1;
      return {
        date: `2026-${String(Math.floor((day - 1) / 28) + 1).padStart(2, '0')}-${String(((day - 1) % 28) + 1).padStart(2, '0')}`,
        open: 10,
        high: 11,
        low: 9,
        close: 10 + index,
        volume: 1000,
      };
    });
    const longModel = {
      ...modelFixture,
      rows: marketRows.map((row, index) => ({
        date: row.date,
        strategy_nav: 100 + index,
        buyhold_nav: 100 + index,
        position: 0.1,
        action: 'HOLD' as const,
        price: row.close,
        excess: 0,
        excess_pct: index / 100,
      })),
    };

    expect(
      computeDefaultDateRange({
        manifest: { ...manifestFixture, latest_market_date: marketRows.at(-1)!.date },
        marketRows,
        models: { smoke: longModel },
        comparison: {
          ...comparisonFixture,
          common_start: marketRows[0].date,
          common_end: marketRows.at(-1)!.date,
        },
      }),
    ).toEqual({
      start: marketRows[30].date,
      end: marketRows[119].date,
    });
  });

  test('preserves the explicit reference baseline as the default action comparison model', () => {
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
    ).toBe('soxl_backtest_policy_reimplementation_v1');
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
        color: '#22c7a9',
        shape: 'circle',
        text: '',
      },
      {
        time: '2026-01-06',
        position: 'aboveBar',
        color: '#22c7a9',
        shape: 'circle',
        text: '',
      },
    ]);
  });

  test('keeps pure position divergence out of the main K-line markers', () => {
    const sameActionComparison = {
      ...comparisonFixture,
      rows: comparisonFixture.rows.map((row) => ({
        ...row,
        action_mismatch_count: 0,
        profiles: {
          smoke: { ...row.profiles.smoke, action: 'HOLD' as const, action_mismatch: false },
          balanced_rule_residual_v22: {
            ...row.profiles.balanced_rule_residual_v22,
            action: 'HOLD' as const,
            action_mismatch: false,
          },
        },
      })),
    };

    expect(
      buildModelDifferenceMarkers(sameActionComparison, ['smoke', 'balanced_rule_residual_v22'], '2026-01-02', '2026-01-06'),
    ).toEqual([]);
  });

  test('keeps K-line readable by leaving position divergence to the delta pane', () => {
    const steadyDivergence = {
      ...comparisonFixture,
      rows: [
        ...comparisonFixture.rows,
        {
          date: '2026-01-07',
          action_mismatch_count: 0,
          profiles: {
            smoke: {
              ...comparisonFixture.rows[2].profiles.smoke,
              action: 'HOLD' as const,
              action_mismatch: false,
              position_delta_vs_baseline: 0.2,
            },
            balanced_rule_residual_v22: {
              ...comparisonFixture.rows[2].profiles.balanced_rule_residual_v22,
              action: 'HOLD' as const,
              action_mismatch: false,
            },
          },
        },
      ].map((row) => ({
        ...row,
        action_mismatch_count: 0,
        profiles: {
          smoke: { ...row.profiles.smoke, action: 'HOLD' as const, action_mismatch: false },
          balanced_rule_residual_v22: {
            ...row.profiles.balanced_rule_residual_v22,
            action: 'HOLD' as const,
            action_mismatch: false,
          },
        },
      })),
    };

    expect(
      buildModelDifferenceMarkers(steadyDivergence, ['smoke', 'balanced_rule_residual_v22'], '2026-01-02', '2026-01-07'),
    ).toEqual([]);
  });

  test('groups same-day model-difference markers to keep the K-line readable', () => {
    expect(
      buildModelDifferenceMarkers(
        {
          ...comparisonFixture,
          compared_profiles: ['smoke', 'balanced_rule_residual_v22', 'candidate'],
          rows: comparisonFixture.rows.map((row) => ({
            ...row,
            profiles: {
              ...row.profiles,
              candidate: { ...row.profiles.smoke },
            },
          })),
        },
        ['smoke', 'candidate', 'balanced_rule_residual_v22'],
        '2026-01-02',
        '2026-01-06',
      ),
    ).toEqual([
      {
        time: '2026-01-05',
        position: 'belowBar',
        color: '#22c7a9',
        shape: 'circle',
        text: '',
      },
      {
        time: '2026-01-06',
        position: 'aboveBar',
        color: '#22c7a9',
        shape: 'circle',
        text: '',
      },
    ]);
  });

  test('limits action-difference markers to a compact recent set by default', () => {
    const rows = Array.from({ length: 50 }, (_, index) => ({
      date: `2026-02-${String(index + 1).padStart(2, '0')}`,
      action_mismatch_count: 1,
      profiles: {
        smoke: {
          ...comparisonFixture.rows[0].profiles.smoke,
          action: index % 2 === 0 ? 'BUY' as const : 'SELL' as const,
          action_mismatch: true,
          position_delta_vs_baseline: index % 2 === 0 ? 0.2 : 0.5,
        },
        balanced_rule_residual_v22: {
          ...comparisonFixture.rows[0].profiles.balanced_rule_residual_v22,
          action: 'HOLD' as const,
          action_mismatch: false,
        },
      },
    }));

    const markers = buildModelDifferenceMarkers(
      { ...comparisonFixture, rows },
      ['smoke', 'balanced_rule_residual_v22'],
      '2026-02-01',
      '2026-02-50',
    );

    expect(markers).toHaveLength(8);
    expect(markers[0].time).toBe('2026-02-43');
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

  test('computes model-difference rows for material position divergence with same action labels', () => {
    const sameActionComparison = {
      ...comparisonFixture,
      rows: comparisonFixture.rows.map((row) => ({
        ...row,
        action_mismatch_count: 0,
        profiles: {
          smoke: { ...row.profiles.smoke, action: 'HOLD' as const, action_mismatch: false },
          balanced_rule_residual_v22: {
            ...row.profiles.balanced_rule_residual_v22,
            action: 'HOLD' as const,
            action_mismatch: false,
          },
        },
      })),
    };

    const rows = computeActionMismatchRows(sameActionComparison, marketFixture, '2026-01-02', '2026-01-06');

    expect(rows.map((row) => `${row.date}:${row.profile}:${row.position_delta_vs_baseline}`)).toEqual([
      '2026-01-05:smoke:0.7',
      '2026-01-06:smoke:0.2',
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

  test('computes a compact focused date range around a mismatch date', () => {
    const rows = [
      { date: '2026-01-02' },
      { date: '2026-01-05' },
      { date: '2026-01-06' },
      { date: '2026-01-07' },
      { date: '2026-01-08' },
      { date: '2026-01-09' },
      { date: '2026-01-12' },
    ];

    expect(computeFocusedDateRange(rows, '2026-01-07', 2)).toEqual({
      start: '2026-01-05',
      end: '2026-01-09',
    });
    expect(computeFocusedDateRange(rows, '2026-01-01', 2)).toEqual({
      start: '2026-01-02',
      end: '2026-01-06',
    });
    expect(computeFocusedDateRange(rows, '2026-01-30', 2)).toEqual({
      start: '2026-01-08',
      end: '2026-01-12',
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
