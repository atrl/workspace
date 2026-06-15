import { describe, expect, test } from 'vitest';
import {
  renderActionLegend,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderInspectorPanel,
  renderModelDeltaStrip,
} from './render';
import { comparisonFixture, manifestFixture } from './testFixtures';

describe('dashboard v2 DOM render helpers', () => {
  test('renders data source provenance and safe replay window', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, manifestFixture);

    expect(panel.querySelector('.status-strip')).not.toBeNull();
    expect(panel.textContent).toContain('Replay');
    expect(panel.textContent).toContain('2026-01-06');
    expect(panel.textContent).not.toContain('model_replay_freshness.json');
  });

  test('renders default model replay freshness dates without opening diagnostics', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, {
      ...manifestFixture,
      default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
      default_comparison_profiles: ['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v22'],
      profile_summaries: {
        soxl_backtest_policy_reimplementation_v1: {
          status: 'reference',
          replay_start: '2026-01-02',
          replay_end: '2026-01-06',
          row_count: 3,
          source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
        },
        balanced_rule_residual_v22: {
          status: 'canonical',
          replay_start: '2020-01-02',
          replay_end: '2026-01-05',
          row_count: 1500,
          source_path: 'runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv',
        },
      },
    });

    expect(panel.querySelector('.freshness-strip')?.textContent).toContain('SOXLBacktest');
    expect(panel.querySelector('.freshness-strip')?.textContent).toContain('2026-01-06');
    expect(panel.querySelector('.freshness-strip')?.textContent).toContain('BRR v22');
    expect(panel.querySelector('.freshness-strip')?.textContent).toContain('2026-01-05');
  });

  test('renders latest market versus actor and baseline dates as a visible freshness alert', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, {
      ...manifestFixture,
      latest_market_date: '2026-01-06',
      safe_backtest_end: '2026-01-05',
      default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
      default_comparison_profiles: ['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v22'],
      profile_summaries: {
        soxl_backtest_policy_reimplementation_v1: {
          status: 'reference',
          replay_start: '2026-01-02',
          replay_end: '2026-01-06',
          row_count: 3,
          source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
        },
        balanced_rule_residual_v22: {
          status: 'canonical',
          replay_start: '2026-01-02',
          replay_end: '2026-01-05',
          row_count: 2,
          source_path: 'runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv',
        },
      },
    });

    expect(panel.querySelector('.freshness-alert')?.textContent).toContain('Market 2026-01-06');
    expect(panel.querySelector('.freshness-alert')?.textContent).toContain('Actors 2026-01-05');
    expect(panel.querySelector('.freshness-alert')?.textContent).toContain('Baseline 2026-01-06');
  });

  test('keeps source warnings inside a collapsed diagnostics block', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, manifestFixture, ['Replay safe end is before market latest.']);

    const diagnostics = panel.querySelector<HTMLDetailsElement>('details.source-diagnostics');
    expect(diagnostics).not.toBeNull();
    expect(diagnostics?.open).toBe(false);
    expect(diagnostics?.querySelector('summary')?.textContent).toContain('Diagnostics');
    expect(diagnostics?.textContent).toContain('Replay safe end is before market latest.');
  });

  test('shows a compact retention gate banner when actor models are blocked', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, {
      ...manifestFixture,
      default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
      default_comparison_profiles: ['soxl_backtest_policy_reimplementation_v1'],
      profile_summaries: {
        soxl_backtest_policy_reimplementation_v1: {
          status: 'reference',
          replay_start: '2026-01-02',
          replay_end: '2026-01-06',
          row_count: 3,
          source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
        },
      },
      retention_filter: {
        policy: 'final_model_retained_or_reference',
        input_profile_count: 3,
        allowed_profile_count: 1,
        blocked_profile_count: 2,
      },
      blocked_profiles: {
        balanced_rule_residual_v22: {
          code: 'missing_final_retention_decision',
          status: 'blocked',
          reason: 'summary lacks final_model_retention_decision',
          summary_path: 'runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json',
          final_model_retained: false,
        },
      },
    });

    expect(panel.querySelector('.gate-banner')?.textContent).toContain('No final-retained actor');
    const warningPill = panel.querySelector<HTMLElement>('.status-pill[data-tone="warning"]');
    expect(warningPill?.textContent).toContain('Blocked');
    expect(warningPill?.textContent).toContain('2');
  });

  test('marks visible blocked actors as diagnostic comparisons', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, {
      ...manifestFixture,
      default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
      default_comparison_profiles: ['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v22'],
      profile_summaries: {
        soxl_backtest_policy_reimplementation_v1: {
          status: 'reference',
          replay_start: '2026-01-02',
          replay_end: '2026-01-06',
          row_count: 3,
          source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
        },
        balanced_rule_residual_v22: {
          status: 'canonical',
          replay_start: '2026-01-02',
          replay_end: '2026-01-05',
          row_count: 2,
          source_path: 'runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv',
          retention: {
            code: 'missing_final_retention_decision',
            status: 'blocked',
            reason: 'summary lacks final_model_retention_decision',
            summary_path: 'runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json',
            final_model_retained: false,
          },
        },
      },
      diagnostic_profiles: ['balanced_rule_residual_v22'],
      retention_filter: {
        policy: 'final_model_retained_or_reference',
        input_profile_count: 2,
        allowed_profile_count: 1,
        blocked_profile_count: 1,
      },
      blocked_profiles: {
        balanced_rule_residual_v22: {
          code: 'missing_final_retention_decision',
          status: 'blocked',
          reason: 'summary lacks final_model_retention_decision',
          summary_path: 'runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json',
          final_model_retained: false,
        },
      },
    });

    expect(panel.querySelector('.gate-banner')?.textContent).toContain('Diagnostic actor comparisons');
    expect(panel.querySelector('.gate-banner')?.textContent).toContain('not final-retained');
  });

  test('renders strict walk-forward retention gate status', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, {
      ...manifestFixture,
      strict_retention_gate: {
        status: 'blocked',
        source_path: 'runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json',
        profiles_scanned: 12,
        passing_profile_count: 0,
        passing_profiles: [],
        reference_profile: 'soxl_backtest_policy_reimplementation_v1',
        reference_return_pct: 132.077978,
        current_best_return_pct: 224.361539,
        best_return_ratio: 0.95,
        required_best_return_pct: 213.143462,
        max_walkforward_drawdown_pct: 20,
        required_wf_windows: ['wf_2020', 'wf_2022', 'wf_2023_2024'],
        closest_wf_drawdown_candidate: {
          profile: 'mode_conditioned_rebound_reward_train_v2',
          worst_wf_max_drawdown_pct: 21.83172,
          wf_drawdown_gap_pct: 1.83172,
          missing_wf_window_count: 0,
          ytd_return_pct: 177.573569,
          best_return_gap_pct: 35.569893,
        },
        closest_ytd_return_candidate: {
          profile: 'smoke',
          ytd_return_pct: 224.361539,
          best_return_gap_pct: 0,
          missing_wf_window_count: 3,
        },
      },
    });

    expect(panel.textContent).toContain('Strict Gate');
    expect(panel.textContent).toContain('0/12');
    expect(panel.textContent).toContain('MaxDD < 20.00%');
    expect(panel.textContent).toContain('Mode rebound v2');
    expect(panel.textContent).toContain('21.83%');
  });

  test('renders inspector rows with tone classes', () => {
    const panel = document.createElement('aside');

    renderInspectorPanel(panel, [
      { label: 'SOXL close', value: '11.00' },
      { label: 'smoke action', value: 'BUY', tone: 'buy' },
    ]);

    expect(panel.querySelector('[data-tone="buy"]')?.textContent).toContain('BUY');
  });

  test('renders comparison panel with model deltas and action mismatch counts', () => {
    const panel = document.createElement('section');

    renderComparisonPanel(panel, comparisonFixture, '2026-01-02', '2026-01-06');

    expect(panel.querySelector('.comparison-cards')).not.toBeNull();
    expect(panel.querySelector('.table-scroll .comparison-table')).toBeNull();
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('Rebased NAV delta');
    expect(panel.textContent).toContain('Mismatch Days');
    expect(panel.textContent).toContain('1');
  });

  test('renders compact return and drawdown rankings for model differences', () => {
    const panel = document.createElement('section');

    renderComparisonPanel(
      panel,
      {
        ...comparisonFixture,
        rankings: [
          { profile: 'smoke', return: 1.25, max_drawdown: -0.18, avg_position: 0.42, row_count: 111 },
          {
            profile: 'balanced_rule_residual_v22',
            return: 0.92,
            max_drawdown: -0.13,
            avg_position: 0.18,
            row_count: 111,
          },
        ],
      },
      '2026-01-02',
      '2026-01-06',
    );

    expect(panel.querySelector('.ranking-strip')).not.toBeNull();
    expect(panel.textContent).toContain('Ret 125.00%');
    expect(panel.textContent).toContain('MaxDD -18.00%');
    expect(panel.textContent).toContain('AvgPos 0.420');
  });

  test('renders compact model delta strip outside the collapsed comparison panel', () => {
    const panel = document.createElement('div');

    renderModelDeltaStrip(panel, comparisonFixture, '2026-01-02', '2026-01-06', undefined, {
      ...manifestFixture,
      profile_summaries: {
        smoke: {
          status: 'canonical',
          replay_start: '2026-01-02',
          replay_end: '2026-01-06',
          row_count: 3,
          source_path: 'runtime/model/card/smoke/smoke_soxl_daily.csv',
        },
        balanced_rule_residual_v22: {
          status: 'canonical',
          replay_start: '2026-01-02',
          replay_end: '2026-01-05',
          row_count: 2,
          source_path: 'runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv',
        },
      },
    });

    expect(panel.querySelectorAll('.model-delta-pill')).toHaveLength(1);
    expect(panel.textContent).toContain('smoke');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Baseline BRR v22');
    expect(panel.textContent).toContain('Compare through 2026-01-06');
    expect(panel.textContent).toContain('End 2026-01-06');
    expect(panel.textContent).toContain('Ret +3.00%');
    expect(panel.textContent).toContain('NAV +2.00');
    expect(panel.textContent).toContain('Pos +0.200');
    expect(panel.textContent).toContain('Mismatch 2');
  });

  test('renders model differences as visible rows with latest and common-window dates', () => {
    const panel = document.createElement('div');

    renderModelDeltaStrip(
      panel,
      {
        ...comparisonFixture,
        rankings: [
          { profile: 'smoke', return: 0.42, max_drawdown: -0.08, avg_position: 0.51, row_count: 3 },
          {
            profile: 'balanced_rule_residual_v22',
            return: 0.12,
            max_drawdown: -0.04,
            avg_position: 0.19,
            row_count: 3,
          },
        ],
      },
      '2026-01-02',
      '2026-01-07',
      undefined,
      {
        ...manifestFixture,
        latest_market_date: '2026-01-07',
        default_baseline_profile: 'balanced_rule_residual_v22',
        profile_summaries: {
          smoke: {
            status: 'canonical',
            replay_start: '2026-01-02',
            replay_end: '2026-01-06',
            row_count: 3,
            source_path: 'runtime/model/card/smoke/smoke_soxl_daily.csv',
          },
          balanced_rule_residual_v22: {
            status: 'reference',
            replay_start: '2026-01-02',
            replay_end: '2026-01-07',
            row_count: 4,
            source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
          },
        },
      },
    );

    expect(panel.querySelector('.model-diff-grid')).not.toBeNull();
    expect(panel.querySelectorAll('.model-difference-row')).toHaveLength(1);
    expect(panel.querySelector('[data-freshness="stale"]')?.textContent).toContain('Cut 2026-01-06');
    expect(panel.querySelector('[data-freshness="stale"]')?.textContent).toContain('Gap to price 2026-01-07');
    expect(panel.textContent).toContain('Ret 42.00%');
    expect(panel.textContent).toContain('MaxDD -8.00%');
    expect(panel.textContent).toContain('AvgPos 0.510');
    expect(panel.textContent).toContain('ΔNAV +2.00');
    expect(panel.textContent).toContain('ΔPos +0.200');
  });

  test('renders latest baseline and market date when they extend past actor comparison rows', () => {
    const panel = document.createElement('div');

    renderModelDeltaStrip(panel, comparisonFixture, '2026-01-02', '2026-01-07', undefined, {
      ...manifestFixture,
      latest_market_date: '2026-01-07',
      default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
      profile_summaries: {
        soxl_backtest_policy_reimplementation_v1: {
          status: 'reference',
          replay_start: '2026-01-02',
          replay_end: '2026-01-07',
          row_count: 4,
          source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
        },
        smoke: {
          status: 'canonical',
          replay_start: '2026-01-02',
          replay_end: '2026-01-06',
          row_count: 3,
          source_path: 'runtime/model/card/smoke/smoke_soxl_daily.csv',
        },
        balanced_rule_residual_v22: {
          status: 'canonical',
          replay_start: '2026-01-02',
          replay_end: '2026-01-07',
          row_count: 4,
          source_path: 'runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv',
        },
      },
    });

    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Compare through 2026-01-06');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Price 2026-01-07');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Baseline BRR v22 2026-01-07');
  });

  test('prioritizes actor model difference rows over the reference baseline', () => {
    const panel = document.createElement('div');

    renderModelDeltaStrip(panel, comparisonFixture, '2026-01-02', '2026-01-07', undefined, {
      ...manifestFixture,
      latest_market_date: '2026-01-07',
      safe_backtest_end: '2026-01-06',
      default_baseline_profile: 'balanced_rule_residual_v22',
      profile_summaries: {
        smoke: {
          status: 'canonical',
          replay_start: '2026-01-02',
          replay_end: '2026-01-06',
          row_count: 3,
          source_path: 'runtime/model/card/smoke/smoke_soxl_daily.csv',
        },
        balanced_rule_residual_v22: {
          status: 'reference',
          replay_start: '2026-01-02',
          replay_end: '2026-01-07',
          row_count: 4,
          source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
        },
      },
    });

    expect(panel.querySelectorAll('.model-delta-pill')).toHaveLength(1);
    expect(panel.querySelector('.model-difference-name')?.textContent).toBe('smoke');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Baseline BRR v22 2026-01-07');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Price 2026-01-07');
    expect(panel.querySelector('[data-freshness="stale"]')?.textContent).toContain('Cut 2026-01-06');
  });

  test('renders clickable action mismatch rows with feature context', () => {
    const panel = document.createElement('section');
    const clicked: string[] = [];

    renderComparisonPanel(
      panel,
      comparisonFixture,
      '2026-01-02',
      '2026-01-06',
      [
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
          features: { rsi6: '60', vix: '18' },
        },
      ],
      (date) => clicked.push(date),
    );

    expect(panel.textContent).toContain('Mismatch Days');
    const diagnostics = panel.querySelector<HTMLDetailsElement>('details.comparison-diagnostics');
    expect(diagnostics).not.toBeNull();
    expect(diagnostics?.open).toBe(false);
    expect(diagnostics?.querySelector('summary')?.textContent).toContain('Action mismatches');
    expect(panel.textContent).toContain('BUY vs HOLD');
    expect(panel.textContent).toContain('rsi6 60');
    panel.querySelector<HTMLButtonElement>('.mismatch-date-button')?.click();
    expect(clicked).toEqual(['2026-01-05']);
  });

  test('caps action mismatch rows to a compact recent list', () => {
    const panel = document.createElement('section');
    const mismatchRows = Array.from({ length: 20 }, (_, index) => ({
      date: `2026-01-${String(index + 1).padStart(2, '0')}`,
      profile: 'smoke',
      model_action: 'BUY' as const,
      baseline_action: 'HOLD' as const,
      model_position: 0.7,
      baseline_position: 0,
      position_delta_vs_baseline: 0.7,
      nav_delta_vs_baseline: index,
      close: 10 + index,
      features: { rsi6: String(50 + index) },
    }));

    renderComparisonPanel(panel, comparisonFixture, '2026-01-02', '2026-01-06', mismatchRows);

    expect(panel.querySelectorAll('.mismatch-row')).toHaveLength(12);
    expect(panel.textContent).toContain('2026-01-20');
    expect(panel.textContent).not.toContain('2026-01-01');
  });

  test('renders visible buy and sell action labels', () => {
    const panel = document.createElement('div');

    renderActionLegend(panel);

    expect(panel.textContent).toContain('BUY');
    expect(panel.textContent).toContain('SELL');
  });
});
