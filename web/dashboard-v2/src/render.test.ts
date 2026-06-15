import { describe, expect, test } from 'vitest';
import {
  renderActionLegend,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderInspectorPanel,
  renderModelDeltaStrip,
} from './render';
import { comparisonFixture, manifestFixture, modelFixture } from './testFixtures';

describe('dashboard v2 DOM render helpers', () => {
  test('renders data source provenance and safe replay window', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, manifestFixture);

    expect(panel.querySelector('.status-strip')).not.toBeNull();
    expect(panel.textContent).toContain('Qlib safe replay');
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
          latest_signal_date: '2026-01-06',
        },
      },
    });

    expect(panel.querySelector('.freshness-alert')?.textContent).toContain('Market 2026-01-06');
    expect(panel.querySelector('.freshness-alert')?.textContent).toContain('Actors 2026-01-05');
    expect(panel.querySelector('.freshness-alert')?.textContent).toContain('Baseline 2026-01-06');
    expect(panel.querySelector('.status-strip')?.textContent).toContain('Baseline latest');
    expect(panel.querySelector('.status-strip')?.textContent).toContain('2026-01-06');
    expect(panel.querySelector('.status-strip')?.textContent).toContain('Actor replay');
    expect(panel.querySelector('.status-strip')?.textContent).toContain('2026-01-05');
    expect(panel.querySelector('.status-strip')?.textContent).toContain('Actor latest signal');
    expect(panel.querySelector('.status-strip')?.textContent).toContain('2026-01-06');
    expect(panel.querySelector('.freshness-alert')?.textContent).toContain('Actor signals 2026-01-06');
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

  test('renders strict candidate retention filter status and next mechanism', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, {
      ...manifestFixture,
      strict_candidate_retention_filter: {
        status: 'blocked',
        source_path: 'runtime/model_selection/card/strict_candidate_retention_filter_v1/summary.json',
        decision: 'strict_candidate_retention_filter_no_promotable_candidates',
        reference_baseline: 'soxl_backtest_policy_reimplementation_v1',
        passing_profile_count: 0,
        promotable_profile_count: 0,
        actor_retention_allowed_count: 0,
        reference_visible_count: 2,
        ppo_allowed_next: false,
        next_mechanism: 'train_bull_trend_participation_with_drawdown_drift_stop_no_promotion',
        row_count: 15,
        candidate_type_counts: {
          checkpoint_selector: 1,
          profile: 12,
          qlib_reference_screen: 1,
          reference_baseline: 1,
        },
      },
    } as any);

    expect(panel.textContent).toContain('Final Retain');
    expect(panel.textContent).toContain('0');
    expect(panel.textContent).toContain('Candidate retention filter: 0 promotable, 0 actor-retained');
    expect(panel.textContent).toContain('train_bull_trend_participation_with_drawdown_drift_stop_no_promotion');
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
    const latestSmoke = {
      profile: 'smoke',
      status: 'canonical',
      source_path: 'runtime/model/card/smoke/smoke_soxl_daily.csv',
      replay_start: '2026-01-02',
      replay_end: '2026-01-06',
      row_count: 3,
      rows: [],
      events: [],
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.55,
        action: 'BUY' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };

    renderModelDeltaStrip(
      panel,
      comparisonFixture,
      '2026-01-02',
      '2026-01-06',
      undefined,
      {
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
      },
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
          features: { rsi6: '60', vix: '18', spread: '0.4' },
        },
      ],
      { smoke: latestSmoke },
    );

    expect(panel.querySelector('.model-diff-table')).not.toBeNull();
    expect(panel.querySelector<HTMLDetailsElement>('details.model-diff-history')?.open).toBe(false);
    expect(panel.querySelectorAll('.model-difference-row')).toHaveLength(1);
    expect(panel.textContent).toContain('smoke');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Baseline BRR v22');
    expect(panel.textContent).toContain('Common comparison 2026-01-06');
    expect(panel.textContent).toContain('Signal BUY 0.550 @01-07');
    expect(panel.textContent).toContain('Ret +3.00%');
    expect(panel.textContent).toContain('+2.00');
    expect(panel.textContent).toContain('+0.200');
    expect(panel.textContent).toContain('2');
    expect(panel.textContent).toContain('Last diff 2026-01-05 BUY vs HOLD');
    expect(panel.textContent).toContain('rsi6 60');
  });

  test('renders latest signal differences against the latest baseline row', () => {
    const panel = document.createElement('div');
    const referenceProfile = 'soxl_backtest_policy_reimplementation_v1';
    const referenceModel = {
      profile: referenceProfile,
      status: 'reference',
      source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
      replay_start: '2026-01-02',
      replay_end: '2026-01-07',
      row_count: 4,
      rows: [
        ...modelFixture.rows,
        {
          date: '2026-01-07',
          strategy_nav: 105,
          buyhold_nav: 110,
          position: 0.2,
          action: 'HOLD' as const,
          price: 10.4,
          excess: -5,
          excess_pct: 0.05,
        },
      ],
      events: [],
    };
    const signalModel = {
      ...modelFixture,
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.55,
        action: 'BUY' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };
    const referenceComparison = {
      ...comparisonFixture,
      baseline_profile: referenceProfile,
      compared_profiles: [referenceProfile, 'smoke'],
      rows: comparisonFixture.rows.map((row) => ({
        ...row,
        profiles: {
          [referenceProfile]: row.profiles.balanced_rule_residual_v22,
          smoke: row.profiles.smoke,
        },
      })),
    };

    renderModelDeltaStrip(
      panel,
      referenceComparison,
      '2026-01-02',
      '2026-01-07',
      undefined,
      {
        ...manifestFixture,
        latest_market_date: '2026-01-07',
        default_baseline_profile: referenceProfile,
        profile_summaries: {
          [referenceProfile]: {
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
        },
      },
      [],
      { [referenceProfile]: referenceModel, smoke: signalModel },
    );

    expect(panel.querySelector('.latest-signal-strip')).toBeNull();
    const matrix = panel.querySelector('.latest-decision-matrix');
    expect(matrix).not.toBeNull();
    expect(matrix?.textContent).toContain('Latest 2026-01-07');
    expect(matrix?.textContent).toContain('baseline SOXLBacktest HOLD 0.200');
    expect(matrix?.textContent).toContain('smoke');
    expect(matrix?.textContent).toContain('BUY 0.550');
    expect(matrix?.textContent).toContain('Δpos +0.350');
    expect(matrix?.textContent).toContain('signal as-of 2026-01-06');
  });

  test('renders a latest decision matrix for baseline versus actor signals', () => {
    const panel = document.createElement('div');
    const referenceProfile = 'soxl_backtest_policy_reimplementation_v1';
    const referenceModel = {
      profile: referenceProfile,
      status: 'reference',
      source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
      replay_start: '2026-01-02',
      replay_end: '2026-01-07',
      row_count: 4,
      rows: [
        ...modelFixture.rows,
        {
          date: '2026-01-07',
          strategy_nav: 105,
          buyhold_nav: 110,
          position: 0.2,
          action: 'HOLD' as const,
          price: 10.4,
          excess: -5,
          excess_pct: 0.05,
        },
      ],
      events: [],
    };
    const buySignalModel = {
      ...modelFixture,
      profile: 'balanced_rule_residual_v22',
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.625,
        action: 'BUY' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };
    const holdSignalModel = {
      ...modelFixture,
      profile: 'smoke',
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.2,
        action: 'HOLD' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };
    const referenceComparison = {
      ...comparisonFixture,
      baseline_profile: referenceProfile,
      compared_profiles: [referenceProfile, 'balanced_rule_residual_v22', 'smoke'],
      rows: comparisonFixture.rows.map((row) => ({
        ...row,
        profiles: {
          [referenceProfile]: row.profiles.balanced_rule_residual_v22,
          balanced_rule_residual_v22: row.profiles.smoke,
          smoke: row.profiles.smoke,
        },
      })),
    };

    renderModelDeltaStrip(
      panel,
      referenceComparison,
      '2026-01-02',
      '2026-01-07',
      undefined,
      {
        ...manifestFixture,
        latest_market_date: '2026-01-07',
        default_baseline_profile: referenceProfile,
        profile_summaries: {
          [referenceProfile]: {
            status: 'reference',
            replay_start: '2026-01-02',
            replay_end: '2026-01-07',
            row_count: 4,
            source_path: 'runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv',
          },
          balanced_rule_residual_v22: {
            status: 'canonical',
            replay_start: '2026-01-02',
            replay_end: '2026-01-06',
            row_count: 3,
            source_path: 'runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv',
            latest_signal_date: '2026-01-07',
          },
          smoke: {
            status: 'canonical',
            replay_start: '2026-01-02',
            replay_end: '2026-01-06',
            row_count: 3,
            source_path: 'runtime/model/card/smoke/smoke_soxl_daily.csv',
            latest_signal_date: '2026-01-07',
          },
        },
      },
      [],
      {
        [referenceProfile]: referenceModel,
        balanced_rule_residual_v22: buySignalModel,
        smoke: holdSignalModel,
      },
    );

    const matrix = panel.querySelector('.latest-decision-matrix');
    expect(matrix).not.toBeNull();
    expect(matrix?.textContent).toContain('Latest model signals');
    expect(matrix?.textContent).toContain('Latest 2026-01-07');
    expect(matrix?.textContent).toContain('baseline SOXLBacktest HOLD 0.200');
    expect(matrix?.textContent).toContain('BRR v22');
    expect(matrix?.textContent).toContain('BUY 0.625');
    expect(matrix?.textContent).toContain('Δpos +0.425');
    expect(matrix?.textContent).toContain('signal as-of 2026-01-06');
    expect(matrix?.querySelector('[data-diff="true"]')?.textContent).toContain('BRR v22');
    expect(matrix?.textContent).toContain('smoke');
    expect(matrix?.textContent).toContain('HOLD 0.200');
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

    expect(panel.querySelector('.model-diff-table')).not.toBeNull();
    expect(panel.querySelectorAll('.model-difference-row')).toHaveLength(1);
    expect(panel.querySelector('[data-freshness="stale"]')?.textContent).toContain('2026-01-06 cut');
    expect(panel.querySelector('[data-freshness="stale"]')?.textContent).toContain('Price gap 2026-01-07');
    expect(panel.textContent).toContain('Ret 42.00%');
    expect(panel.textContent).toContain('-8.00%');
    expect(panel.textContent).toContain('0.510');
    expect(panel.textContent).toContain('+2.00');
    expect(panel.textContent).toContain('+0.200');
  });

  test('renders latest baseline and market date when they extend past actor comparison rows', () => {
    const panel = document.createElement('div');
    const referenceComparison = {
      ...comparisonFixture,
      baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
      compared_profiles: ['soxl_backtest_policy_reimplementation_v1', 'smoke'],
      rows: comparisonFixture.rows.map((row) => ({
        ...row,
        profiles: {
          soxl_backtest_policy_reimplementation_v1: row.profiles.balanced_rule_residual_v22,
          smoke: row.profiles.smoke,
        },
      })),
    };

    renderModelDeltaStrip(panel, referenceComparison, '2026-01-02', '2026-01-07', undefined, {
      ...manifestFixture,
      latest_market_date: '2026-01-07',
      safe_backtest_end: '2026-01-06',
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
      },
    });

    expect(panel.querySelector('.model-delta-summary')?.textContent).toContain('2 models');
    expect(panel.querySelector('.model-delta-summary')?.textContent).toContain('3 common rows');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Common comparison 2026-01-06');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('SOXL K-line 2026-01-07');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Baseline SOXLBacktest latest 2026-01-07');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Qlib actors safe replay 2026-01-06');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('safe end 2026-01-06');
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

    expect(panel.querySelectorAll('.model-difference-row')).toHaveLength(1);
    expect(panel.querySelector('.model-difference-name')?.textContent).toBe('smoke');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('Baseline BRR v22 latest 2026-01-07');
    expect(panel.querySelector('.model-delta-caption')?.textContent).toContain('SOXL K-line 2026-01-07');
    expect(panel.querySelector('[data-freshness="stale"]')?.textContent).toContain('2026-01-06 cut');
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
    expect(panel.textContent).toContain('DIFF');
  });
});
