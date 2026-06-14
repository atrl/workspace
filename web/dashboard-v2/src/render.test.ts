import { describe, expect, test } from 'vitest';
import {
  renderActionLegend,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderFeatureComparisonPanel,
  renderFeatureDateSnapshotPanel,
  renderFeatureSpotlightPanel,
  renderInspectorPanel,
  renderOperationsPanel,
} from './render';
import { betaModelFixture, comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';

describe('dashboard v2 DOM render helpers', () => {
  test('renders data source provenance and safe replay window', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, {
      ...manifestFixture,
      model_retention_gate: {
        ...manifestFixture.model_retention_gate!,
        registry_keep_actor_count: 3,
        registry_keep_actor_strict_violation_count: 2,
      },
      auto_update: {
        ...manifestFixture.auto_update!,
        tasks: [
          {
            ...manifestFixture.auto_update!.tasks[0],
            installed: true,
            loaded: true,
            log_path: 'runtime/soxl/paper_trading/dashboard_refresh.log',
            command_path: 'production/run_dashboard_refresh.sh',
          } as never,
        ],
      },
    });

    expect(panel.textContent).toContain('safe_backtest_end');
    expect(panel.textContent).toContain('2026-01-06');
    expect(panel.textContent).toContain('runnable replay profiles');
    expect(panel.textContent).toContain('1 / 1');
    expect(panel.textContent).toContain('current=1');
    expect(panel.textContent).toContain('model_replay_freshness.json');
    expect(panel.textContent).toContain('runnable profiles');
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('reference profiles');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('evidence-only profiles');
    expect(panel.textContent).toContain('candidate');
    expect(panel.textContent).toContain('unavailable profiles');
    expect(panel.textContent).toContain('Unavailable Model Replays');
    expect(panel.textContent).toContain('missing_actor_model');
    expect(panel.textContent).toContain('missing_actor_artifact');
    expect(panel.textContent).toContain('Auto Update');
    expect(panel.textContent).toContain('dashboard_refresh');
    expect(panel.textContent).toContain('manual refresh');
    expect(panel.textContent).toContain('scripts/refresh_dashboard_model_replays.py -> scripts/build_dashboard_bundle.py');
    expect(panel.textContent).toContain('installed');
    expect(panel.textContent).toContain('loaded');
    expect(panel.textContent).toContain('runtime/soxl/paper_trading/dashboard_refresh.log');
    expect(panel.textContent).toContain('production/run_dashboard_refresh.sh');
    expect(panel.textContent).toContain('06:30 Tue-Sat Asia/Shanghai');
    expect(panel.textContent).toContain('Retention Gate');
    expect(panel.textContent).toContain('0 / 2 promotable');
    expect(panel.textContent).toContain('MaxDD < 20.00%');
    expect(panel.textContent).toContain('keep_actor strict violations');
    expect(panel.textContent).toContain('2 / 3 blocked');
    expect(panel.textContent).toContain('Training Frontier');
    expect(panel.textContent).toContain('cap10_metric_replay_handoff_audit_v1');
    expect(panel.textContent).toContain('inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training');
    expect(panel.textContent).toContain('walkforward_relative_return_gap');
  });

  test('renders first-screen operations status with replay counts and tasks', () => {
    const panel = document.createElement('section');

    renderOperationsPanel(panel, manifestFixture, [
      'Replay refresh found 0 runnable model profiles; dashboard is using retained exports/reference rows only.',
    ]);

    expect(panel.textContent).toContain('Operations');
    expect(panel.textContent).toContain('replay ready');
    expect(panel.textContent).toContain('1 / 1');
    expect(panel.textContent).toContain('runnable inference');
    expect(panel.textContent).toContain('1 / 3');
    expect(panel.textContent).toContain('current=1');
    expect(panel.textContent).toContain('dashboard_refresh');
    expect(panel.textContent).toContain('com.soxl.dashboard.refresh');
    expect(panel.textContent).toContain('installed');
    expect(panel.textContent).toContain('production/run_dashboard_refresh.sh');
    expect(panel.textContent).toContain('runtime/soxl/paper_trading/dashboard_refresh.log');
    expect(panel.textContent).toContain('pipeline');
    expect(panel.textContent).toContain('scripts/check_dashboard_static_export.py');
    expect(panel.textContent).toContain('Unavailable Model Replays');
    expect(panel.textContent).toContain('missing_actor_model');
    expect(panel.querySelector('.task-card[data-enabled="true"]')?.textContent).toContain('configured');
    expect(panel.textContent).toContain('next training');
    expect(panel.textContent).toContain('cap10_metric_replay_handoff_audit_v1');
    expect(panel.textContent).toContain('walkforward_relative_return_gap');
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

    renderComparisonPanel(panel, comparisonFixture, '2026-01-02', '2026-01-06', undefined, {
      smoke: { ...modelFixture, retention_decision: 'keep_reference_baseline' },
      balanced_rule_residual_v22: betaModelFixture,
    });

    expect(panel.querySelector('.table-scroll .comparison-table')).not.toBeNull();
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('Action mismatches');
    expect(panel.textContent).toContain('BuyHold');
    expect(panel.textContent).toContain('Vs BuyHold');
    expect(panel.textContent).toContain('Retention');
    expect(panel.textContent).toContain('baseline');
    expect(panel.textContent).toContain('evidence-only');
    expect(panel.textContent).toContain('wf_max_drawdown_ge_20:wf_2022');
    expect(panel.textContent).toContain('-5.00pp');
    expect(panel.textContent).toContain('1');
  });

  test('renders visible buy and sell action labels', () => {
    const panel = document.createElement('div');

    renderActionLegend(panel);

    expect(panel.textContent).toContain('BUY');
    expect(panel.textContent).toContain('SELL');
  });

  test('renders model-specific buy and sell feature comparison rows', () => {
    const panel = document.createElement('section');

    renderFeatureComparisonPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
      baselineProfile: 'balanced_rule_residual_v22',
    });

    expect(panel.textContent).toContain('Buy/Sell Feature Comparison');
    expect(panel.textContent).toContain('Model Feature Schemas');
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('alpha123');
    expect(panel.textContent).toContain('beta456');
    expect(panel.textContent).toContain('alpha_signal 1.25');
    expect(panel.textContent).toContain('beta_signal -2.50');
    expect(panel.textContent).toContain('Buy/Sell Feature Summary');
    expect(panel.textContent).toContain('Trade Feature vs Baseline');
    expect(panel.textContent).toContain('baseline SELL');
    expect(panel.textContent).toContain('-10.00');
    expect(panel.textContent).toContain('alpha_signal');
    expect(panel.textContent).toContain('1.75');
    expect(panel.textContent).toContain('beta_signal');
    expect(panel.textContent).toContain('-2.50');
    expect(panel.textContent).toContain('Recent Buy/Sell Events');
  });

  test('renders selected-date feature snapshot as a lightweight side panel', () => {
    const panel = document.createElement('section');

    renderFeatureDateSnapshotPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
    });

    expect(panel.textContent).toContain('Selected Date Features');
    expect(panel.textContent).toContain('2026-01-05');
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('alpha_signal 1.25');
    expect(panel.textContent).toContain('beta_signal -2.50');
    expect(panel.textContent).not.toContain('Buy/Sell Feature Summary');
    expect(panel.textContent).not.toContain('Recent Buy/Sell Events');
  });

  test('renders complete first-screen buy and sell feature spotlight', () => {
    const panel = document.createElement('section');

    renderFeatureSpotlightPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
      baselineProfile: 'balanced_rule_residual_v22',
    });

    expect(panel.textContent).toContain('Buy/Sell Features');
    expect(panel.textContent).toContain('2026-01-05');
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('Model Feature Schemas');
    expect(panel.textContent).toContain('alpha123');
    expect(panel.textContent).toContain('beta456');
    expect(panel.textContent).toContain('alpha_signal');
    expect(panel.textContent).toContain('beta_signal');
    expect(panel.textContent).toContain('Trade Feature vs Baseline');
    expect(panel.textContent).toContain('baseline SELL');
    expect(panel.textContent).toContain('BUY-SELL');
    expect(panel.textContent).toContain('Recent Buy/Sell Events');
    expect(panel.querySelector('h3')?.textContent).toBe('Buy/Sell Feature Summary');
  });

  test('limits hover feature snapshots so mouse interaction stays lightweight', () => {
    const panel = document.createElement('section');
    const features = Object.fromEntries(Array.from({ length: 20 }, (_, index) => [`feature_${index}`, index]));
    const model = {
      ...modelFixture,
      feature_columns: Object.keys(features),
      snapshot_columns: Object.keys(features),
      rows: [
        {
          ...modelFixture.rows[1],
          feature_snapshot: features,
        },
      ],
      events: [],
    };

    renderFeatureSpotlightPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: model },
      baselineProfile: 'smoke',
    });

    const snapshotCell = panel.querySelector('.feature-snapshot-cell');
    expect(snapshotCell?.textContent).toContain('feature_0 0.00');
    expect(snapshotCell?.textContent).toContain('+8 more');
    expect(snapshotCell?.textContent).not.toContain('feature_19');
  });
});
