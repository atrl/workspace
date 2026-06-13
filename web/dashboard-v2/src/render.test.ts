import { describe, expect, test } from 'vitest';
import {
  renderActionLegend,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderFeatureComparisonPanel,
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
    expect(panel.textContent).toContain('Auto Update');
    expect(panel.textContent).toContain('dashboard_refresh');
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
  });

  test('renders first-screen operations status with replay counts and tasks', () => {
    const panel = document.createElement('section');

    renderOperationsPanel(panel, manifestFixture, [
      'Replay refresh found 0 runnable model profiles; dashboard is using retained exports/reference rows only.',
    ]);

    expect(panel.textContent).toContain('Operations');
    expect(panel.textContent).toContain('replay ready');
    expect(panel.textContent).toContain('1 / 1');
    expect(panel.textContent).toContain('current=1');
    expect(panel.textContent).toContain('dashboard_refresh');
    expect(panel.textContent).toContain('production/run_dashboard_refresh.sh');
    expect(panel.querySelector('.task-card[data-enabled="true"]')?.textContent).toContain('configured');
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
    expect(panel.textContent).toContain('alpha_signal');
    expect(panel.textContent).toContain('1.75');
    expect(panel.textContent).toContain('beta_signal');
    expect(panel.textContent).toContain('-2.50');
    expect(panel.textContent).toContain('Recent Buy/Sell Events');
  });

  test('renders compact first-screen buy and sell feature spotlight', () => {
    const panel = document.createElement('section');

    renderFeatureSpotlightPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
    });

    expect(panel.textContent).toContain('Buy/Sell Features');
    expect(panel.textContent).toContain('2026-01-05');
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('alpha_signal');
    expect(panel.textContent).toContain('beta_signal');
    expect(panel.textContent).toContain('BUY-SELL');
    expect(panel.textContent).not.toContain('Recent Buy/Sell Events');
  });
});
