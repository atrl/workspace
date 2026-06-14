import { describe, expect, test } from 'vitest';
import {
  renderActionLegend,
  renderChartCommandBar,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderFeatureComparisonPanel,
  renderFeatureDateSnapshotPanel,
  renderFeatureSpotlightPanel,
  renderInspectorPanel,
  renderOperationsPanel,
  renderDashboardStatusStrip,
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
    expect(panel.textContent).toContain('replay candidate note');
    expect(panel.textContent).toContain('strict retention gate');
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
    expect(panel.textContent).toContain('install auto update');
    expect(panel.textContent).toContain('production/install_dashboard_refresh_launchagent.sh');
    expect(panel.textContent).toContain('--run-now');
    expect(panel.textContent).toContain('scripts/refresh_dashboard_model_replays.py -> scripts/build_dashboard_bundle.py');
    expect(panel.textContent).toContain('installed');
    expect(panel.textContent).toContain('loaded');
    expect(panel.textContent).toContain('runtime/soxl/paper_trading/dashboard_refresh.log');
    expect(panel.textContent).toContain('production/run_dashboard_refresh.sh');
    expect(panel.textContent).toContain('06:30 Tue-Sat Asia/Shanghai');
    expect(panel.textContent).toContain('Model Runtime Status');
    expect(panel.textContent).toContain('missing_actor_model');
    expect(panel.textContent).toContain('unavailable');
    expect(panel.textContent).toContain('missing_actor_artifact');
    expect(panel.textContent).toContain('alpha123 · 2 features');
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
    expect(panel.textContent).toContain('replay candidate note');
    expect(panel.textContent).toContain('strict retention gate');
    expect(panel.textContent).toContain('dashboard_refresh');
    expect(panel.textContent).toContain('com.soxl.dashboard.refresh');
    expect(panel.textContent).toContain('installed');
    expect(panel.textContent).toContain('production/run_dashboard_refresh.sh');
    expect(panel.textContent).toContain('production/install_dashboard_refresh_launchagent.sh');
    expect(panel.textContent).toContain('runtime/soxl/paper_trading/dashboard_refresh.log');
    expect(panel.textContent).toContain('pipeline');
    expect(panel.textContent).toContain('scripts/check_dashboard_static_export.py');
    expect(panel.textContent).toContain('Model Runtime Status');
    expect(panel.textContent).toContain('runnable');
    expect(panel.textContent).toContain('reference');
    expect(panel.textContent).toContain('missing_actor_model');
    expect(panel.textContent).toContain('missing_actor_artifact');
    expect(panel.textContent).toContain('Unavailable Model Replays');
    expect(panel.textContent).toContain('missing_actor_model');
    expect(panel.querySelector('.task-card[data-enabled="true"]')?.textContent).toContain('configured');
    expect(panel.textContent).toContain('next training');
    expect(panel.textContent).toContain('cap10_metric_replay_handoff_audit_v1');
    expect(panel.textContent).toContain('walkforward_relative_return_gap');
  });

  test('puts auto update task cards before dense operations metadata', () => {
    const panel = document.createElement('section');

    renderOperationsPanel(panel, {
      ...manifestFixture,
      auto_update: {
        ...manifestFixture.auto_update!,
        tasks: [
          {
            name: 'preopen',
            label: 'com.soxl.preopen.scheduler',
            enabled: true,
            configured: true,
            installed: true,
            loaded: true,
            schedule: '21:00 Mon-Fri Asia/Shanghai',
            command_path: 'production/run_soxl_scheduler.sh',
            log_path: 'runtime/soxl/paper_trading/scheduler_preopen.log',
          },
          {
            name: 'postclose',
            label: 'com.soxl.daily.scheduler',
            enabled: true,
            configured: true,
            installed: true,
            loaded: true,
            schedule: '06:15 Tue-Sat Asia/Shanghai',
            command_path: 'production/run_soxl_scheduler.sh',
            log_path: 'runtime/soxl/paper_trading/scheduler_postclose.log',
          },
          {
            name: 'dashboard_refresh',
            label: 'com.soxl.dashboard.refresh',
            enabled: true,
            configured: true,
            installed: true,
            loaded: true,
            schedule: '06:30 Tue-Sat Asia/Shanghai',
            command_path: 'production/run_dashboard_refresh.sh',
            log_path: 'runtime/soxl/paper_trading/dashboard_refresh.log',
          },
        ],
      },
    });

    const text = panel.textContent ?? '';
    expect(text).toContain('Auto Update Tasks');
    expect(text).toContain('3 local tasks');
    expect(text).toContain('preopen');
    expect(text).toContain('postclose');
    expect(text).toContain('dashboard_refresh');
    expect(text.indexOf('Auto Update Tasks')).toBeLessThan(text.indexOf('replay ready'));
    expect(panel.querySelectorAll('.task-card')).toHaveLength(3);
  });

  test('renders a first-screen dashboard status strip with auto update and runtime roles', () => {
    const panel = document.createElement('div');

    renderDashboardStatusStrip(panel, {
      ...manifestFixture,
      available_profiles: [
        'soxl_backtest_policy_reimplementation_v1',
        'state_conditioned_drawdown_rebound_reward_train_v1',
        'drawdown_rebound_reward_train_v1',
        'mode_conditioned_rebound_reward_train_v2',
        'balanced_rule_residual_v19_moe_v2',
      ],
      profile_groups: {
        runnable: [],
        reference: ['soxl_backtest_policy_reimplementation_v1'],
        evidence_only: [
          'state_conditioned_drawdown_rebound_reward_train_v1',
          'drawdown_rebound_reward_train_v1',
          'mode_conditioned_rebound_reward_train_v2',
          'balanced_rule_residual_v19_moe_v2',
        ],
      },
      auto_update: {
        ...manifestFixture.auto_update!,
        model_replay_freshness: {
          ...manifestFixture.auto_update!.model_replay_freshness!,
          ready_count: 0,
          profile_count: 0,
          safe_backtest_end: '2026-06-09',
        },
        tasks: [
          {
            name: 'preopen',
            label: 'com.soxl.preopen.scheduler',
            enabled: true,
            configured: true,
            installed: true,
            loaded: true,
            schedule: '21:00 Mon-Fri Asia/Shanghai',
          },
          {
            name: 'postclose',
            label: 'com.soxl.daily.scheduler',
            enabled: true,
            configured: true,
            installed: true,
            loaded: true,
            schedule: '06:15 Tue-Sat Asia/Shanghai',
          },
          {
            name: 'dashboard_refresh',
            label: 'com.soxl.dashboard.refresh',
            enabled: true,
            configured: true,
            installed: true,
            loaded: true,
            schedule: '06:30 Tue-Sat Asia/Shanghai',
          },
        ],
      },
    });

    expect(panel.textContent).toContain('Auto Update');
    expect(panel.textContent).toContain('3 / 3 enabled');
    expect(panel.textContent).toContain('3 / 3 loaded');
    expect(panel.textContent).toContain('0 / 0 replay candidates');
    expect(panel.textContent).toContain('preopen');
    expect(panel.textContent).toContain('postclose');
    expect(panel.textContent).toContain('dashboard_refresh');
    expect(panel.textContent).toContain('RL Runtime');
    expect(panel.textContent).toContain('0 / 5 runnable');
    expect(panel.textContent).toContain('SOXLBacktest baseline');
    expect(panel.textContent).toContain('Replay Window');
    expect(panel.textContent).toContain('safe 2026-06-09');
    expect(panel.textContent).toContain('Training Frontier');
    expect(panel.textContent).toContain('walkforward_relative_return_gap');
    expect(panel.textContent).toContain('preopen');
    expect(panel.textContent).toContain('postclose');
    expect(panel.textContent).toContain('dashboard_refresh');
    expect(panel.querySelectorAll('.status-task-chip')).toHaveLength(3);
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

  test('renders rerun comparison values from model rows when static comparison lacks the rerun profile', () => {
    const panel = document.createElement('section');
    const rerun = {
      ...modelFixture,
      profile: 'rerun:smoke:2026-01-02:2026-01-06',
      display_name: 'smoke rerun',
      rows: modelFixture.rows.map((row, index) =>
        index === modelFixture.rows.length - 1
          ? { ...row, strategy_nav: 112, buyhold_nav: 108, position: 0.4, action: 'BUY' as const }
          : row,
      ),
    };

    renderComparisonPanel(
      panel,
      comparisonFixture,
      '2026-01-02',
      '2026-01-06',
      ['rerun:smoke:2026-01-02:2026-01-06', 'smoke'],
      { 'rerun:smoke:2026-01-02:2026-01-06': rerun, smoke: modelFixture },
      'smoke',
    );

    expect(panel.textContent).toContain('smoke rerun');
    expect(panel.textContent).toContain('12.00%');
    expect(panel.textContent).toContain('8.00%');
    expect(panel.textContent).toContain('4.00pp');
    expect(panel.textContent).not.toContain('smoke rerunn/a');
  });

  test('renders visible action, price, and position labels', () => {
    const panel = document.createElement('div');

    renderActionLegend(panel);

    expect(panel.textContent).toContain('BUY');
    expect(panel.textContent).toContain('SELL');
    expect(panel.textContent).toContain('SOXL Price');
    expect(panel.textContent).toContain('Position');
  });

  test('renders chart command buttons with inspect and date-range click modes', () => {
    const panel = document.createElement('div');
    const commands: string[] = [];
    const modes: string[] = [];

    renderChartCommandBar(
      panel,
      (command) => commands.push(command),
      'set_start',
      (mode) => modes.push(mode),
    );

    expect(panel.querySelector('[aria-label="Inspect candles"]')?.getAttribute('data-active')).toBe('false');
    expect(panel.querySelector('[aria-label="Click chart to set Start"]')?.getAttribute('data-active')).toBe('true');
    expect(panel.querySelector('[aria-label="Click chart to set End"]')?.getAttribute('data-active')).toBe('false');
    expect(panel.querySelector('[aria-label="Zoom in"]')?.getAttribute('title')).toBe('Zoom in (+/=)');
    expect(panel.querySelector('[aria-label="Show last 60 bars"]')?.getAttribute('title')).toBe('Show last 60 bars (60)');
    expect(panel.querySelector('[aria-label="Show last 120 bars"]')?.getAttribute('title')).toBe('Show last 120 bars (120)');
    expect(panel.querySelector('[aria-label="Show last 250 bars"]')?.getAttribute('title')).toBe('Show last 250 bars (250)');
    expect(panel.querySelector('[aria-label="Follow latest"]')?.getAttribute('title')).toBe('Follow latest (End)');
    expect(panel.querySelector('[aria-label="Click chart to set Start"]')?.getAttribute('title')).toBe(
      'Click chart to set Start (S)',
    );

    (panel.querySelector('[aria-label="Click chart to set End"]') as HTMLButtonElement).click();
    (panel.querySelector('[aria-label="Show last 60 bars"]') as HTMLButtonElement).click();
    (panel.querySelector('[aria-label="Zoom in"]') as HTMLButtonElement).click();

    expect(modes).toEqual(['set_end']);
    expect(commands).toEqual(['last_60', 'zoom_in']);
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
    expect(panel.textContent).toContain('Feature Driver Matrix');
    expect(panel.textContent).toContain('Model Feature Schemas');
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('alpha123');
    expect(panel.textContent).toContain('beta456');
    expect(panel.textContent).toContain('Model-only');
    expect(panel.textContent).toContain('1: alpha_signal');
    expect(panel.textContent).toContain('1: beta_signal');
    expect(panel.textContent).toContain('1: shared_feature');
    expect(panel.textContent).toContain('alpha_signal 1.25');
    expect(panel.textContent).toContain('beta_signal -2.50');
    expect(panel.textContent).toContain('alpha123 · 2 cols');
    expect(panel.textContent).toContain('beta456 · 2 cols');
    expect(panel.textContent).toContain('Buy/Sell Feature Summary');
    expect(panel.textContent).toContain('Selected Decision Features');
    expect(panel.textContent).toContain('Model-only');
    expect(panel.textContent).toContain('Shared vs baseline');
    expect(panel.textContent).toContain('BUY/SELL drivers');
    expect(panel.textContent).toContain('Trade Feature vs Baseline');
    expect(panel.textContent).toContain('baseline SELL');
    expect(panel.textContent).toContain('-10.00');
    expect(panel.textContent).toContain('alpha_signal');
    expect(panel.textContent).toContain('1.75');
    expect(panel.textContent).toContain('beta_signal');
    expect(panel.textContent).toContain('-2.50');
    expect(panel.textContent).toContain('Recent Buy/Sell Events');
  });

  test('prioritizes buy/sell summary and baseline contrast before decision cards', () => {
    const panel = document.createElement('section');

    renderFeatureComparisonPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
      baselineProfile: 'balanced_rule_residual_v22',
    });

    const text = panel.textContent ?? '';
    expect(text.indexOf('Buy/Sell Feature Summary')).toBeLessThan(text.indexOf('Selected Decision Features'));
    expect(text.indexOf('Trade Feature vs Baseline')).toBeLessThan(text.indexOf('Selected Decision Features'));
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

  test('renders compact first-screen buy and sell feature spotlight', () => {
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
    expect(panel.textContent).toContain('alpha_signal');
    expect(panel.textContent).toContain('beta_signal');
    expect(panel.textContent).toContain('Selected Decision Features');
    expect(panel.textContent).toContain('baseline SELL');
    expect(panel.textContent).toContain('BUY-SELL');
    expect([...panel.querySelectorAll('h3')].map((node) => node.textContent)).toContain('Buy/Sell Feature Summary');
    expect(panel.textContent).not.toContain('Schema vs Baseline');
    expect(panel.textContent).not.toContain('Feature Driver Matrix');
    expect(panel.textContent).not.toContain('Trade Feature vs Baseline');
    expect(panel.textContent).not.toContain('Recent Buy/Sell Events');
    expect(panel.textContent).not.toContain('Model Feature Schemas');
  });

  test('renders baseline-relative schema and buy/sell driver comparisons per model', () => {
    const panel = document.createElement('section');

    renderFeatureComparisonPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
      baselineProfile: 'balanced_rule_residual_v22',
    });

    expect(panel.textContent).toContain('Schema vs Baseline');
    expect(panel.textContent).toContain('baseline balanced_rule_residual_v22');
    expect(panel.textContent).toContain('model-only vs baseline');
    expect(panel.textContent).toContain('baseline-only');
    expect(panel.textContent).toContain('alpha_signal');
    expect(panel.textContent).toContain('beta_signal');
    expect(panel.textContent).toContain('Buy/Sell Drivers vs Baseline');
    expect(panel.textContent).toContain('model BUY mean');
    expect(panel.textContent).toContain('baseline SELL mean');
    expect(panel.textContent).toContain('model_only');
    expect(panel.textContent).toContain('baseline_only');
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

    const snapshotCell = [...panel.querySelectorAll('.feature-snapshot-cell')].find((node) =>
      node.textContent?.includes('feature_0 0.00'),
    );
    expect(snapshotCell?.textContent).toContain('feature_0 0.00');
    expect(snapshotCell?.textContent).toContain('+8 more');
    expect(snapshotCell?.textContent).not.toContain('feature_19');
  });
});
