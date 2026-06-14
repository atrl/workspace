import { describe, expect, test } from 'vitest';
import {
  buildActionFeatureRows,
  buildBaselineFeatureContrastRows,
  buildFeatureSchemaOverlapRows,
  buildActionMarkers,
  buildSelectedDateDecisionFeatureCards,
  canRunSelectedWindow,
  clampComparisonProfiles,
  clipRowsByDate,
  computeCommonWindow,
  computeDateRangePreset,
  computeChartCommandDateRange,
  computeInspectorRows,
  computeKeyboardDateRange,
  computeVisibleComparison,
  chooseDefaultChartProfile,
  chooseDefaultRunProfile,
  runnableProfileReason,
  selectableChartProfiles,
  selectableRunProfiles,
} from './domain';
import { betaModelFixture, comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';
import { viewUpdateRequiresChartRender, viewUpdateRequiresFeatureRender, viewUpdateRequiresToolbarRender } from './viewState';

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

  test('prefers selected-window runnable baseline over evidence-only default chart profile', () => {
    const reference = {
      ...modelFixture,
      profile: 'soxl_backtest_policy_reimplementation_v1',
      runnable: false,
      runnable_reason: 'reference_baseline',
      source_kind: 'soxlbacktest_qlib_buysell',
    };
    const evidenceOnly = {
      ...betaModelFixture,
      profile: 'balanced_rule_residual_v19_moe_v2',
      runnable: false,
      runnable_reason: 'evidence_only_retained_dashboard_export',
      source_kind: 'retained_dashboard_export',
    };

    expect(
      chooseDefaultChartProfile(
        {
          ...manifestFixture,
          available_profiles: ['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v19_moe_v2'],
          default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
          default_chart_profile: 'balanced_rule_residual_v19_moe_v2',
          default_comparison_profiles: ['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v19_moe_v2'],
        },
        {
          soxl_backtest_policy_reimplementation_v1: reference,
          balanced_rule_residual_v19_moe_v2: evidenceOnly,
        },
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
    const baselineWithOwnFeature = {
      ...betaModelFixture,
      feature_columns: [...betaModelFixture.feature_columns, 'baseline_only_signal'],
      snapshot_columns: [...betaModelFixture.snapshot_columns, 'baseline_only_signal'],
      rows: betaModelFixture.rows.map((row) =>
        row.date === '2026-01-05'
          ? {
              ...row,
              feature_snapshot: {
                ...(row.feature_snapshot ?? {}),
                baseline_only_signal: 99,
              },
            }
          : row,
      ),
    };
    const rows = buildBaselineFeatureContrastRows({
      models: { smoke: modelFixture, balanced_rule_residual_v22: baselineWithOwnFeature },
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
    const baselineOnlyFeature = rows.find(
      (row) => row.date === '2026-01-05' && row.feature === 'baseline_only_signal',
    );
    expect(baselineOnlyFeature).toMatchObject({
      profile: 'smoke',
      action: 'BUY',
      baseline_action: 'SELL',
      value: null,
      baseline_value: 99,
      delta: null,
    });
  });

  test('builds selected-date decision cards with model-only features and baseline deltas', () => {
    const cards = buildSelectedDateDecisionFeatureCards({
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
      baselineProfile: 'balanced_rule_residual_v22',
    });

    const smoke = cards.find((card) => card.profile === 'smoke');
    const beta = cards.find((card) => card.profile === 'balanced_rule_residual_v22');

    expect(smoke).toMatchObject({
      action: 'BUY',
      baseline_action: 'SELL',
      model_only_features: [{ feature: 'alpha_signal', value: 1.25 }],
      shared_feature_deltas: [
        { feature: 'shared_feature', value: 10, baseline_value: 20, delta: -10 },
      ],
      buy_sell_features: ['alpha_signal', 'shared_feature'],
    });
    expect(smoke?.position_delta_vs_baseline).toBeCloseTo(0.5);
    expect(beta).toMatchObject({
      action: 'SELL',
      baseline_action: 'SELL',
      position_delta_vs_baseline: 0,
      model_only_features: [{ feature: 'beta_signal', value: -2.5 }],
      shared_feature_deltas: [],
    });
  });

  test('uses latest available model row for decision cards when selected date is beyond replay end', () => {
    const truncatedBeta = {
      ...betaModelFixture,
      rows: betaModelFixture.rows.filter((row) => row.date <= '2026-01-05'),
    };
    const cards = buildSelectedDateDecisionFeatureCards({
      date: '2026-01-06',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture, balanced_rule_residual_v22: truncatedBeta },
      baselineProfile: 'smoke',
    });

    const beta = cards.find((card) => card.profile === 'balanced_rule_residual_v22');

    expect(beta).toMatchObject({
      date: '2026-01-05',
      is_exact_date: false,
      action: 'SELL',
      model_only_features: [{ feature: 'beta_signal', value: -2.5 }],
    });
  });

  test('builds feature schema overlap rows with model-only and shared features', () => {
    const rows = buildFeatureSchemaOverlapRows(
      { smoke: modelFixture, balanced_rule_residual_v22: betaModelFixture },
      { featureNameLimit: 5 },
    );
    const smoke = rows.find((row) => row.profile === 'smoke');
    const beta = rows.find((row) => row.profile === 'balanced_rule_residual_v22');

    expect(smoke).toMatchObject({
      shared_count: 1,
      model_only_count: 1,
      shared_features: ['shared_feature'],
      model_only_features: ['alpha_signal'],
      top_buy_sell_features: ['alpha_signal', 'shared_feature'],
    });
    expect(beta).toMatchObject({
      shared_features: ['shared_feature'],
      model_only_features: ['beta_signal'],
      runtime: 'runnable',
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
    expect(viewUpdateRequiresToolbarRender({ runProfile: 'soxl_backtest_policy_reimplementation_v1' })).toBe(true);
    expect(viewUpdateRequiresChartRender({ runProfile: 'soxl_backtest_policy_reimplementation_v1' })).toBe(false);
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

  test('maps TradingView chart commands to selected rerun date windows', () => {
    const rows = Array.from({ length: 100 }, (_, index) => {
      const date = new Date(Date.UTC(2026, 0, 2 + index));
      return { date: date.toISOString().slice(0, 10) };
    });

    expect(computeChartCommandDateRange(rows, rows[0].date, rows[99].date, 'last_60')).toEqual({
      start: rows[40].date,
      end: rows[99].date,
    });
    expect(computeChartCommandDateRange(rows, rows[10].date, rows[39].date, 'latest')).toEqual({
      start: rows[70].date,
      end: rows[99].date,
    });
    expect(computeChartCommandDateRange(rows, rows[10].date, rows[39].date, 'zoom_in')).toEqual({
      start: rows[18].date,
      end: rows[32].date,
    });
    expect(computeChartCommandDateRange(rows, rows[10].date, rows[39].date, 'pan_right')).toEqual({
      start: rows[17].date,
      end: rows[46].date,
    });
    expect(computeChartCommandDateRange(rows, rows[10].date, rows[39].date, 'reset')).toEqual({
      start: rows[0].date,
      end: rows[99].date,
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
    ).toBe('deterministic baseline selected-window replay');
    expect(
      runnableProfileReason({
        ...betaModelFixture,
        runnable: false,
        runnable_reason: 'missing_actor_artifact',
      }),
    ).toBe('missing actor artifact');
  });

  test('allows selected-window reruns for deterministic SOXLBacktest baseline without marking it as an RL actor', () => {
    const baseline = {
      ...modelFixture,
      runnable: false,
      runnable_reason: 'reference_baseline',
      source_kind: 'soxlbacktest_qlib_buysell',
    };
    const evidenceOnly = {
      ...modelFixture,
      runnable: false,
      runnable_reason: 'evidence_only_retained_dashboard_export',
      source_kind: 'retained_dashboard_export',
    };

    expect(canRunSelectedWindow(modelFixture)).toBe(true);
    expect(canRunSelectedWindow(baseline)).toBe(true);
    expect(canRunSelectedWindow(evidenceOnly)).toBe(false);
  });

  test('keeps evidence-only models selectable for chart review even when a runnable baseline is selected', () => {
    const baseline = {
      ...modelFixture,
      profile: 'soxl_backtest_policy_reimplementation_v1',
      runnable: false,
      runnable_reason: 'reference_baseline',
      source_kind: 'soxlbacktest_qlib_buysell',
    };
    const evidenceOnly = {
      ...betaModelFixture,
      runnable: false,
      runnable_reason: 'evidence_only_retained_dashboard_export',
      source_kind: 'retained_dashboard_export',
    };

    expect(
      selectableChartProfiles(
        {
          soxl_backtest_policy_reimplementation_v1: baseline,
          balanced_rule_residual_v22: evidenceOnly,
        },
        ['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v22'],
      ),
    ).toEqual(['soxl_backtest_policy_reimplementation_v1', 'balanced_rule_residual_v22']);
  });

  test('keeps selected-window run sources separate from evidence-only chart review profiles', () => {
    const baseline = {
      ...modelFixture,
      profile: 'soxl_backtest_policy_reimplementation_v1',
      runnable: false,
      runnable_reason: 'reference_baseline',
      source_kind: 'soxlbacktest_qlib_buysell',
    };
    const evidenceOnly = {
      ...betaModelFixture,
      profile: 'balanced_rule_residual_v19_moe_v2',
      runnable: false,
      runnable_reason: 'evidence_only_retained_dashboard_export',
      source_kind: 'retained_dashboard_export',
    };

    const models = {
      soxl_backtest_policy_reimplementation_v1: baseline,
      balanced_rule_residual_v19_moe_v2: evidenceOnly,
      smoke: modelFixture,
    };
    const manifest = {
      ...manifestFixture,
      available_profiles: ['balanced_rule_residual_v19_moe_v2', 'soxl_backtest_policy_reimplementation_v1', 'smoke'],
      default_chart_profile: 'balanced_rule_residual_v19_moe_v2',
      default_baseline_profile: 'soxl_backtest_policy_reimplementation_v1',
      default_comparison_profiles: ['balanced_rule_residual_v19_moe_v2', 'soxl_backtest_policy_reimplementation_v1'],
    };

    expect(selectableRunProfiles(models, manifest.available_profiles)).toEqual([
      'soxl_backtest_policy_reimplementation_v1',
      'smoke',
    ]);
    expect(chooseDefaultRunProfile(manifest, models, 'balanced_rule_residual_v19_moe_v2')).toBe(
      'soxl_backtest_policy_reimplementation_v1',
    );
    expect(chooseDefaultRunProfile(manifest, models, 'smoke')).toBe('smoke');
  });
});
