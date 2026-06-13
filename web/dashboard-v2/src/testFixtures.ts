import type { ComparisonPayload, Manifest, MarketPayload, ModelPayload } from './types';

export const manifestFixture: Manifest = {
  generated_at: '2026-06-13T00:00:00+00:00',
  latest_market_date: '2026-01-06',
  safe_backtest_end: '2026-01-06',
  available_profiles: ['smoke', 'balanced_rule_residual_v22', 'candidate'],
  default_baseline_profile: 'balanced_rule_residual_v22',
  default_comparison_profiles: ['smoke', 'balanced_rule_residual_v22'],
  source_paths: {
    model_replay_freshness: 'dashboard_data/model_replay_freshness.json',
    market: 'data/processed/panels/soxl_soxx_vix_full.csv',
  },
};

export const marketFixture: MarketPayload = {
  symbol: 'SOXL',
  rows: [
    { date: '2026-01-02', open: 9.5, high: 10.2, low: 9.4, close: 10.0, volume: 1000, vix: 19.0, rsi6: 50.0 },
    { date: '2026-01-05', open: 10.2, high: 11.5, low: 10.1, close: 11.0, volume: 1200, vix: 18.0, rsi6: 60.0, spread: 0.4 },
    { date: '2026-01-06', open: 11.0, high: 11.1, low: 10.5, close: 10.8, volume: 1300, vix: 18.5, rsi6: 58.0 },
  ],
};

export const modelFixture: ModelPayload = {
  profile: 'smoke',
  status: 'canonical',
  source_path: 'runtime/model/card/smoke/smoke_soxl_daily.csv',
  replay_start: '2026-01-02',
  replay_end: '2026-01-06',
  row_count: 3,
  rows: [
    { date: '2026-01-02', strategy_nav: 100, buyhold_nav: 100, position: 0, action: 'HOLD', price: 10, excess: 0, excess_pct: 0 },
    { date: '2026-01-05', strategy_nav: 105, buyhold_nav: 110, position: 0.7, action: 'BUY', price: 11, excess: -5, excess_pct: 0.05 },
    { date: '2026-01-06', strategy_nav: 103, buyhold_nav: 108, position: 0.2, action: 'SELL', price: 10.8, excess: -5, excess_pct: 0.03 },
  ],
  events: [
    { date: '2026-01-05', kind: 'BUY', price: 11, position: 0.7, position_delta: 0.7 },
    { date: '2026-01-06', kind: 'SELL', price: 10.8, position: 0.2, position_delta: -0.5 },
  ],
};

export const comparisonFixture: ComparisonPayload = {
  baseline_profile: 'balanced_rule_residual_v22',
  compared_profiles: ['smoke', 'balanced_rule_residual_v22'],
  common_start: '2026-01-02',
  common_end: '2026-01-06',
  rows: [
    {
      date: '2026-01-02',
      action_mismatch_count: 0,
      profiles: {
        smoke: {
          strategy_nav: 100,
          buyhold_nav: 100,
          position: 0,
          action: 'HOLD',
          nav_delta_vs_baseline: 0,
          buyhold_nav_delta_vs_baseline: 0,
          strategy_return: 0,
          buyhold_return: 0,
          return_delta_vs_baseline: 0,
          return_delta_vs_buyhold: 0,
          position_delta_vs_baseline: 0,
          action_mismatch: false,
        },
        balanced_rule_residual_v22: {
          strategy_nav: 100,
          buyhold_nav: 100,
          position: 0,
          action: 'HOLD',
          nav_delta_vs_baseline: 0,
          buyhold_nav_delta_vs_baseline: 0,
          strategy_return: 0,
          buyhold_return: 0,
          return_delta_vs_baseline: 0,
          return_delta_vs_buyhold: 0,
          position_delta_vs_baseline: 0,
          action_mismatch: false,
        },
      },
    },
    {
      date: '2026-01-05',
      action_mismatch_count: 1,
      profiles: {
        smoke: {
          strategy_nav: 105,
          buyhold_nav: 110,
          position: 0.7,
          action: 'BUY',
          nav_delta_vs_baseline: 5,
          buyhold_nav_delta_vs_baseline: 0,
          strategy_return: 0.05,
          buyhold_return: 0.1,
          return_delta_vs_baseline: 0.05,
          return_delta_vs_buyhold: -0.05,
          position_delta_vs_baseline: 0.7,
          action_mismatch: true,
        },
        balanced_rule_residual_v22: {
          strategy_nav: 100,
          buyhold_nav: 110,
          position: 0,
          action: 'HOLD',
          nav_delta_vs_baseline: 0,
          buyhold_nav_delta_vs_baseline: 0,
          strategy_return: 0,
          buyhold_return: 0.1,
          return_delta_vs_baseline: 0,
          return_delta_vs_buyhold: -0.1,
          position_delta_vs_baseline: 0,
          action_mismatch: false,
        },
      },
    },
    {
      date: '2026-01-06',
      action_mismatch_count: 1,
      profiles: {
        smoke: {
          strategy_nav: 103,
          buyhold_nav: 108,
          position: 0.2,
          action: 'SELL',
          nav_delta_vs_baseline: 2,
          buyhold_nav_delta_vs_baseline: 0,
          strategy_return: 0.03,
          buyhold_return: 0.08,
          return_delta_vs_baseline: 0.02,
          return_delta_vs_buyhold: -0.05,
          position_delta_vs_baseline: 0.2,
          action_mismatch: true,
        },
        balanced_rule_residual_v22: {
          strategy_nav: 101,
          buyhold_nav: 108,
          position: 0,
          action: 'HOLD',
          nav_delta_vs_baseline: 0,
          buyhold_nav_delta_vs_baseline: 0,
          strategy_return: 0.01,
          buyhold_return: 0.08,
          return_delta_vs_baseline: 0,
          return_delta_vs_buyhold: -0.07,
          position_delta_vs_baseline: 0,
          action_mismatch: false,
        },
      },
    },
  ],
};
