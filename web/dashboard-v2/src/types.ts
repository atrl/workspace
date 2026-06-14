export type Action = 'BUY' | 'SELL' | 'HOLD';
export type EventKind = 'BUY' | 'SELL' | 'POSITION_INCREASE' | 'POSITION_DECREASE';
export type Tone = 'buy' | 'sell' | 'warning' | 'muted';
export type FeatureValue = string | number | boolean | null;

export interface AutoUpdateTask {
  name: string;
  enabled: boolean;
  configured?: boolean;
  installed?: boolean;
  loaded?: boolean | null;
  label?: string;
  schedule?: string;
  source_path?: string;
  log_path?: string;
  command_path?: string;
}

export interface AutoUpdateStatus {
  configured: boolean;
  timezone: string;
  launchd_detected?: boolean;
  launchagents_dir?: string;
  tasks: AutoUpdateTask[];
  manual_refresh_command?: string;
  install_command?: string;
  run_now_command?: string;
  refresh_pipeline?: string[];
  model_replay_freshness?: {
    generated_at?: string;
    latest_data_date?: string;
    safe_backtest_end?: string;
    counts?: Record<string, number>;
    profile_count?: number;
    ready_count?: number;
    source_path?: string;
    note?: string;
  };
  status_paths?: Record<string, string>;
  remote_deploy_required?: boolean;
}

export interface UnavailableProfile {
  profile: string;
  status?: string;
  result: string;
  reason: string;
  actor?: string;
  profile_yaml?: string;
  output?: string;
}

export type ModelRuntimeRole = 'runnable' | 'reference' | 'evidence_only' | 'unavailable';

export interface ModelRuntimeStatus {
  profile: string;
  display_name?: string;
  status?: string;
  role: ModelRuntimeRole;
  runnable: boolean;
  runnable_reason: string;
  source_kind: string;
  source_path?: string;
  feature_schema_signature?: string;
  feature_schema_source?: string;
  feature_count: number;
  snapshot_count: number;
  row_count: number;
  replay_start?: string;
  replay_end?: string;
  actor?: string;
  profile_yaml?: string;
  output?: string;
}

export interface ModelRetentionGateStatus {
  source_path?: string;
  profiles_scanned?: number;
  passing_profile_count?: number;
  passing_profiles?: string[];
  reference_profile?: string;
  required_best_return_pct?: number;
  max_walkforward_drawdown_pct?: number;
  required_wf_windows?: string[];
  registry_keep_actor_count?: number;
  registry_keep_actor_strict_violation_count?: number;
  registry_keep_actor_strict_violations?: string[];
}

export interface ModelRetentionFrontierStatus {
  source_path?: string;
  rows_scanned?: number;
  completed_handoff_count?: number;
  recommended_next_source?: string | null;
  recommended_next_action?: string | null;
  recommended_primary_blocker?: string | null;
  required_best_return_pct?: number | null;
  reference_return_pct?: number | null;
  max_walkforward_drawdown_pct?: number | null;
  required_wf_windows?: string[];
}

export interface Manifest {
  generated_at: string;
  latest_market_date: string;
  safe_backtest_end: string;
  available_profiles: string[];
  unavailable_profiles?: UnavailableProfile[];
  model_runtime_status?: ModelRuntimeStatus[];
  profile_groups?: {
    runnable: string[];
    reference: string[];
    evidence_only: string[];
  };
  default_baseline_profile: string;
  default_chart_profile?: string;
  default_comparison_profiles: string[];
  model_retention_gate?: ModelRetentionGateStatus | null;
  model_retention_frontier?: ModelRetentionFrontierStatus | null;
  source_paths: Record<string, string>;
  auto_update?: AutoUpdateStatus;
}

export interface MarketRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  [key: string]: string | number | boolean | null;
}

export interface ModelRow {
  date: string;
  strategy_nav: number;
  buyhold_nav: number;
  position: number;
  action: Action;
  price: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  excess: number;
  excess_pct: number;
  feature_snapshot?: Record<string, FeatureValue>;
}

export interface ModelEvent {
  date: string;
  kind: EventKind;
  price: number;
  position: number;
  position_delta: number;
  feature_snapshot?: Record<string, FeatureValue>;
}

export interface FeatureActionSummaryRow {
  feature: string;
  buy_count: number;
  sell_count: number;
  buy_mean: number | null;
  sell_mean: number | null;
  buy_minus_sell: number | null;
}

export interface ModelPayload {
  profile: string;
  status: string | null;
  source_kind: string;
  display_name: string;
  runnable: boolean;
  runnable_reason?: string;
  retention_gate_pass?: boolean;
  retention_decision?: string;
  retention_failures?: string[];
  worst_wf_max_drawdown_pct?: number | null;
  required_best_return_pct?: number | null;
  required_soxlbacktest_return_pct?: number | null;
  feature_packs: string[];
  feature_columns: string[];
  feature_schema_signature?: string;
  feature_schema_source?: string;
  snapshot_columns: string[];
  source_path: string;
  replay_start: string;
  replay_end: string;
  row_count: number;
  rows: ModelRow[];
  events: ModelEvent[];
  feature_action_summary: FeatureActionSummaryRow[];
  replay_aligned?: boolean;
  warmup_start_date?: string;
}

export interface MarketPayload {
  symbol: 'SOXL';
  rows: MarketRow[];
}

export interface ComparisonProfileValue {
  strategy_nav: number;
  buyhold_nav: number;
  position: number;
  action: Action;
  nav_delta_vs_baseline: number;
  buyhold_nav_delta_vs_baseline: number;
  strategy_return: number;
  buyhold_return: number;
  return_delta_vs_baseline: number;
  return_delta_vs_buyhold: number;
  position_delta_vs_baseline: number;
  action_mismatch: boolean;
  drawdown_delta_vs_baseline?: number;
}

export interface ComparisonRow {
  date: string;
  profiles: Record<string, ComparisonProfileValue>;
  action_mismatch_count: number;
}

export interface ComparisonPayload {
  baseline_profile: string;
  compared_profiles: string[];
  common_start: string;
  common_end: string;
  rows: ComparisonRow[];
}

export interface InspectorRow {
  label: string;
  value: string;
  tone?: Tone;
}

export interface ActionFeatureRow {
  profile: string;
  display_name: string;
  action: Action;
  position: number;
  is_trade: boolean;
  source_kind: string;
  feature_packs: string[];
  feature_columns: string[];
  features: Record<string, FeatureValue>;
}

export interface FeatureNameValue {
  feature: string;
  value: FeatureValue;
}

export interface FeatureDeltaValue {
  feature: string;
  value: FeatureValue;
  baseline_value: FeatureValue;
  delta: number | null;
}

export interface DecisionFeatureCard {
  profile: string;
  display_name: string;
  date: string;
  is_exact_date: boolean;
  action: Action;
  position: number;
  baseline_action: Action | 'n/a';
  position_delta_vs_baseline: number | null;
  is_trade: boolean;
  source_kind: string;
  model_only_features: FeatureNameValue[];
  shared_feature_deltas: FeatureDeltaValue[];
  baseline_only_features: FeatureNameValue[];
  buy_sell_features: string[];
}

export interface BaselineFeatureContrastRow {
  date: string;
  profile: string;
  display_name: string;
  action: Action;
  baseline_action: Action | 'n/a';
  position_delta_vs_baseline: number | null;
  feature: string;
  value: FeatureValue;
  baseline_value: FeatureValue;
  delta: number | null;
}

export interface FeatureSchemaOverlapRow {
  profile: string;
  display_name: string;
  source_kind: string;
  runtime: string;
  feature_schema_signature: string;
  feature_schema_source: string;
  feature_count: number;
  snapshot_count: number;
  shared_count: number;
  model_only_count: number;
  shared_features: string[];
  model_only_features: string[];
  top_buy_sell_features: string[];
}

export interface FeatureBaselineSchemaRow {
  profile: string;
  display_name: string;
  runtime: string;
  feature_schema_signature: string;
  feature_schema_source: string;
  feature_count: number;
  snapshot_count: number;
  shared_count: number;
  model_only_count: number;
  baseline_only_count: number;
  shared_features: string[];
  model_only_features: string[];
  baseline_only_features: string[];
  top_buy_sell_features: string[];
}

export interface BuySellFeatureBaselineContrastRow {
  profile: string;
  display_name: string;
  feature: string;
  presence: 'shared' | 'model_only' | 'baseline_only';
  model_buy_count: number;
  model_sell_count: number;
  model_buy_mean: number | null;
  model_sell_mean: number | null;
  model_buy_minus_sell: number | null;
  baseline_buy_count: number;
  baseline_sell_count: number;
  baseline_buy_mean: number | null;
  baseline_sell_mean: number | null;
  baseline_buy_minus_sell: number | null;
  buy_mean_delta: number | null;
  sell_mean_delta: number | null;
  driver_delta_vs_baseline: number | null;
}

export interface ActionMarker {
  time: string;
  position: 'aboveBar' | 'belowBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown';
  text?: 'BUY' | 'SELL' | '';
}
