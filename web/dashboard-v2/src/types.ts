export type Action = 'BUY' | 'SELL' | 'HOLD';
export type EventKind = 'BUY' | 'SELL' | 'POSITION_INCREASE' | 'POSITION_DECREASE';
export type Tone = 'buy' | 'sell' | 'warning' | 'muted';

export interface Manifest {
  generated_at: string;
  latest_market_date: string;
  safe_backtest_end: string;
  available_profiles: string[];
  deployable_profiles?: string[];
  diagnostic_profiles?: string[];
  default_baseline_profile: string;
  default_comparison_profiles: string[];
  retention_filter?: {
    policy: string;
    input_profile_count: number;
    allowed_profile_count: number;
    blocked_profile_count: number;
  };
  strict_retention_gate?: {
    status: 'passed' | 'blocked';
    source_path: string;
    profiles_scanned: number;
    passing_profile_count: number;
    passing_profiles: string[];
    reference_profile?: string | null;
    reference_return_pct?: number | null;
    current_best_return_pct?: number | null;
    best_return_ratio?: number | null;
    required_best_return_pct?: number | null;
    max_walkforward_drawdown_pct?: number | null;
    required_wf_windows: string[];
    closest_wf_drawdown_candidate?: {
      profile: string;
      worst_wf_max_drawdown_pct?: number | null;
      wf_drawdown_gap_pct?: number | null;
      missing_wf_window_count?: number | null;
      ytd_return_pct?: number | null;
      best_return_gap_pct?: number | null;
    } | null;
    closest_ytd_return_candidate?: {
      profile: string;
      ytd_return_pct?: number | null;
      best_return_gap_pct?: number | null;
      worst_wf_max_drawdown_pct?: number | null;
      wf_drawdown_gap_pct?: number | null;
      missing_wf_window_count?: number | null;
    } | null;
  };
  blocked_profiles?: Record<
    string,
    {
      code: string;
      status: string;
      reason: string;
      summary_path?: string | null;
      final_model_retained: boolean;
    }
  >;
  profile_summaries?: Record<
    string,
    {
      status: string | null;
      retention?: {
        code: string;
        status: string;
        reason: string;
        summary_path?: string | null;
        final_model_retained: boolean;
      };
      replay_start: string;
      replay_end: string;
      row_count: number;
      source_path: string;
      window?: string | null;
    }
  >;
  source_paths: Record<string, string>;
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
  excess: number;
  excess_pct: number;
}

export interface ModelEvent {
  date: string;
  kind: EventKind;
  price: number;
  position: number;
  position_delta: number;
}

export interface ModelPayload {
  profile: string;
  status: string | null;
  source_path: string;
  replay_start: string;
  replay_end: string;
  row_count: number;
  rows: ModelRow[];
  events: ModelEvent[];
}

export interface MarketPayload {
  symbol: 'SOXL';
  rows: MarketRow[];
}

export interface ComparisonProfileValue {
  strategy_nav: number;
  position: number;
  action: Action;
  nav_delta_vs_baseline: number;
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

export interface ActionMismatchRow {
  date: string;
  profile: string;
  model_action: Action;
  baseline_action: Action;
  model_position: number;
  baseline_position: number;
  position_delta_vs_baseline: number;
  nav_delta_vs_baseline: number;
  close: number | null;
  features: Record<string, string>;
}

export interface InspectorRow {
  label: string;
  value: string;
  tone?: Tone;
}

export interface ActionMarker {
  time: string;
  position: 'aboveBar' | 'belowBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown';
  text?: 'BUY' | 'SELL' | '';
}
