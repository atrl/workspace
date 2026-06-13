export type Action = 'BUY' | 'SELL' | 'HOLD';
export type EventKind = 'BUY' | 'SELL' | 'POSITION_INCREASE' | 'POSITION_DECREASE';
export type Tone = 'buy' | 'sell' | 'warning' | 'muted';

export interface Manifest {
  generated_at: string;
  latest_market_date: string;
  safe_backtest_end: string;
  available_profiles: string[];
  default_baseline_profile: string;
  default_comparison_profiles: string[];
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
