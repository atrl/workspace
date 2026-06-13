import type { Action, FeatureActionSummaryRow, FeatureValue, ModelEvent, ModelPayload, ModelRow } from './types';

const DASHBOARD_API_PATH = '/api/run-backtest';
const DASHBOARD_API_FALLBACK = 'http://127.0.0.1:8765/api/run-backtest';
const FALLBACK_STATUSES = new Set([404, 405, 501]);
const API_BASE_QUERY_KEY = 'apiBase';
const API_BASE_STORAGE_KEY = 'dashboardApiBase';

export interface RunBacktestRequest {
  profile: string;
  startDate: string;
  endDate: string;
  tag?: string;
}

interface ApiRow {
  datetime?: string;
  date?: string;
  strategy_nav: number;
  buyhold_nav: number;
  position: number;
  action?: string;
  price?: number;
  close?: number;
  excess?: number;
  excess_pct?: number;
  feature_snapshot?: Record<string, FeatureValue>;
  [key: string]: unknown;
}

interface ApiResponse {
  profile: string;
  tag?: string;
  start_date: string;
  end_date: string;
  warmup_start_date?: string;
  replay_aligned?: boolean;
  feature_packs?: string[];
  feature_columns?: string[];
  snapshot_columns?: string[];
  source_kind?: string;
  rows: ApiRow[];
}

const API_ROW_FIELDS = new Set([
  'datetime',
  'date',
  'strategy_nav',
  'buyhold_nav',
  'position',
  'action',
  'price',
  'excess',
  'excess_pct',
  'feature_snapshot',
  'open',
  'high',
  'low',
  'close',
  'volume',
  'rsi6',
  'ma180',
  'hv_20',
  'vol_coef',
  'panic_score',
  'market_state',
]);

function normalizeAction(value: unknown): Action {
  const action = String(value || 'HOLD').toUpperCase();
  return action === 'BUY' || action === 'SELL' ? action : 'HOLD';
}

function numberOr(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function isFeatureValue(value: unknown): value is FeatureValue {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value);
}

function normalizeFeatureSnapshot(row: ApiRow): Record<string, FeatureValue> | undefined {
  const nested = row.feature_snapshot;
  if (nested && typeof nested === 'object') {
    const entries = Object.entries(nested).filter((entry): entry is [string, FeatureValue] => isFeatureValue(entry[1]));
    return entries.length ? Object.fromEntries(entries) : undefined;
  }

  const snapshot = Object.fromEntries(
    Object.entries(row).filter((entry): entry is [string, FeatureValue] => {
      const [key, value] = entry;
      if (API_ROW_FIELDS.has(key)) return false;
      return isFeatureValue(value);
    }),
  );
  return Object.keys(snapshot).length ? snapshot : undefined;
}

function normalizeRows(rows: readonly ApiRow[]): ModelRow[] {
  return rows.map((row) => {
    const date = String(row.datetime || row.date || '');
    const price = numberOr(row.price ?? row.close, 0);
    const buyholdNav = numberOr(row.buyhold_nav, numberOr(row.strategy_nav, 0));
    const strategyNav = numberOr(row.strategy_nav, buyholdNav);
    return {
      date,
      strategy_nav: strategyNav,
      buyhold_nav: buyholdNav,
      position: numberOr(row.position, 0),
      action: normalizeAction(row.action),
      price,
      excess: numberOr(row.excess, strategyNav - buyholdNav),
      excess_pct: numberOr(row.excess_pct, 0),
      feature_snapshot: normalizeFeatureSnapshot(row),
    };
  });
}

function deriveEvents(rows: readonly ModelRow[], materialPositionDelta = 0.1): ModelEvent[] {
  const events: ModelEvent[] = [];
  let previousPosition = 0;
  for (const row of rows) {
    const positionDelta = row.position - previousPosition;
    if (row.action === 'BUY' || row.action === 'SELL') {
      events.push({
        date: row.date,
        kind: row.action,
        price: row.price,
        position: row.position,
        position_delta: positionDelta,
        feature_snapshot: row.feature_snapshot,
      });
    }
    if (Math.abs(positionDelta) >= materialPositionDelta) {
      events.push({
        date: row.date,
        kind: positionDelta > 0 ? 'POSITION_INCREASE' : 'POSITION_DECREASE',
        price: row.price,
        position: row.position,
        position_delta: positionDelta,
        feature_snapshot: row.feature_snapshot,
      });
    }
    previousPosition = row.position;
  }
  return events;
}

function orderedSnapshotColumns(rows: readonly ModelRow[], preferred: readonly string[] = []): string[] {
  const columns: string[] = [];
  for (const column of preferred) {
    if (!columns.includes(column)) columns.push(column);
  }
  for (const row of rows) {
    for (const column of Object.keys(row.feature_snapshot ?? {})) {
      if (!columns.includes(column)) columns.push(column);
    }
  }
  return columns;
}

function featureNumber(value: FeatureValue): number | null {
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value;
}

function buildFeatureActionSummary(rows: readonly ModelRow[], columns: readonly string[]): FeatureActionSummaryRow[] {
  return columns
    .map((feature) => {
      const buyValues: number[] = [];
      const sellValues: number[] = [];
      for (const row of rows) {
        if (row.action !== 'BUY' && row.action !== 'SELL') continue;
        const value = featureNumber(row.feature_snapshot?.[feature] ?? null);
        if (value === null) continue;
        if (row.action === 'BUY') buyValues.push(value);
        if (row.action === 'SELL') sellValues.push(value);
      }
      const buyMean = buyValues.length ? buyValues.reduce((sum, value) => sum + value, 0) / buyValues.length : null;
      const sellMean = sellValues.length ? sellValues.reduce((sum, value) => sum + value, 0) / sellValues.length : null;
      return {
        feature,
        buy_count: buyValues.length,
        sell_count: sellValues.length,
        buy_mean: buyMean,
        sell_mean: sellMean,
        buy_minus_sell: buyMean !== null && sellMean !== null ? buyMean - sellMean : null,
      };
    })
    .filter((row) => row.buy_count > 0 || row.sell_count > 0)
    .sort((left, right) => {
      const leftScore = Math.abs(left.buy_minus_sell ?? 0);
      const rightScore = Math.abs(right.buy_minus_sell ?? 0);
      return rightScore - leftScore || left.feature.localeCompare(right.feature);
    });
}

function toModelPayload(payload: ApiResponse, request: RunBacktestRequest): ModelPayload {
  const rows = normalizeRows(payload.rows || []);
  const profile = `rerun:${request.profile}:${request.startDate}:${request.endDate}`;
  const snapshotColumns = orderedSnapshotColumns(rows, payload.snapshot_columns ?? payload.feature_columns ?? []);
  const featureColumns = payload.feature_columns?.length ? payload.feature_columns : snapshotColumns;
  return {
    profile,
    status: 'rerun',
    source_kind: payload.source_kind ?? 'interactive_rerun',
    display_name: `${request.profile} rerun ${request.startDate} to ${request.endDate}`,
    runnable: true,
    runnable_reason: 'interactive_rerun',
    feature_packs: payload.feature_packs ?? [],
    feature_columns: featureColumns,
    snapshot_columns: snapshotColumns,
    source_path: 'api/run-backtest',
    replay_start: rows[0]?.date ?? request.startDate,
    replay_end: rows[rows.length - 1]?.date ?? request.endDate,
    row_count: rows.length,
    rows,
    events: deriveEvents(rows),
    feature_action_summary: buildFeatureActionSummary(rows, snapshotColumns),
    warmup_start_date: payload.warmup_start_date,
    replay_aligned: Boolean(payload.replay_aligned),
  };
}

async function postBacktest(url: string, request: RunBacktestRequest): Promise<Response> {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      profile: request.profile,
      start_date: request.startDate,
      end_date: request.endDate,
      tag: request.tag ?? '',
    }),
  });
}

function normalizeApiBase(value: string | null | undefined): string | null {
  const trimmed = String(value || '').trim();
  if (!trimmed) {
    return null;
  }
  return trimmed.replace(/\/+$/, '');
}

function configuredApiFallback(): string {
  if (typeof window === 'undefined') {
    return DASHBOARD_API_FALLBACK;
  }

  const queryBase = normalizeApiBase(new URLSearchParams(window.location.search).get(API_BASE_QUERY_KEY));
  if (queryBase) {
    return `${queryBase}${DASHBOARD_API_PATH}`;
  }

  const storageBase = normalizeApiBase(window.localStorage?.getItem(API_BASE_STORAGE_KEY));
  if (storageBase) {
    return `${storageBase}${DASHBOARD_API_PATH}`;
  }

  return DASHBOARD_API_FALLBACK;
}

export async function runBacktest(request: RunBacktestRequest): Promise<ModelPayload> {
  let response = await postBacktest(DASHBOARD_API_PATH, request);
  if (!response.ok && FALLBACK_STATUSES.has(response.status)) {
    response = await postBacktest(configuredApiFallback(), request);
  }
  const payload = (await response.json()) as ApiResponse & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error || `run-backtest failed: HTTP ${response.status}`);
  }
  return toModelPayload(payload, request);
}
