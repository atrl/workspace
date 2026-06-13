import type { Action, ModelEvent, ModelPayload, ModelRow } from './types';

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
  [key: string]: unknown;
}

interface ApiResponse {
  profile: string;
  tag?: string;
  start_date: string;
  end_date: string;
  warmup_start_date?: string;
  replay_aligned?: boolean;
  rows: ApiRow[];
}

function normalizeAction(value: unknown): Action {
  const action = String(value || 'HOLD').toUpperCase();
  return action === 'BUY' || action === 'SELL' ? action : 'HOLD';
}

function numberOr(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
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
      });
    }
    if (Math.abs(positionDelta) >= materialPositionDelta) {
      events.push({
        date: row.date,
        kind: positionDelta > 0 ? 'POSITION_INCREASE' : 'POSITION_DECREASE',
        price: row.price,
        position: row.position,
        position_delta: positionDelta,
      });
    }
    previousPosition = row.position;
  }
  return events;
}

function toModelPayload(payload: ApiResponse, request: RunBacktestRequest): ModelPayload {
  const rows = normalizeRows(payload.rows || []);
  const profile = `rerun:${request.profile}:${request.startDate}:${request.endDate}`;
  return {
    profile,
    status: 'rerun',
    source_kind: 'interactive_rerun',
    display_name: `${request.profile} rerun ${request.startDate} to ${request.endDate}`,
    runnable: true,
    runnable_reason: 'interactive_rerun',
    feature_packs: [],
    feature_columns: [],
    snapshot_columns: [],
    source_path: 'api/run-backtest',
    replay_start: rows[0]?.date ?? request.startDate,
    replay_end: rows[rows.length - 1]?.date ?? request.endDate,
    row_count: rows.length,
    rows,
    events: deriveEvents(rows),
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
