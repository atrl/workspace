import type { ComparisonPayload, Manifest, MarketPayload, ModelPayload } from './types';

export interface DashboardData {
  manifest: Manifest;
  market: MarketPayload;
  models: Record<string, ModelPayload>;
  comparison: ComparisonPayload;
}

async function fetchJson<T>(baseUrl: string, path: string): Promise<T> {
  const response = await fetch(`${baseUrl.replace(/\/$/, '')}/${path}`);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function loadDashboardData(baseUrl = '/dashboard_data/v2'): Promise<DashboardData> {
  const manifest = await fetchJson<Manifest>(baseUrl, 'manifest.json');
  const market = await fetchJson<MarketPayload>(baseUrl, 'market/SOXL.json');
  const comparison = await fetchJson<ComparisonPayload>(baseUrl, 'comparison/default.json');
  const models: Record<string, ModelPayload> = {};
  for (const profile of manifest.default_comparison_profiles) {
    models[profile] = await fetchJson<ModelPayload>(baseUrl, `models/${profile}.json`);
  }
  return { manifest, market, models, comparison };
}
