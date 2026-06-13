import { afterEach, describe, expect, test, vi } from 'vitest';
import { loadDashboardData } from './data';
import {
  betaModelFixture,
  comparisonFixture,
  manifestFixture,
  marketFixture,
  modelFixture,
} from './testFixtures';

describe('dashboard v2 data loader', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test('loads every available model payload, not only default comparison profiles', async () => {
    const candidate = { ...modelFixture, profile: 'candidate', display_name: 'candidate' };
    const payloads: Record<string, unknown> = {
      '/dashboard_data/v2/manifest.json': manifestFixture,
      '/dashboard_data/v2/market/SOXL.json': marketFixture,
      '/dashboard_data/v2/comparison/default.json': comparisonFixture,
      '/dashboard_data/v2/models/smoke.json': modelFixture,
      '/dashboard_data/v2/models/balanced_rule_residual_v22.json': betaModelFixture,
      '/dashboard_data/v2/models/candidate.json': candidate,
    };
    const fetchMock = vi.fn(async (url: string) => ({
      ok: true,
      json: async () => payloads[url],
    }));
    vi.stubGlobal('fetch', fetchMock);

    const data = await loadDashboardData();

    expect(Object.keys(data.models)).toEqual(['smoke', 'balanced_rule_residual_v22', 'candidate']);
    expect(fetchMock).toHaveBeenCalledWith('/dashboard_data/v2/models/candidate.json');
  });
});
