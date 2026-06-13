import { afterEach, describe, expect, test, vi } from 'vitest';
import { runBacktest } from './api';

describe('dashboard v2 backtest API client', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.history.replaceState({}, '', '/');
    window.localStorage?.clear?.();
  });

  test('falls back to dashboard_server and normalizes rows for chart rendering', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 501, json: async () => ({ error: 'static server' }) })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          profile: 'balanced_rule_residual_v22',
          start_date: '2026-01-05',
          end_date: '2026-01-06',
          warmup_start_date: '2020-01-02',
          replay_aligned: true,
          rows: [
            {
              datetime: '2026-01-05',
              strategy_nav: 100,
              buyhold_nav: 100,
              position: 0.0,
              action: 'HOLD',
              price: 11,
              excess: 0,
              excess_pct: 0,
              close: 11,
              feature_snapshot: {
                alpha_signal: 0.5,
                shared_feature: 10,
              },
            },
            {
              datetime: '2026-01-06',
              strategy_nav: 102,
              buyhold_nav: 101,
              position: 0.7,
              action: 'BUY',
              price: 12,
              excess: 1,
              excess_pct: 1,
              open: 11.5,
              high: 12.5,
              low: 11.2,
              close: 12,
              feature_snapshot: {
                alpha_signal: 1.25,
                shared_feature: 20,
              },
            },
          ],
        }),
      });
    vi.stubGlobal('fetch', fetchMock);

    const model = await runBacktest({
      profile: 'balanced_rule_residual_v22',
      startDate: '2026-01-05',
      endDate: '2026-01-06',
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/run-backtest',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://127.0.0.1:8765/api/run-backtest',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(model.profile).toBe('rerun:balanced_rule_residual_v22:2026-01-05:2026-01-06');
    expect(model.display_name).toContain('balanced_rule_residual_v22 rerun');
    expect(model.runnable).toBe(true);
    expect(model.rows[0]).toMatchObject({ date: '2026-01-05', price: 11, position: 0 });
    expect(model.rows[1].feature_snapshot).toEqual({ alpha_signal: 1.25, shared_feature: 20 });
    expect(model.snapshot_columns).toEqual(['alpha_signal', 'shared_feature']);
    expect(model.feature_columns).toEqual(['alpha_signal', 'shared_feature']);
    expect(model.feature_action_summary).toEqual([
      {
        feature: 'alpha_signal',
        buy_count: 1,
        sell_count: 0,
        buy_mean: 1.25,
        sell_mean: null,
        buy_minus_sell: null,
      },
      {
        feature: 'shared_feature',
        buy_count: 1,
        sell_count: 0,
        buy_mean: 20,
        sell_mean: null,
        buy_minus_sell: null,
      },
    ]);
    expect(model.events.map((event) => event.kind)).toEqual(['BUY', 'POSITION_INCREASE']);
    expect(model.events[0].feature_snapshot).toEqual({ alpha_signal: 1.25, shared_feature: 20 });
    expect(model.replay_aligned).toBe(true);
    expect(model.warmup_start_date).toBe('2020-01-02');
  });

  test('allows overriding the dashboard_server fallback port from the URL', async () => {
    window.history.replaceState({}, '', '/?apiBase=http://127.0.0.1:8766/');
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 501, json: async () => ({ error: 'static server' }) })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          profile: 'state_conditioned_drawdown_rebound_reward_train_v1',
          start_date: '2026-06-01',
          end_date: '2026-06-09',
          rows: [],
        }),
      });
    vi.stubGlobal('fetch', fetchMock);

    await runBacktest({
      profile: 'state_conditioned_drawdown_rebound_reward_train_v1',
      startDate: '2026-06-01',
      endDate: '2026-06-09',
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://127.0.0.1:8766/api/run-backtest',
      expect.objectContaining({ method: 'POST' }),
    );
  });

  test('does not treat flat market fields as feature snapshots', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        profile: 'legacy_flat_feature_model',
        start_date: '2026-01-05',
        end_date: '2026-01-05',
        rows: [
          {
            datetime: '2026-01-05',
            strategy_nav: 101,
            buyhold_nav: 100,
            position: 0.5,
            action: 'BUY',
            open: 10,
            high: 11,
            low: 9,
            close: 10.5,
            volume: 1200,
            rsi6: 63,
            alpha_signal: 1.5,
          },
        ],
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const model = await runBacktest({
      profile: 'legacy_flat_feature_model',
      startDate: '2026-01-05',
      endDate: '2026-01-05',
    });

    expect(model.rows[0].feature_snapshot).toEqual({ alpha_signal: 1.5 });
    expect(model.snapshot_columns).toEqual(['alpha_signal']);
    expect(model.feature_action_summary[0]).toMatchObject({
      feature: 'alpha_signal',
      buy_count: 1,
      buy_mean: 1.5,
    });
  });
});
