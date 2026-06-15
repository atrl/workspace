import { beforeEach, describe, expect, test, vi } from 'vitest';
import {
  renderDeltaChart,
  renderMarketChart,
  renderNavChart,
  renderPositionChart,
  syncCrosshairPositions,
  syncVisibleTimeRanges,
} from './chart';
import { comparisonFixture, marketFixture, modelFixture } from './testFixtures';

const chartMocks = vi.hoisted(() => {
  const seriesRecords: { kind: string; options: Record<string, unknown>; data: unknown[] }[] = [];
  const chartRecords: {
    addSeries: ReturnType<typeof vi.fn>;
    remove: ReturnType<typeof vi.fn>;
    clearCrosshairPosition: ReturnType<typeof vi.fn>;
    setCrosshairPosition: ReturnType<typeof vi.fn>;
    priceScale: ReturnType<typeof vi.fn>;
    subscribeClick: ReturnType<typeof vi.fn>;
    subscribeCrosshairMove: ReturnType<typeof vi.fn>;
    unsubscribeCrosshairMove: ReturnType<typeof vi.fn>;
    timeScaleApi: { fitContent: ReturnType<typeof vi.fn> };
  }[] = [];
  const priceScaleRecords: { id: string; applyOptions: ReturnType<typeof vi.fn> }[] = [];
  const addSeries = vi.fn((kind: string, options: Record<string, unknown> = {}) => {
    const record = { kind, options, data: [] as unknown[] };
    seriesRecords.push(record);
    return {
      setData: vi.fn((data: unknown[]) => {
        record.data = data;
      }),
      priceScale: vi.fn(() => {
        const id = String(options.priceScaleId ?? 'right');
        const scaleRecord = { id, applyOptions: vi.fn() };
        priceScaleRecords.push(scaleRecord);
        return { applyOptions: scaleRecord.applyOptions };
      }),
    };
  });
  return {
    addSeries,
    createChart: vi.fn(() => {
      const chart = {
        addSeries,
        remove: vi.fn(),
        clearCrosshairPosition: vi.fn(),
        setCrosshairPosition: vi.fn(),
        priceScale: vi.fn((id: string) => {
          const record = { id, applyOptions: vi.fn() };
          priceScaleRecords.push(record);
          return { applyOptions: record.applyOptions };
        }),
        subscribeClick: vi.fn(),
        subscribeCrosshairMove: vi.fn(),
        unsubscribeCrosshairMove: vi.fn(),
        timeScaleApi: { fitContent: vi.fn() },
      };
      chartRecords.push(chart);
      return { ...chart, timeScale: vi.fn(() => chart.timeScaleApi) };
    }),
    createSeriesMarkers: vi.fn(),
    seriesRecords,
    chartRecords,
    priceScaleRecords,
  };
});

vi.mock('lightweight-charts', () => ({
  CandlestickSeries: 'candlestick',
  HistogramSeries: 'histogram',
  LineSeries: 'line',
  createChart: chartMocks.createChart,
  createSeriesMarkers: chartMocks.createSeriesMarkers,
}));

describe('dashboard v2 chart rendering', () => {
  beforeEach(() => {
    chartMocks.addSeries.mockClear();
    chartMocks.createChart.mockClear();
    chartMocks.createSeriesMarkers.mockClear();
    chartMocks.seriesRecords.length = 0;
    chartMocks.chartRecords.length = 0;
    chartMocks.priceScaleRecords.length = 0;
  });

  test('renders price candles with close line, MA and volume without position overlays on the main chart', () => {
    const container = document.createElement('div');
    const challenger = { ...modelFixture, profile: 'balanced_rule_residual_v22' };

    renderMarketChart(
      container,
      marketFixture,
      challenger,
      { smoke: modelFixture, balanced_rule_residual_v22: challenger },
      () => {},
    );

    expect(chartMocks.seriesRecords.map((record) => record.kind)).toEqual([
      'candlestick',
      'line',
      'line',
      'histogram',
    ]);
    expect(chartMocks.seriesRecords[1].options).toMatchObject({
      title: 'SOXL close',
      priceLineVisible: false,
    });
    expect(chartMocks.seriesRecords[1].data).toEqual(
      marketFixture.rows.map((row) => ({ time: row.date, value: row.close })),
    );
    expect(chartMocks.seriesRecords[2].options).toMatchObject({
      title: 'MA180',
      priceLineVisible: false,
    });
    expect(chartMocks.seriesRecords[2].data).toEqual(
      marketFixture.rows.map((row) => ({ time: row.date, value: row.ma180 })),
    );
    expect(chartMocks.seriesRecords[3].options).toMatchObject({
      priceScaleId: 'volume',
      priceFormat: { type: 'volume' },
      priceLineVisible: false,
      lastValueVisible: false,
    });
    expect(chartMocks.seriesRecords[3].data).toEqual(
      marketFixture.rows.map((row) => ({
        time: row.date,
        value: row.volume,
        color: row.close >= row.open ? 'rgba(31, 185, 129, 0.24)' : 'rgba(217, 107, 95, 0.24)',
      })),
    );
    expect(chartMocks.seriesRecords.some((record) => record.options.priceScaleId === 'position')).toBe(false);
  });

  test('adds selected model action-difference markers directly on the main K-line', () => {
    const container = document.createElement('div');

    renderMarketChart(
      container,
      marketFixture,
      modelFixture,
      { smoke: modelFixture },
      () => {},
      comparisonFixture,
      ['smoke', 'balanced_rule_residual_v22'],
    );

    expect(chartMocks.createSeriesMarkers).toHaveBeenCalledTimes(1);
    expect(chartMocks.createSeriesMarkers.mock.calls[0][1]).toEqual([
      { time: '2026-01-05', position: 'belowBar', color: '#0f766e', shape: 'arrowUp', text: '' },
      { time: '2026-01-06', position: 'aboveBar', color: '#b91c1c', shape: 'arrowDown', text: '' },
      { time: '2026-01-05', position: 'belowBar', color: '#39b7c5', shape: 'arrowUp', text: '' },
      { time: '2026-01-06', position: 'aboveBar', color: '#39b7c5', shape: 'arrowDown', text: '' },
    ]);
  });

  test('keeps price candles dominant and volume in a compact lower lane', () => {
    const container = document.createElement('div');

    renderMarketChart(container, marketFixture, modelFixture, { smoke: modelFixture }, () => {});

    expect(chartMocks.priceScaleRecords.map((record) => record.id)).toEqual(['right', 'volume']);
    expect(chartMocks.priceScaleRecords[0].applyOptions).toHaveBeenCalledWith({
      scaleMargins: { top: 0.04, bottom: 0.18 },
    });
    expect(chartMocks.priceScaleRecords[1].applyOptions).toHaveBeenCalledWith({
      scaleMargins: { top: 0.90, bottom: 0.02 },
    });
    expect(chartMocks.seriesRecords[1].options).toMatchObject({
      color: 'rgba(237, 240, 243, 0.54)',
      lineWidth: 1,
      lastValueVisible: false,
    });
  });

  test('renders actual model positions in the dedicated position pane', () => {
    const container = document.createElement('div');
    const challenger = {
      ...modelFixture,
      profile: 'balanced_rule_residual_v22',
      rows: modelFixture.rows.map((row) => ({ ...row, position: row.position / 2 })),
    };

    renderPositionChart(container, { smoke: modelFixture, balanced_rule_residual_v22: challenger }, 'smoke');

    expect(chartMocks.seriesRecords.map((record) => record.options.title)).toEqual(['smoke', 'BRR v22']);
    expect(chartMocks.seriesRecords[0].data).toEqual([
      { time: '2026-01-02', value: 0 },
      { time: '2026-01-05', value: 0.7 },
      { time: '2026-01-06', value: 0.2 },
    ]);
    expect(chartMocks.seriesRecords[1].data).toEqual([
      { time: '2026-01-02', value: 0 },
      { time: '2026-01-05', value: 0.35 },
      { time: '2026-01-06', value: 0.1 },
    ]);
  });

  test('extends the dedicated position pane with latest signal rows', () => {
    const container = document.createElement('div');
    const latestModel = {
      ...modelFixture,
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.55,
        action: 'BUY' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };

    renderPositionChart(container, { smoke: latestModel }, 'smoke');

    expect(chartMocks.seriesRecords[0].data).toEqual([
      ...modelFixture.rows.map((row) => ({ time: row.date, value: row.position })),
      { time: '2026-01-07', value: 0.55 },
    ]);
  });

  test('renders position deltas against the comparison baseline in the position pane', () => {
    const container = document.createElement('div');

    renderDeltaChart(container, comparisonFixture, ['smoke', 'balanced_rule_residual_v22'], 'position_delta_vs_baseline');

    expect(chartMocks.seriesRecords.map((record) => record.options.title)).toEqual(['smoke Δpos']);
    expect(chartMocks.seriesRecords[0].data).toEqual([
      { time: '2026-01-02', value: 0 },
      { time: '2026-01-05', value: 0.7 },
      { time: '2026-01-06', value: 0.2 },
    ]);
  });

  test('clips delta panes to the current selected date range', () => {
    const container = document.createElement('div');

    renderDeltaChart(
      container,
      comparisonFixture,
      ['smoke', 'balanced_rule_residual_v22'],
      'position_delta_vs_baseline',
      '2026-01-05',
      '2026-01-05',
    );

    expect(chartMocks.seriesRecords[0].data).toEqual([{ time: '2026-01-05', value: 0.7 }]);
  });

  test('keeps latest signal out of the main price chart position overlays', () => {
    const container = document.createElement('div');
    const latestModel = {
      ...modelFixture,
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.55,
        action: 'BUY' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };

    renderMarketChart(container, marketFixture, latestModel, { smoke: latestModel }, () => {});

    expect(chartMocks.seriesRecords.some((record) => record.options.priceScaleId === 'position')).toBe(false);
  });

  test('adds aggregated latest signal buy and sell markers on the main K-line for selected models', () => {
    const container = document.createElement('div');
    const market = {
      ...marketFixture,
      rows: [
        ...marketFixture.rows,
        { date: '2026-01-07', open: 10.7, high: 10.9, low: 10.2, close: 10.4, volume: 1400, ma180: 10.2 },
      ],
    };
    const latestBuy = {
      ...modelFixture,
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.55,
        action: 'BUY' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };
    const latestSell = {
      ...modelFixture,
      profile: 'balanced_rule_residual_v22',
      latest_signal: {
        date: '2026-01-07',
        price: 10.4,
        position: 0.05,
        action: 'SELL' as const,
        replay_asof: '2026-01-06',
        source: 'latest_signal_stateless',
      },
    };

    renderMarketChart(
      container,
      market,
      latestBuy,
      { smoke: latestBuy, balanced_rule_residual_v22: latestSell },
      () => {},
    );

    expect(chartMocks.createSeriesMarkers.mock.calls[0][1]).toContainEqual({
      time: '2026-01-07',
      position: 'belowBar',
      color: '#1fb981',
      shape: 'arrowUp',
      text: '1 latest BUY',
    });
    expect(chartMocks.createSeriesMarkers.mock.calls[0][1]).toContainEqual({
      time: '2026-01-07',
      position: 'aboveBar',
      color: '#d96b5f',
      shape: 'arrowDown',
      text: '1 latest SELL',
    });
  });

  test('renders NAV deltas against the comparison baseline in the NAV pane', () => {
    const container = document.createElement('div');

    renderDeltaChart(container, comparisonFixture, ['smoke', 'balanced_rule_residual_v22'], 'nav_delta_vs_baseline');

    expect(chartMocks.seriesRecords[0].data).toEqual([
      { time: '2026-01-02', value: 0 },
      { time: '2026-01-05', value: 5 },
      { time: '2026-01-06', value: 2 },
    ]);
  });

  test('renders rebased NAV curves for all selected models in the NAV pane', () => {
    const container = document.createElement('div');
    const challenger = {
      ...modelFixture,
      profile: 'balanced_rule_residual_v22',
      rows: [
        { ...modelFixture.rows[0], strategy_nav: 200 },
        { ...modelFixture.rows[1], strategy_nav: 220 },
        { ...modelFixture.rows[2], strategy_nav: 210 },
      ],
    };

    renderNavChart(container, { smoke: modelFixture, balanced_rule_residual_v22: challenger });

    expect(chartMocks.seriesRecords.map((record) => record.options.title)).toEqual(['smoke', 'BRR v22']);
    expect(chartMocks.seriesRecords[0].data).toEqual([
      { time: '2026-01-02', value: 100 },
      { time: '2026-01-05', value: 105 },
      { time: '2026-01-06', value: 103 },
    ]);
    expect(chartMocks.seriesRecords[1].data).toEqual([
      { time: '2026-01-02', value: 100 },
      { time: '2026-01-05', value: 110 },
      { time: '2026-01-06', value: 105 },
    ]);
  });

  test('disposes an old chart before rendering a replacement in the same container', () => {
    const container = document.createElement('div');

    renderMarketChart(container, marketFixture, modelFixture, { smoke: modelFixture }, () => {});
    renderMarketChart(container, marketFixture, modelFixture, { smoke: modelFixture }, () => {});

    expect(chartMocks.chartRecords[0].remove).toHaveBeenCalledTimes(1);
  });

  test('syncs visible logical time ranges across chart panes', () => {
    const subscriptions: ((range: { from: number; to: number } | null) => void)[] = [];
    const rangeChanged = vi.fn();
    const fakeCharts = Array.from({ length: 3 }, () => {
      const api = {
        setVisibleLogicalRange: vi.fn(),
        subscribeVisibleLogicalRangeChange: vi.fn((callback) => subscriptions.push(callback)),
        unsubscribeVisibleLogicalRangeChange: vi.fn(),
      };
      return { timeScale: () => api };
    });

    const cleanup = syncVisibleTimeRanges(fakeCharts as never, rangeChanged);
    subscriptions[0]({ from: 10, to: 20 });
    subscriptions[1](null);
    cleanup();

    expect(fakeCharts[0].timeScale().setVisibleLogicalRange).not.toHaveBeenCalled();
    expect(fakeCharts[1].timeScale().setVisibleLogicalRange).toHaveBeenCalledWith({ from: 10, to: 20 });
    expect(fakeCharts[2].timeScale().setVisibleLogicalRange).toHaveBeenCalledWith({ from: 10, to: 20 });
    expect(rangeChanged).toHaveBeenCalledWith({ from: 10, to: 20 });
    expect(fakeCharts[0].timeScale().unsubscribeVisibleLogicalRangeChange).toHaveBeenCalledTimes(1);
    expect(fakeCharts[1].timeScale().unsubscribeVisibleLogicalRangeChange).toHaveBeenCalledTimes(1);
    expect(fakeCharts[2].timeScale().unsubscribeVisibleLogicalRangeChange).toHaveBeenCalledTimes(1);
  });

  test('syncs crosshair position and clears it across chart panes', () => {
    const subscriptions: ((param: { time?: string }) => void)[] = [];
    const handles = [
      { value: 11 },
      { value: 0.7 },
      { value: 105 },
    ].map(({ value }) => {
      const chart = {
        setCrosshairPosition: vi.fn(),
        clearCrosshairPosition: vi.fn(),
        subscribeCrosshairMove: vi.fn((callback) => subscriptions.push(callback)),
        unsubscribeCrosshairMove: vi.fn(),
      };
      const primarySeries = {};
      return {
        chart,
        primarySeries,
        valueByTime: new Map([['2026-01-05', value]]),
      };
    });

    const cleanup = syncCrosshairPositions(handles as never);
    subscriptions[0]({ time: '2026-01-05' });
    subscriptions[1]({});
    cleanup();

    expect(handles[0].chart.setCrosshairPosition).not.toHaveBeenCalled();
    expect(handles[1].chart.setCrosshairPosition).toHaveBeenCalledWith(0.7, '2026-01-05', handles[1].primarySeries);
    expect(handles[2].chart.setCrosshairPosition).toHaveBeenCalledWith(105, '2026-01-05', handles[2].primarySeries);
    expect(handles[0].chart.clearCrosshairPosition).toHaveBeenCalledTimes(1);
    expect(handles[2].chart.clearCrosshairPosition).toHaveBeenCalledTimes(1);
    expect(handles[0].chart.unsubscribeCrosshairMove).toHaveBeenCalledTimes(1);
    expect(handles[1].chart.unsubscribeCrosshairMove).toHaveBeenCalledTimes(1);
    expect(handles[2].chart.unsubscribeCrosshairMove).toHaveBeenCalledTimes(1);
  });
});
