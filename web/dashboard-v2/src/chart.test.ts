import { beforeEach, describe, expect, test, vi } from 'vitest';
import {
  renderDeltaChart,
  renderMarketChart,
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

  test('renders price candles with close line, MA and volume without model position overlays on the main chart', () => {
    const container = document.createElement('div');
    const challenger = { ...modelFixture, profile: 'balanced_rule_residual_v22' };

    renderMarketChart(
      container,
      marketFixture,
      challenger,
      { smoke: modelFixture, balanced_rule_residual_v22: challenger },
      () => {},
    );

    expect(chartMocks.seriesRecords.map((record) => record.kind)).toEqual(['candlestick', 'line', 'line', 'histogram']);
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
    expect(chartMocks.seriesRecords.some((record) => record.options.priceScaleId === 'left')).toBe(false);
  });

  test('keeps price and volume in readable vertical bands', () => {
    const container = document.createElement('div');

    renderMarketChart(container, marketFixture, modelFixture, { smoke: modelFixture }, () => {});

    expect(chartMocks.priceScaleRecords.map((record) => record.id)).toEqual(['right', 'volume']);
    expect(chartMocks.priceScaleRecords[0].applyOptions).toHaveBeenCalledWith({
      scaleMargins: { top: 0.04, bottom: 0.18 },
    });
    expect(chartMocks.priceScaleRecords[1].applyOptions).toHaveBeenCalledWith({
      scaleMargins: { top: 0.84, bottom: 0.02 },
    });
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

  test('renders NAV deltas against the comparison baseline in the NAV pane', () => {
    const container = document.createElement('div');

    renderDeltaChart(container, comparisonFixture, ['smoke', 'balanced_rule_residual_v22'], 'nav_delta_vs_baseline');

    expect(chartMocks.seriesRecords[0].data).toEqual([
      { time: '2026-01-02', value: 0 },
      { time: '2026-01-05', value: 5 },
      { time: '2026-01-06', value: 2 },
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
