import { beforeEach, describe, expect, test, vi } from 'vitest';
import { renderMarketChart, renderPositionChart, syncCrosshairPositions, syncVisibleTimeRanges } from './chart';
import { marketFixture, modelFixture } from './testFixtures';

const chartMocks = vi.hoisted(() => {
  const seriesRecords: { kind: string; options: Record<string, unknown>; data: unknown[] }[] = [];
  const chartRecords: {
    addSeries: ReturnType<typeof vi.fn>;
    remove: ReturnType<typeof vi.fn>;
    clearCrosshairPosition: ReturnType<typeof vi.fn>;
    setCrosshairPosition: ReturnType<typeof vi.fn>;
    subscribeClick: ReturnType<typeof vi.fn>;
    subscribeCrosshairMove: ReturnType<typeof vi.fn>;
    unsubscribeCrosshairMove: ReturnType<typeof vi.fn>;
    timeScaleApi: { fitContent: ReturnType<typeof vi.fn> };
  }[] = [];
  const addSeries = vi.fn((kind: string, options: Record<string, unknown> = {}) => {
    const record = { kind, options, data: [] as unknown[] };
    seriesRecords.push(record);
    return {
      setData: vi.fn((data: unknown[]) => {
        record.data = data;
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
  };
});

vi.mock('lightweight-charts', () => ({
  CandlestickSeries: 'candlestick',
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
  });

  test('renders candles with SOXL close and position overlays on the main chart', () => {
    const container = document.createElement('div');
    const challenger = { ...modelFixture, profile: 'balanced_rule_residual_v22' };

    renderMarketChart(
      container,
      marketFixture,
      challenger,
      { smoke: modelFixture, balanced_rule_residual_v22: challenger },
      () => {},
    );

    expect(chartMocks.seriesRecords.map((record) => record.kind)).toEqual(['candlestick', 'line', 'line']);
    expect(chartMocks.seriesRecords[1].data).toEqual(
      marketFixture.rows.map((row) => ({ time: row.date, value: row.close })),
    );
    expect(chartMocks.seriesRecords[2].options).toMatchObject({
      priceScaleId: 'left',
      lineWidth: 2,
      title: 'BRR v22 Position',
    });
    expect(chartMocks.seriesRecords[2].data).toEqual(
      challenger.rows.map((row) => ({ time: row.date, value: row.position })),
    );
  });

  test('renders multiple position model lines in the position pane', () => {
    const container = document.createElement('div');
    const challenger = { ...modelFixture, profile: 'balanced_rule_residual_v22' };

    renderPositionChart(container, { smoke: modelFixture, balanced_rule_residual_v22: challenger }, 'balanced_rule_residual_v22');

    expect(chartMocks.seriesRecords.map((record) => record.options.title)).toEqual(['smoke', 'BRR v22']);
    expect(chartMocks.seriesRecords[1].options).toMatchObject({ lineWidth: 2 });
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
