import { beforeEach, describe, expect, test, vi } from 'vitest';
import { renderMarketChart, syncVisibleTimeRanges } from './chart';
import { marketFixture, modelFixture } from './testFixtures';

const chartMocks = vi.hoisted(() => {
  const seriesRecords: { kind: string; options: Record<string, unknown>; data: unknown[] }[] = [];
  const chartRecords: {
    addSeries: ReturnType<typeof vi.fn>;
    remove: ReturnType<typeof vi.fn>;
    subscribeClick: ReturnType<typeof vi.fn>;
    subscribeCrosshairMove: ReturnType<typeof vi.fn>;
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
        subscribeClick: vi.fn(),
        subscribeCrosshairMove: vi.fn(),
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

    renderMarketChart(container, marketFixture, modelFixture, () => {});

    expect(chartMocks.seriesRecords.map((record) => record.kind)).toEqual(['candlestick', 'line', 'line']);
    expect(chartMocks.seriesRecords[1].data).toEqual(
      marketFixture.rows.map((row) => ({ time: row.date, value: row.close })),
    );
    expect(chartMocks.seriesRecords[2].options).toMatchObject({
      priceScaleId: 'left',
      title: 'Position',
    });
    expect(chartMocks.seriesRecords[2].data).toEqual(
      modelFixture.rows.map((row) => ({ time: row.date, value: row.position })),
    );
  });

  test('disposes an old chart before rendering a replacement in the same container', () => {
    const container = document.createElement('div');

    renderMarketChart(container, marketFixture, modelFixture, () => {});
    renderMarketChart(container, marketFixture, modelFixture, () => {});

    expect(chartMocks.chartRecords[0].remove).toHaveBeenCalledTimes(1);
  });

  test('syncs visible logical time ranges across chart panes', () => {
    const subscriptions: ((range: { from: number; to: number } | null) => void)[] = [];
    const fakeCharts = Array.from({ length: 3 }, () => {
      const api = {
        setVisibleLogicalRange: vi.fn(),
        subscribeVisibleLogicalRangeChange: vi.fn((callback) => subscriptions.push(callback)),
        unsubscribeVisibleLogicalRangeChange: vi.fn(),
      };
      return { timeScale: () => api };
    });

    const cleanup = syncVisibleTimeRanges(fakeCharts as never);
    subscriptions[0]({ from: 10, to: 20 });
    subscriptions[1](null);
    cleanup();

    expect(fakeCharts[0].timeScale().setVisibleLogicalRange).not.toHaveBeenCalled();
    expect(fakeCharts[1].timeScale().setVisibleLogicalRange).toHaveBeenCalledWith({ from: 10, to: 20 });
    expect(fakeCharts[2].timeScale().setVisibleLogicalRange).toHaveBeenCalledWith({ from: 10, to: 20 });
    expect(fakeCharts[0].timeScale().unsubscribeVisibleLogicalRangeChange).toHaveBeenCalledTimes(1);
    expect(fakeCharts[1].timeScale().unsubscribeVisibleLogicalRangeChange).toHaveBeenCalledTimes(1);
    expect(fakeCharts[2].timeScale().unsubscribeVisibleLogicalRangeChange).toHaveBeenCalledTimes(1);
  });
});
