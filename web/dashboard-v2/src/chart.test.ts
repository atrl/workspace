import { beforeEach, describe, expect, test, vi } from 'vitest';
import { renderMarketChart } from './chart';
import { marketFixture, modelFixture } from './testFixtures';

const chartMocks = vi.hoisted(() => {
  const seriesRecords: { kind: string; options: Record<string, unknown>; data: unknown[] }[] = [];
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
    createChart: vi.fn(() => ({
      addSeries,
      subscribeClick: vi.fn(),
      subscribeCrosshairMove: vi.fn(),
      timeScale: vi.fn(() => ({ fitContent: vi.fn() })),
    })),
    createSeriesMarkers: vi.fn(),
    seriesRecords,
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
});
