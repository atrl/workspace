import { beforeEach, describe, expect, test, vi } from 'vitest';
import { applyChartRangeCommand, renderMarketChart, renderNavChart } from './chart';
import { marketFixture, modelFixture } from './testFixtures';

const chartMocks = vi.hoisted(() => {
  const seriesRecords: { kind: string; options: Record<string, unknown>; data: unknown[] }[] = [];
  const subscriptions = {
    click: undefined as ((param: { time?: string }) => void) | undefined,
    crosshairMove: undefined as ((param: { time?: string }) => void) | undefined,
  };
  const addSeries = vi.fn((kind: string, options: Record<string, unknown> = {}) => {
    const record = { kind, options, data: [] as unknown[] };
    seriesRecords.push(record);
    return {
      setData: vi.fn((data: unknown[]) => {
        record.data = data;
      }),
    };
  });
  const timeScaleApi = {
    fitContent: vi.fn(),
    scrollToRealTime: vi.fn(),
    getVisibleLogicalRange: vi.fn(() => ({ from: 10, to: 30 })),
    setVisibleLogicalRange: vi.fn(),
    subscribeVisibleLogicalRangeChange: vi.fn(),
    unsubscribeVisibleLogicalRangeChange: vi.fn(),
  };
  return {
    addSeries,
    timeScaleApi,
    createChart: vi.fn(() => ({
      addSeries,
      subscribeClick: vi.fn((callback: (param: { time?: string }) => void) => {
        subscriptions.click = callback;
      }),
      subscribeCrosshairMove: vi.fn((callback: (param: { time?: string }) => void) => {
        subscriptions.crosshairMove = callback;
      }),
      timeScale: vi.fn(() => timeScaleApi),
      remove: vi.fn(),
    })),
    createSeriesMarkers: vi.fn(),
    subscriptions,
    seriesRecords,
  };
});

vi.mock('lightweight-charts', () => ({
  CandlestickSeries: 'candlestick',
  LineSeries: 'line',
  LineStyle: { Dashed: 2 },
  createChart: chartMocks.createChart,
  createSeriesMarkers: chartMocks.createSeriesMarkers,
}));

describe('dashboard v2 chart rendering', () => {
  beforeEach(() => {
    chartMocks.addSeries.mockClear();
    chartMocks.createChart.mockClear();
    chartMocks.createSeriesMarkers.mockClear();
    chartMocks.timeScaleApi.fitContent.mockClear();
    chartMocks.timeScaleApi.scrollToRealTime.mockClear();
    chartMocks.timeScaleApi.getVisibleLogicalRange.mockClear();
    chartMocks.timeScaleApi.setVisibleLogicalRange.mockClear();
    chartMocks.subscriptions.click = undefined;
    chartMocks.subscriptions.crosshairMove = undefined;
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

  test('enables TradingView-style mouse wheel, drag, and kinetic interactions', () => {
    const container = document.createElement('div');

    renderMarketChart(container, marketFixture, modelFixture, () => {});

    const chartOptions = (chartMocks.createChart.mock.calls[0] as unknown[])[1];
    expect(chartOptions).toMatchObject({
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        mouseWheel: true,
        pinch: true,
      },
      kineticScroll: {
        mouse: true,
        touch: true,
      },
    });
  });

  test('applies chart range commands to existing chart handles', () => {
    applyChartRangeCommand(
      [
        {
          timeScale: () => chartMocks.timeScaleApi,
        } as never,
      ],
      'zoom_in',
    );

    expect(chartMocks.timeScaleApi.setVisibleLogicalRange).toHaveBeenCalledWith({ from: 15, to: 25 });

    applyChartRangeCommand(
      [
        {
          timeScale: () => chartMocks.timeScaleApi,
        } as never,
      ],
      'latest',
    );
    expect(chartMocks.timeScaleApi.scrollToRealTime).toHaveBeenCalled();
  });

  test('coalesces crosshair hover date updates into one animation frame', () => {
    const container = document.createElement('div');
    const callbacks: FrameRequestCallback[] = [];
    const onDate = vi.fn();
    const previousRequestAnimationFrame = window.requestAnimationFrame;
    const previousCancelAnimationFrame = window.cancelAnimationFrame;
    window.requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
      callbacks.push(callback);
      return callbacks.length;
    });
    window.cancelAnimationFrame = vi.fn();

    try {
      renderMarketChart(container, marketFixture, modelFixture, onDate);
      chartMocks.subscriptions.crosshairMove?.({ time: '2026-01-02' });
      chartMocks.subscriptions.crosshairMove?.({ time: '2026-01-05' });
      chartMocks.subscriptions.crosshairMove?.({ time: '2026-01-06' });

      expect(onDate).not.toHaveBeenCalled();
      callbacks[0](0);
      expect(onDate).toHaveBeenCalledTimes(1);
      expect(onDate).toHaveBeenCalledWith('2026-01-06');
    } finally {
      window.requestAnimationFrame = previousRequestAnimationFrame;
      window.cancelAnimationFrame = previousCancelAnimationFrame;
    }
  });

  test('renders buyhold baseline on the NAV chart before strategy lines', () => {
    const container = document.createElement('div');

    renderNavChart(container, { smoke: modelFixture }, 'smoke');

    expect(chartMocks.seriesRecords.map((record) => record.kind)).toEqual(['line', 'line']);
    expect(chartMocks.seriesRecords[0].options).toMatchObject({
      title: 'SOXL BuyHold',
      lineStyle: 2,
    });
    expect(chartMocks.seriesRecords[0].data).toEqual(
      modelFixture.rows.map((row) => ({ time: row.date, value: row.buyhold_nav })),
    );
  });
});
