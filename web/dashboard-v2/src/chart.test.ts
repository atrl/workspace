import { beforeEach, describe, expect, test, vi } from 'vitest';
import {
  applyChartRangeCommand,
  renderMarketChart,
  renderNavChart,
  renderPositionChart,
  syncVisibleLogicalRange,
} from './chart';
import { marketFixture, modelFixture } from './testFixtures';

const chartMocks = vi.hoisted(() => {
  const seriesRecords: { kind: string; options: Record<string, unknown>; data: unknown[] }[] = [];
  const subscriptions = {
    click: undefined as ((param: { time?: string }) => void) | undefined,
    crosshairMove: undefined as ((param: { time?: string }) => void) | undefined,
  };
  type VisibleLogicalRange = { from: number; to: number };
  type TimeScaleMock = {
    fitContent: ReturnType<typeof vi.fn>;
    getVisibleLogicalRange: ReturnType<typeof vi.fn>;
    scrollToRealTime: ReturnType<typeof vi.fn>;
    setVisibleLogicalRange: ReturnType<typeof vi.fn>;
    subscribeVisibleLogicalRangeChange: ReturnType<typeof vi.fn>;
    unsubscribeVisibleLogicalRangeChange: ReturnType<typeof vi.fn>;
  };
  type ChartMock = {
    remove: ReturnType<typeof vi.fn>;
    timeScaleApi: TimeScaleMock;
    visibleLogicalRangeHandler?: (range: VisibleLogicalRange | null) => void;
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
  return {
    addSeries,
    chartInstances: [] as ChartMock[],
    createChart: vi.fn(() => {
      const timeScaleApi: TimeScaleMock = {
        fitContent: vi.fn(),
        getVisibleLogicalRange: vi.fn(() => ({ from: 4, to: 20 })),
        scrollToRealTime: vi.fn(),
        setVisibleLogicalRange: vi.fn(),
        subscribeVisibleLogicalRangeChange: vi.fn((callback: (range: VisibleLogicalRange | null) => void) => {
          chart.visibleLogicalRangeHandler = callback;
        }),
        unsubscribeVisibleLogicalRangeChange: vi.fn(),
      };
      const chart = {
        addSeries,
        remove: vi.fn(),
        timeScaleApi,
        subscribeClick: vi.fn((callback: (param: { time?: string }) => void) => {
          subscriptions.click = callback;
        }),
        subscribeCrosshairMove: vi.fn((callback: (param: { time?: string }) => void) => {
          subscriptions.crosshairMove = callback;
        }),
        timeScale: vi.fn(() => timeScaleApi),
      } as ChartMock & {
        addSeries: typeof addSeries;
        subscribeClick: ReturnType<typeof vi.fn>;
        subscribeCrosshairMove: ReturnType<typeof vi.fn>;
        timeScale: ReturnType<typeof vi.fn>;
      };
      chartMocks.chartInstances.push(chart);
      return chart;
    }),
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
    chartMocks.subscriptions.click = undefined;
    chartMocks.subscriptions.crosshairMove = undefined;
    chartMocks.seriesRecords.length = 0;
    chartMocks.chartInstances.length = 0;
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

  test('enables wheel zoom, drag pan, touch drag, and kinetic scrolling on chart surfaces', () => {
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
        axisPressedMouseMove: { time: true, price: true },
        axisDoubleClickReset: { time: true, price: true },
      },
      kineticScroll: { mouse: true, touch: true },
    });
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

  test('removes the previous chart instance before rerendering into the same container', () => {
    const container = document.createElement('div');

    renderMarketChart(container, marketFixture, modelFixture, () => {});
    const firstChart = chartMocks.chartInstances[0];
    renderMarketChart(container, marketFixture, modelFixture, () => {});

    expect(firstChart.remove).toHaveBeenCalledTimes(1);
    expect(chartMocks.chartInstances[1].remove).not.toHaveBeenCalled();
  });

  test('syncs visible logical range from the main chart to companion charts', () => {
    const main = renderMarketChart(document.createElement('div'), marketFixture, modelFixture, () => {});
    const position = renderPositionChart(document.createElement('div'), modelFixture);
    const nav = renderNavChart(document.createElement('div'), { smoke: modelFixture }, 'smoke');

    syncVisibleLogicalRange(main.chart, [position.chart, nav]);
    chartMocks.chartInstances[0].visibleLogicalRangeHandler?.({ from: 1, to: 12 });

    expect(chartMocks.chartInstances[1].timeScaleApi.setVisibleLogicalRange).toHaveBeenCalledWith({ from: 1, to: 12 });
    expect(chartMocks.chartInstances[2].timeScaleApi.setVisibleLogicalRange).toHaveBeenCalledWith({ from: 1, to: 12 });
  });

  test('applies trading chart range commands to synced chart time scales', () => {
    const main = renderMarketChart(document.createElement('div'), marketFixture, modelFixture, () => {});
    const position = renderPositionChart(document.createElement('div'), modelFixture);
    const nav = renderNavChart(document.createElement('div'), { smoke: modelFixture }, 'smoke');

    applyChartRangeCommand([main.chart, position.chart, nav], 'zoom_in');

    expect(chartMocks.chartInstances[0].timeScaleApi.setVisibleLogicalRange).toHaveBeenCalledWith({ from: 8, to: 16 });
    expect(chartMocks.chartInstances[1].timeScaleApi.setVisibleLogicalRange).toHaveBeenCalledWith({ from: 8, to: 16 });
    expect(chartMocks.chartInstances[2].timeScaleApi.setVisibleLogicalRange).toHaveBeenCalledWith({ from: 8, to: 16 });

    applyChartRangeCommand([main.chart, position.chart, nav], 'pan_right');
    expect(chartMocks.chartInstances[0].timeScaleApi.setVisibleLogicalRange).toHaveBeenLastCalledWith({
      from: 8,
      to: 24,
    });

    applyChartRangeCommand([main.chart, position.chart, nav], 'reset');
    expect(chartMocks.chartInstances[0].timeScaleApi.fitContent).toHaveBeenCalledTimes(2);
    expect(chartMocks.chartInstances[1].timeScaleApi.fitContent).toHaveBeenCalledTimes(2);
    expect(chartMocks.chartInstances[2].timeScaleApi.fitContent).toHaveBeenCalledTimes(2);

    applyChartRangeCommand([main.chart, position.chart, nav], 'latest');
    expect(chartMocks.chartInstances[0].timeScaleApi.scrollToRealTime).toHaveBeenCalledTimes(1);
    expect(chartMocks.chartInstances[1].timeScaleApi.scrollToRealTime).toHaveBeenCalledTimes(1);
    expect(chartMocks.chartInstances[2].timeScaleApi.scrollToRealTime).toHaveBeenCalledTimes(1);
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
