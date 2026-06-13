import {
  CandlestickSeries,
  LineStyle,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type MouseEventParams,
  type Time,
} from 'lightweight-charts';
import { buildActionMarkers } from './domain';
import type { MarketPayload, ModelPayload } from './types';

function toTime(date: string): Time {
  return date as Time;
}

function eventDate(param: MouseEventParams<Time>): string | null {
  if (!param.time) return null;
  if (typeof param.time === 'string') return param.time;
  if (typeof param.time === 'number') return new Date(param.time * 1000).toISOString().slice(0, 10);
  return `${param.time.year}-${String(param.time.month).padStart(2, '0')}-${String(param.time.day).padStart(2, '0')}`;
}

export interface ChartHandles {
  chart: IChartApi;
  primarySeries: ISeriesApi<'Candlestick'> | ISeriesApi<'Line'>;
}

const chartByContainer = new WeakMap<HTMLElement, IChartApi>();

function resetChartContainer(container: HTMLElement): void {
  const previousChart = chartByContainer.get(container);
  if (previousChart) {
    previousChart.remove();
    chartByContainer.delete(container);
  }
  container.replaceChildren();
}

function registerChart(container: HTMLElement, chart: IChartApi): void {
  chartByContainer.set(container, chart);
}

function createDateEmitter(onDate: (date: string) => void): {
  onMove: (date: string) => void;
  onClick: (date: string) => void;
} {
  const requestFrame =
    window.requestAnimationFrame?.bind(window) ??
    ((callback: FrameRequestCallback) => window.setTimeout(() => callback(performance.now()), 16));
  const cancelFrame = window.cancelAnimationFrame?.bind(window) ?? window.clearTimeout.bind(window);
  let pendingDate: string | null = null;
  let frameId: number | null = null;
  let lastDate: string | null = null;

  const flush = () => {
    frameId = null;
    const date = pendingDate;
    pendingDate = null;
    if (!date || date === lastDate) return;
    lastDate = date;
    onDate(date);
  };

  return {
    onMove(date: string) {
      pendingDate = date;
      if (frameId !== null) return;
      frameId = requestFrame(flush);
    },
    onClick(date: string) {
      if (frameId !== null) {
        cancelFrame(frameId);
        frameId = null;
      }
      pendingDate = null;
      if (date === lastDate) return;
      lastDate = date;
      onDate(date);
    },
  };
}

export function renderMarketChart(
  container: HTMLElement,
  market: MarketPayload,
  model: ModelPayload,
  onDate: (date: string) => void,
): ChartHandles {
  resetChartContainer(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#111827' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#253044' }, horzLines: { color: '#253044' } },
    leftPriceScale: { visible: true, borderColor: '#374151' },
    rightPriceScale: { borderColor: '#374151' },
    timeScale: { borderColor: '#374151', timeVisible: false },
    crosshair: { mode: 1 },
  });
  registerChart(container, chart);
  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: '#0f766e',
    downColor: '#b91c1c',
    borderVisible: false,
    wickUpColor: '#34d399',
    wickDownColor: '#f87171',
  });
  candleSeries.setData(
    market.rows.map((row) => ({
      time: toTime(row.date),
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
    })),
  );
  const closeLine = chart.addSeries(LineSeries, {
    color: '#e5e7eb',
    lineWidth: 1,
    priceLineVisible: false,
    title: 'SOXL Close',
  });
  closeLine.setData(market.rows.map((row) => ({ time: toTime(row.date), value: Number(row.close) })));

  const positionLine = chart.addSeries(LineSeries, {
    color: '#38bdf8',
    lineWidth: 2,
    priceScaleId: 'left',
    priceLineVisible: true,
    title: 'Position',
  });
  positionLine.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.position })));
  createSeriesMarkers(
    candleSeries,
    buildActionMarkers(model.events, { showText: false }).map((marker) => ({
      ...marker,
      time: toTime(marker.time),
    })),
  );
  const emitDate = createDateEmitter(onDate);
  chart.subscribeCrosshairMove((param) => {
    const date = eventDate(param);
    if (date) emitDate.onMove(date);
  });
  chart.subscribeClick((param) => {
    const date = eventDate(param);
    if (date) emitDate.onClick(date);
  });
  chart.timeScale().fitContent();
  return { chart, primarySeries: candleSeries };
}

export function renderPositionChart(container: HTMLElement, model: ModelPayload): ChartHandles {
  resetChartContainer(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#0b1220' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
    rightPriceScale: { borderColor: '#374151' },
    timeScale: { borderColor: '#374151' },
  });
  registerChart(container, chart);
  const line = chart.addSeries(LineSeries, { color: '#38bdf8', lineWidth: 2 });
  line.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.position })));
  chart.timeScale().fitContent();
  return { chart, primarySeries: line };
}

export function renderNavChart(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
  baselineProfile?: string,
): IChartApi {
  resetChartContainer(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#0b1220' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
    rightPriceScale: { borderColor: '#374151' },
    timeScale: { borderColor: '#374151' },
  });
  registerChart(container, chart);
  const buyholdSource = (baselineProfile ? models[baselineProfile] : undefined) ?? Object.values(models)[0];
  if (buyholdSource) {
    const buyholdSeries = chart.addSeries(LineSeries, {
      color: '#e5e7eb',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      title: 'SOXL BuyHold',
    });
    buyholdSeries.setData(
      buyholdSource.rows.map((row) => ({ time: toTime(row.date), value: row.buyhold_nav })),
    );
  }
  const colors = ['#38bdf8', '#a78bfa', '#f59e0b', '#22c55e'];
  Object.entries(models).forEach(([profile, model], index) => {
    const series = chart.addSeries(LineSeries, { color: colors[index % colors.length], lineWidth: 2, title: profile });
    series.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.strategy_nav })));
  });
  chart.timeScale().fitContent();
  return chart;
}
