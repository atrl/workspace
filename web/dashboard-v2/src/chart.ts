import {
  CandlestickSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type LogicalRange,
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

function replaceChart(container: HTMLElement): void {
  chartByContainer.get(container)?.remove();
  chartByContainer.delete(container);
  container.replaceChildren();
}

function trackChart(container: HTMLElement, chart: IChartApi): void {
  chartByContainer.set(container, chart);
}

export function syncVisibleTimeRanges(charts: readonly IChartApi[]): () => void {
  let syncing = false;
  const subscriptions: { chart: IChartApi; handler: (range: LogicalRange | null) => void }[] = [];
  for (const source of charts) {
    const handler = (range: LogicalRange | null) => {
      if (syncing || !range) return;
      syncing = true;
      try {
        for (const target of charts) {
          if (target !== source) target.timeScale().setVisibleLogicalRange(range);
        }
      } finally {
        syncing = false;
      }
    };
    source.timeScale().subscribeVisibleLogicalRangeChange(handler);
    subscriptions.push({ chart: source, handler });
  }
  return () => {
    for (const { chart, handler } of subscriptions) {
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler);
    }
  };
}

export function renderMarketChart(
  container: HTMLElement,
  market: MarketPayload,
  model: ModelPayload,
  onDate: (date: string) => void,
): ChartHandles {
  replaceChart(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#111827' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#253044' }, horzLines: { color: '#253044' } },
    leftPriceScale: { visible: true, borderColor: '#374151' },
    rightPriceScale: { borderColor: '#374151' },
    timeScale: { borderColor: '#374151', timeVisible: false },
    crosshair: { mode: 1 },
  });
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
  chart.subscribeCrosshairMove((param) => {
    const date = eventDate(param);
    if (date) onDate(date);
  });
  chart.subscribeClick((param) => {
    const date = eventDate(param);
    if (date) onDate(date);
  });
  chart.timeScale().fitContent();
  trackChart(container, chart);
  return { chart, primarySeries: candleSeries };
}

export function renderPositionChart(container: HTMLElement, model: ModelPayload): ChartHandles {
  replaceChart(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#0b1220' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
    rightPriceScale: { borderColor: '#374151' },
    timeScale: { borderColor: '#374151' },
  });
  const line = chart.addSeries(LineSeries, { color: '#38bdf8', lineWidth: 2 });
  line.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.position })));
  chart.timeScale().fitContent();
  trackChart(container, chart);
  return { chart, primarySeries: line };
}

export function renderNavChart(container: HTMLElement, models: Record<string, ModelPayload>): IChartApi {
  replaceChart(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#0b1220' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
    rightPriceScale: { borderColor: '#374151' },
    timeScale: { borderColor: '#374151' },
  });
  const colors = ['#38bdf8', '#a78bfa', '#f59e0b', '#22c55e'];
  Object.entries(models).forEach(([profile, model], index) => {
    const series = chart.addSeries(LineSeries, { color: colors[index % colors.length], lineWidth: 2, title: profile });
    series.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.strategy_nav })));
  });
  chart.timeScale().fitContent();
  trackChart(container, chart);
  return chart;
}
