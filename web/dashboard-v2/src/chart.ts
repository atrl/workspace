import {
  CandlestickSeries,
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

export function renderMarketChart(
  container: HTMLElement,
  market: MarketPayload,
  model: ModelPayload,
  onDate: (date: string) => void,
): ChartHandles {
  container.replaceChildren();
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#111827' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#253044' }, horzLines: { color: '#253044' } },
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
  return { chart, primarySeries: candleSeries };
}

export function renderPositionChart(container: HTMLElement, model: ModelPayload): ChartHandles {
  container.replaceChildren();
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
  return { chart, primarySeries: line };
}

export function renderNavChart(container: HTMLElement, models: Record<string, ModelPayload>): IChartApi {
  container.replaceChildren();
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
  return chart;
}
