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
import { buildActionMarkers, formatProfileLabel } from './domain';
import type { MarketPayload, ModelPayload } from './types';

const MODEL_COLORS = ['#39b7c5', '#a597ff', '#e1aa3e', '#1fb981'];

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
  valueByTime: Map<string, number>;
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

export function syncCrosshairPositions(handles: readonly ChartHandles[]): () => void {
  let syncing = false;
  const subscriptions: { handle: ChartHandles; handler: (param: MouseEventParams<Time>) => void }[] = [];
  for (const source of handles) {
    const handler = (param: MouseEventParams<Time>) => {
      if (syncing) return;
      syncing = true;
      try {
        const date = eventDate(param);
        if (!date || !param.time) {
          for (const target of handles) {
            if (target !== source) target.chart.clearCrosshairPosition();
          }
          return;
        }
        for (const target of handles) {
          if (target === source) continue;
          const value = target.valueByTime.get(date);
          if (value === undefined) {
            target.chart.clearCrosshairPosition();
          } else {
            target.chart.setCrosshairPosition(value, param.time, target.primarySeries);
          }
        }
      } finally {
        syncing = false;
      }
    };
    source.chart.subscribeCrosshairMove(handler);
    subscriptions.push({ handle: source, handler });
  }
  return () => {
    for (const { handle, handler } of subscriptions) {
      handle.chart.unsubscribeCrosshairMove(handler);
    }
  };
}

export function renderMarketChart(
  container: HTMLElement,
  market: MarketPayload,
  model: ModelPayload,
  models: Record<string, ModelPayload>,
  onDate: (date: string) => void,
): ChartHandles {
  replaceChart(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#15171a' }, textColor: '#edf0f3' },
    grid: { vertLines: { color: '#23272c' }, horzLines: { color: '#23272c' } },
    leftPriceScale: { visible: true, borderColor: '#30343a' },
    rightPriceScale: { borderColor: '#30343a' },
    timeScale: { borderColor: '#30343a', timeVisible: false },
    crosshair: { mode: 1 },
  });
  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: '#1fb981',
    downColor: '#d96b5f',
    borderVisible: false,
    wickUpColor: '#62d7ad',
    wickDownColor: '#ee958b',
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
    color: '#edf0f3',
    lineWidth: 1,
    priceLineVisible: false,
    title: 'SOXL Close',
  });
  closeLine.setData(market.rows.map((row) => ({ time: toTime(row.date), value: Number(row.close) })));

  const primaryIndex = Math.max(0, Object.keys(models).indexOf(model.profile));
  const positionLine = chart.addSeries(LineSeries, {
    color: MODEL_COLORS[primaryIndex % MODEL_COLORS.length],
    lineWidth: 2,
    priceScaleId: 'left',
    priceLineVisible: true,
    title: `${formatProfileLabel(model.profile)} Position`,
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
  return {
    chart,
    primarySeries: candleSeries,
    valueByTime: new Map(market.rows.map((row) => [row.date, Number(row.close)])),
  };
}

export function renderPositionChart(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
  primaryProfile: string,
): ChartHandles {
  replaceChart(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#0f1012' }, textColor: '#edf0f3' },
    grid: { vertLines: { color: '#23272c' }, horzLines: { color: '#23272c' } },
    rightPriceScale: { borderColor: '#30343a' },
    timeScale: { borderColor: '#30343a' },
  });
  let primarySeries: ISeriesApi<'Line'> | null = null;
  let primaryRows: ModelPayload['rows'] | null = null;
  for (const [index, [profile, model]] of Object.entries(models).entries()) {
    const isPrimary = profile === primaryProfile;
    const line = chart.addSeries(LineSeries, {
      color: MODEL_COLORS[index % MODEL_COLORS.length],
      lineWidth: isPrimary ? 2 : 1,
      title: formatProfileLabel(profile),
    });
    line.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.position })));
    if (isPrimary || !primarySeries) {
      primarySeries = line;
      primaryRows = model.rows;
    }
  }
  if (!primarySeries || !primaryRows) {
    throw new Error('position chart requires at least one model');
  }
  chart.timeScale().fitContent();
  trackChart(container, chart);
  return {
    chart,
    primarySeries,
    valueByTime: new Map(primaryRows.map((row) => [row.date, row.position])),
  };
}

export function renderNavChart(container: HTMLElement, models: Record<string, ModelPayload>): ChartHandles {
  replaceChart(container);
  const chart = createChart(container, {
    autoSize: true,
    layout: { background: { color: '#0f1012' }, textColor: '#edf0f3' },
    grid: { vertLines: { color: '#23272c' }, horzLines: { color: '#23272c' } },
    rightPriceScale: { borderColor: '#30343a' },
    timeScale: { borderColor: '#30343a' },
  });
  let primarySeries: ISeriesApi<'Line'> | null = null;
  let primaryRows: ModelPayload['rows'] | null = null;
  for (const [index, [profile, model]] of Object.entries(models).entries()) {
    const series = chart.addSeries(LineSeries, {
      color: MODEL_COLORS[index % MODEL_COLORS.length],
      lineWidth: 2,
      title: formatProfileLabel(profile),
    });
    series.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.strategy_nav })));
    if (!primarySeries) {
      primarySeries = series;
      primaryRows = model.rows;
    }
  }
  if (!primarySeries || !primaryRows) {
    throw new Error('nav chart requires at least one model');
  }
  chart.timeScale().fitContent();
  trackChart(container, chart);
  return {
    chart,
    primarySeries,
    valueByTime: new Map(primaryRows.map((row) => [row.date, row.strategy_nav])),
  };
}
