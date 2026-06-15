import {
  CandlestickSeries,
  HistogramSeries,
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
import type { ComparisonPayload, MarketPayload, ModelPayload } from './types';

const MODEL_COLORS = ['#39b7c5', '#a597ff', '#e1aa3e', '#1fb981', '#f472b6', '#f97316'];

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

export function syncVisibleTimeRanges(
  charts: readonly IChartApi[],
  onRangeChange?: (range: LogicalRange) => void,
): () => void {
  let syncing = false;
  const subscriptions: { chart: IChartApi; handler: (range: LogicalRange | null) => void }[] = [];
  for (const source of charts) {
    const handler = (range: LogicalRange | null) => {
      if (syncing || !range) return;
      onRangeChange?.(range);
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
    leftPriceScale: { visible: true, borderColor: '#30343a', textColor: '#a5adb8' },
    rightPriceScale: { borderColor: '#30343a' },
    timeScale: { borderColor: '#30343a', timeVisible: false, barSpacing: 8, minBarSpacing: 3 },
    crosshair: { mode: 1 },
  });
  chart.priceScale('right').applyOptions({
    scaleMargins: { top: 0.04, bottom: 0.18 },
  });
  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: '#1fb981',
    downColor: '#d96b5f',
    borderVisible: true,
    borderUpColor: '#1fb981',
    borderDownColor: '#d96b5f',
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
    lineWidth: 2,
    priceLineVisible: false,
    title: 'SOXL close',
  });
  closeLine.setData(
    market.rows.map((row) => ({
      time: toTime(row.date),
      value: Number(row.close),
    })),
  );
  const maLine = chart.addSeries(LineSeries, {
    color: '#e1aa3e',
    lineWidth: 1,
    priceLineVisible: false,
    title: 'MA180',
  });
  maLine.setData(
    market.rows
      .filter((row) => typeof row.ma180 === 'number' && Number.isFinite(row.ma180))
      .map((row) => ({ time: toTime(row.date), value: Number(row.ma180) })),
  );
  const volumeBars = chart.addSeries(HistogramSeries, {
    priceScaleId: 'volume',
    priceFormat: { type: 'volume' },
    priceLineVisible: false,
    lastValueVisible: false,
  });
  volumeBars.priceScale().applyOptions({
    scaleMargins: { top: 0.84, bottom: 0.02 },
  });
  volumeBars.setData(
    market.rows.map((row) => ({
      time: toTime(row.date),
      value: Number(row.volume),
      color: Number(row.close) >= Number(row.open) ? 'rgba(31, 185, 129, 0.24)' : 'rgba(217, 107, 95, 0.24)',
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
  trackChart(container, chart);
  return {
    chart,
    primarySeries: candleSeries,
    valueByTime: new Map(market.rows.map((row) => [row.date, Number(row.close)])),
  };
}

type DeltaMetric = 'position_delta_vs_baseline' | 'nav_delta_vs_baseline';

function deltaLabel(metric: DeltaMetric): string {
  return metric === 'position_delta_vs_baseline' ? 'Δpos' : 'Δnav';
}

function deltaData(comparison: ComparisonPayload, profile: string, metric: DeltaMetric): { time: Time; value: number }[] {
  return comparison.rows
    .filter((row) => row.profiles[profile]?.[metric] !== undefined)
    .map((row) => ({
      time: toTime(row.date),
      value: Number(row.profiles[profile][metric] ?? 0),
    }));
}

export function renderDeltaChart(
  container: HTMLElement,
  comparison: ComparisonPayload,
  profiles: readonly string[],
  metric: DeltaMetric,
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
  let primaryRows: { time: Time; value: number }[] | null = null;
  const available = profiles.filter((profile) => comparison.compared_profiles.includes(profile));
  const deltas = available.filter((profile) => profile !== comparison.baseline_profile);
  const selected = deltas.length ? deltas : available;
  for (const [index, profile] of selected.entries()) {
    const rows = deltaData(comparison, profile, metric);
    if (rows.length === 0) continue;
    const isBaseline = profile === comparison.baseline_profile;
    const line = chart.addSeries(LineSeries, {
      color: isBaseline ? '#a5adb8' : MODEL_COLORS[index % MODEL_COLORS.length],
      lineWidth: isBaseline ? 1 : 2,
      priceLineVisible: false,
      title: `${formatProfileLabel(profile)} ${deltaLabel(metric)}`,
    });
    line.setData(rows);
    if (!primarySeries || !isBaseline) {
      primarySeries = line;
      primaryRows = rows;
    }
  }
  if (!primarySeries || !primaryRows) {
    throw new Error('delta chart requires at least one comparable model');
  }
  chart.timeScale().fitContent();
  trackChart(container, chart);
  return {
    chart,
    primarySeries,
    valueByTime: new Map(primaryRows.map((row) => [String(row.time), row.value])),
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

function rebasedNavData(rows: ModelPayload['rows']): { time: Time; value: number }[] {
  const first = rows.find((row) => Number.isFinite(row.strategy_nav) && row.strategy_nav !== 0)?.strategy_nav;
  return rows.map((row) => ({
    time: toTime(row.date),
    value: first ? Number(((row.strategy_nav / first) * 100).toFixed(6)) : row.strategy_nav,
  }));
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
    series.setData(rebasedNavData(model.rows));
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
    valueByTime: new Map(rebasedNavData(primaryRows).map((row) => [String(row.time), row.value])),
  };
}
