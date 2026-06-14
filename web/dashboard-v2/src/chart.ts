import {
  CandlestickSeries,
  LineStyle,
  LineSeries,
  PriceScaleMode,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type LogicalRange,
  type MouseEventParams,
  type Time,
} from 'lightweight-charts';
import { buildActionMarkers } from './domain';
import type { MarketPayload, MarketRow, ModelPayload, ModelRow } from './types';

function toTime(date: string): Time {
  return (Date.parse(`${date}T00:00:00Z`) / 1000) as Time;
}

function eventDate(param: MouseEventParams<Time>): string | null {
  if (!param.time) return null;
  if (typeof param.time === 'string') return param.time;
  if (typeof param.time === 'number') return new Date(param.time * 1000).toISOString().slice(0, 10);
  return `${param.time.year}-${String(param.time.month).padStart(2, '0')}-${String(param.time.day).padStart(2, '0')}`;
}

interface PriceRow {
  date: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  price?: number;
}

function finiteNumber(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function priceRowFromMarket(row: MarketRow): PriceRow {
  return {
    date: row.date,
    open: finiteNumber(row.open),
    high: finiteNumber(row.high),
    low: finiteNumber(row.low),
    close: finiteNumber(row.close),
    price: finiteNumber(row.close),
  };
}

function priceRowFromModel(row: ModelRow): PriceRow {
  return {
    date: row.date,
    open: finiteNumber(row.open),
    high: finiteNumber(row.high),
    low: finiteNumber(row.low),
    close: finiteNumber(row.close ?? row.price),
    price: finiteNumber(row.price ?? row.close),
  };
}

function mergePriceRows(marketRows: readonly MarketRow[], modelRows: readonly ModelRow[]): PriceRow[] {
  const byDate = new Map<string, PriceRow>();
  const modelDates = new Set(modelRows.map((row) => row.date));
  for (const row of modelRows) {
    byDate.set(row.date, priceRowFromModel(row));
  }
  for (const row of marketRows) {
    if (modelDates.size > 0 && !modelDates.has(row.date)) continue;
    const marketRow = priceRowFromMarket(row);
    byDate.set(row.date, { ...byDate.get(row.date), ...marketRow });
  }
  return [...byDate.values()].sort((left, right) => left.date.localeCompare(right.date));
}

function dateRangeFromRows(rows: readonly { date: string }[]): { from: Time; to: Time } | null {
  const dates = [...new Set(rows.map((row) => row.date).filter(Boolean))].sort();
  if (dates.length === 0) return null;
  return { from: toTime(dates[0]), to: toTime(dates[dates.length - 1]) };
}

function applyInitialDateRange(chart: IChartApi, rows: readonly { date: string }[]): void {
  const range = dateRangeFromRows(rows);
  if (range) {
    chart.timeScale().setVisibleRange(range);
  } else {
    chart.timeScale().fitContent();
  }
}

function rangeFromValues(values: readonly unknown[]): { minValue: number; maxValue: number } | null {
  const finiteValues = values.map(finiteNumber).filter((value): value is number => value !== undefined);
  if (finiteValues.length === 0) return null;
  let minValue = Math.min(...finiteValues);
  let maxValue = Math.max(...finiteValues);
  if (minValue === maxValue) {
    const padding = Math.max(1, Math.abs(minValue) * 0.01);
    minValue -= padding;
    maxValue += padding;
  }
  return { minValue, maxValue };
}

function fixedAutoscaleInfo(
  range: { minValue: number; maxValue: number } | null,
  margins: { above: number; below: number },
): (() => { priceRange: { minValue: number; maxValue: number } | null; margins: { above: number; below: number } }) | undefined {
  if (!range) return undefined;
  return () => ({ priceRange: range, margins });
}

function priceRangeFromRows(rows: readonly PriceRow[]): { minValue: number; maxValue: number } | null {
  return rangeFromValues(rows.flatMap((row) => [row.low, row.high, row.close, row.price]));
}

function hasOhlc(row: PriceRow): row is Required<Pick<PriceRow, 'date' | 'open' | 'high' | 'low' | 'close'>> & PriceRow {
  return [row.open, row.high, row.low, row.close].every((value) => typeof value === 'number' && Number.isFinite(value));
}

export interface ChartHandles {
  chart: IChartApi;
  primarySeries: ISeriesApi<'Candlestick'> | ISeriesApi<'Line'>;
}

export type ChartRangeCommand =
  | 'zoom_in'
  | 'zoom_out'
  | 'pan_left'
  | 'pan_right'
  | 'reset'
  | 'latest'
  | 'last_60'
  | 'last_120'
  | 'last_250';
export type ChartRangeTarget = IChartApi;
export type ChartDateInteraction = 'hover' | 'select';

const TRADING_VIEW_INTERACTION_OPTIONS = {
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
} as const;

const HOVER_UPDATE_THROTTLE_MS = 64;
const chartByContainer = new WeakMap<HTMLElement, IChartApi>();
const rowCountByChart = new WeakMap<IChartApi, number>();

function exposeChartDebug(name: string, payload: Record<string, unknown>): void {
  if (!window.location.search.includes('debugChart=1')) return;
  const debugWindow = window as typeof window & {
    __SOXL_DASHBOARD_CHART_DEBUG__?: Record<string, Record<string, unknown>>;
  };
  debugWindow.__SOXL_DASHBOARD_CHART_DEBUG__ = {
    ...(debugWindow.__SOXL_DASHBOARD_CHART_DEBUG__ ?? {}),
    [name]: payload,
  };
}

function resetChartContainer(container: HTMLElement): void {
  const previousChart = chartByContainer.get(container);
  if (previousChart) {
    rowCountByChart.delete(previousChart);
    previousChart.remove();
    chartByContainer.delete(container);
  }
  container.replaceChildren();
}

function registerChart(container: HTMLElement, chart: IChartApi, rowCount: number): void {
  chartByContainer.set(container, chart);
  rowCountByChart.set(chart, rowCount);
}

function barCountForCommand(command: ChartRangeCommand): number | null {
  if (command === 'last_60') return 60;
  if (command === 'last_120') return 120;
  if (command === 'last_250') return 250;
  return null;
}

export function syncVisibleLogicalRange(source: IChartApi, targets: IChartApi[]): () => void {
  const sourceTimeScale = source.timeScale();
  const syncRange = (range: LogicalRange | null) => {
    if (!range) return;
    for (const target of targets) {
      target.timeScale().setVisibleLogicalRange(range);
    }
  };
  sourceTimeScale.subscribeVisibleLogicalRangeChange(syncRange);
  return () => sourceTimeScale.unsubscribeVisibleLogicalRangeChange(syncRange);
}

export function syncVisibleLogicalRanges(charts: readonly IChartApi[]): () => void {
  const unsubscribers: (() => void)[] = [];
  let syncing = false;
  let lastRangeKey: string | null = null;

  for (const source of charts) {
    const sourceTimeScale = source.timeScale();
    const syncRange = (range: LogicalRange | null) => {
      if (!range || syncing) return;
      const rangeKey = `${Number(range.from).toPrecision(12)}:${Number(range.to).toPrecision(12)}`;
      if (rangeKey === lastRangeKey) return;
      lastRangeKey = rangeKey;
      syncing = true;
      try {
        for (const target of charts) {
          if (target === source) continue;
          target.timeScale().setVisibleLogicalRange(range);
        }
      } finally {
        syncing = false;
      }
    };
    sourceTimeScale.subscribeVisibleLogicalRangeChange(syncRange);
    unsubscribers.push(() => sourceTimeScale.unsubscribeVisibleLogicalRangeChange(syncRange));
  }

  return () => {
    for (const unsubscribe of unsubscribers) unsubscribe();
  };
}

export function applyChartRangeCommand(charts: readonly ChartRangeTarget[], command: ChartRangeCommand): void {
  if (charts.length === 0) return;
  if (command === 'reset') {
    for (const chart of charts) chart.timeScale().fitContent();
    return;
  }
  if (command === 'latest') {
    for (const chart of charts) chart.timeScale().scrollToRealTime();
    return;
  }

  const fixedBarCount = barCountForCommand(command);
  if (fixedBarCount !== null) {
    for (const chart of charts) {
      const rowCount = rowCountByChart.get(chart);
      const fallbackRange = chart.timeScale().getVisibleLogicalRange();
      const to = rowCount ?? (fallbackRange ? Number(fallbackRange.to) : 0);
      if (to <= 0) continue;
      chart.timeScale().setVisibleLogicalRange({
        from: Math.max(0, to - fixedBarCount),
        to,
      });
    }
    return;
  }

  const sourceRange = charts[0].timeScale().getVisibleLogicalRange();
  if (!sourceRange) return;
  const from = Number(sourceRange.from);
  const to = Number(sourceRange.to);
  const width = Math.max(2, to - from);
  const center = (from + to) / 2;
  let nextFrom = from;
  let nextTo = to;

  if (command === 'zoom_in' || command === 'zoom_out') {
    const nextWidth = command === 'zoom_in' ? Math.max(2, width / 2) : width * 2;
    nextFrom = center - nextWidth / 2;
    nextTo = center + nextWidth / 2;
  } else {
    const shift = Math.max(1, width * 0.25);
    nextFrom = command === 'pan_left' ? from - shift : from + shift;
    nextTo = command === 'pan_left' ? to - shift : to + shift;
  }

  const nextRange = { from: nextFrom, to: nextTo };
  for (const chart of charts) chart.timeScale().setVisibleLogicalRange(nextRange);
}

function createDateEmitter(onDate: (date: string, interaction: ChartDateInteraction) => void): {
  onMove: (date: string) => void;
  onClick: (date: string) => void;
} {
  const requestFrame =
    window.requestAnimationFrame?.bind(window) ??
    ((callback: FrameRequestCallback) => window.setTimeout(() => callback(performance.now()), 16));
  const cancelFrame = window.cancelAnimationFrame?.bind(window) ?? window.clearTimeout.bind(window);
  let pendingDate: string | null = null;
  let frameId: number | null = null;
  let lastHoverDate: string | null = null;
  let lastSelectedDate: string | null = null;
  let lastHoverEmitAt = Number.NEGATIVE_INFINITY;

  const scheduleFlush = () => {
    if (frameId !== null) return;
    frameId = requestFrame(flush);
  };

  const flush = (timestamp: number) => {
    frameId = null;
    const date = pendingDate;
    if (!date || date === lastHoverDate) return;
    if (timestamp - lastHoverEmitAt < HOVER_UPDATE_THROTTLE_MS) {
      scheduleFlush();
      return;
    }
    pendingDate = null;
    lastHoverDate = date;
    lastHoverEmitAt = timestamp;
    onDate(date, 'hover');
  };

  return {
    onMove(date: string) {
      pendingDate = date;
      scheduleFlush();
    },
    onClick(date: string) {
      if (frameId !== null) {
        cancelFrame(frameId);
        frameId = null;
      }
      pendingDate = null;
      if (date === lastSelectedDate) return;
      lastSelectedDate = date;
      onDate(date, 'select');
    },
  };
}

function wireDateEvents(chart: IChartApi, onDate?: (date: string, interaction: ChartDateInteraction) => void): void {
  if (!onDate) return;
  const emitDate = createDateEmitter(onDate);
  chart.subscribeCrosshairMove((param) => {
    const date = eventDate(param);
    if (date) emitDate.onMove(date);
  });
  chart.subscribeClick((param) => {
    const date = eventDate(param);
    if (date) emitDate.onClick(date);
  });
}

export function renderMarketChart(
  container: HTMLElement,
  market: MarketPayload,
  model: ModelPayload,
  onDate: (date: string, interaction: ChartDateInteraction) => void,
): ChartHandles {
  resetChartContainer(container);
  const chart = createChart(container, {
    autoSize: true,
    ...TRADING_VIEW_INTERACTION_OPTIONS,
    layout: { background: { color: '#111827' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#253044' }, horzLines: { color: '#253044' } },
    leftPriceScale: { visible: true, borderColor: '#374151', mode: PriceScaleMode.Normal },
    rightPriceScale: { borderColor: '#374151', mode: PriceScaleMode.Normal },
    timeScale: { borderColor: '#374151', timeVisible: false },
    crosshair: { mode: 1 },
  });
  const priceRows = mergePriceRows(market.rows, model.rows);
  registerChart(container, chart, priceRows.length || model.rows.length);
  const priceAutoscale = fixedAutoscaleInfo(priceRangeFromRows(priceRows), { above: 24, below: 16 });
  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: '#0f766e',
    downColor: '#b91c1c',
    borderVisible: false,
    wickUpColor: '#34d399',
    wickDownColor: '#f87171',
    priceScaleId: 'right',
    autoscaleInfoProvider: priceAutoscale,
  });
  candleSeries.setData(
    priceRows.filter(hasOhlc).map((row) => ({
      time: toTime(row.date),
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    })),
  );
  const closeLine = chart.addSeries(LineSeries, {
    color: '#e5e7eb',
    lineWidth: 1,
    priceScaleId: 'right',
    priceLineVisible: false,
    title: 'SOXL Price',
    autoscaleInfoProvider: priceAutoscale,
  });
  closeLine.setData(
    priceRows
      .map((row) => ({ time: toTime(row.date), value: finiteNumber(row.close ?? row.price) }))
      .filter((row): row is { time: Time; value: number } => row.value !== undefined),
  );

  const positionLine = chart.addSeries(LineSeries, {
    color: '#38bdf8',
    lineWidth: 2,
    priceScaleId: 'left',
    priceLineVisible: true,
    title: 'Position',
    autoscaleInfoProvider: fixedAutoscaleInfo({ minValue: 0, maxValue: 1 }, { above: 8, below: 8 }),
  });
  positionLine.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.position })));
  createSeriesMarkers(
    candleSeries,
    buildActionMarkers(model.events, { showText: false }).map((marker) => ({
      ...marker,
      time: toTime(marker.time),
    })),
  );
  wireDateEvents(chart, onDate);
  applyInitialDateRange(chart, priceRows.length ? priceRows : model.rows);
  exposeChartDebug('market', { chart, candleSeries, closeLine, positionLine, priceRows, modelRows: model.rows });
  return { chart, primarySeries: candleSeries };
}

export function renderPositionChart(
  container: HTMLElement,
  model: ModelPayload,
  onDate?: (date: string, interaction: ChartDateInteraction) => void,
): ChartHandles {
  resetChartContainer(container);
  const chart = createChart(container, {
    autoSize: true,
    ...TRADING_VIEW_INTERACTION_OPTIONS,
    layout: { background: { color: '#0b1220' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
    rightPriceScale: { borderColor: '#374151', mode: PriceScaleMode.Normal },
    timeScale: { borderColor: '#374151' },
  });
  registerChart(container, chart, model.rows.length);
  const line = chart.addSeries(LineSeries, { color: '#38bdf8', lineWidth: 2 });
  line.setData(model.rows.map((row) => ({ time: toTime(row.date), value: row.position })));
  wireDateEvents(chart, onDate);
  applyInitialDateRange(chart, model.rows);
  return { chart, primarySeries: line };
}

export function renderNavChart(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
  baselineProfile?: string,
  onDate?: (date: string, interaction: ChartDateInteraction) => void,
): IChartApi {
  resetChartContainer(container);
  const chart = createChart(container, {
    autoSize: true,
    ...TRADING_VIEW_INTERACTION_OPTIONS,
    layout: { background: { color: '#0b1220' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
    rightPriceScale: { borderColor: '#374151', mode: PriceScaleMode.Normal },
    timeScale: { borderColor: '#374151' },
  });
  registerChart(
    container,
    chart,
    Math.max(0, ...Object.values(models).map((model) => model.rows.length)),
  );
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
  wireDateEvents(chart, onDate);
  applyInitialDateRange(
    chart,
    Object.values(models).flatMap((model) => model.rows),
  );
  return chart;
}
