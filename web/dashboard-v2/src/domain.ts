import type {
  ActionMarker,
  ActionFeatureRow,
  ComparisonPayload,
  ComparisonRow,
  InspectorRow,
  Manifest,
  MarketPayload,
  MarketRow,
  ModelEvent,
  ModelPayload,
  ModelRow,
} from './types';

export function clampComparisonProfiles(
  selected: readonly string[],
  available: readonly string[],
): string[] {
  const unique = selected.filter(
    (profile, index) => selected.indexOf(profile) === index && available.includes(profile),
  );
  const filled = [...unique];
  for (const profile of available) {
    if (filled.length >= 2) break;
    if (!filled.includes(profile)) filled.push(profile);
  }
  return filled.slice(0, 4);
}

export function computeCommonWindow(
  marketRows: readonly MarketRow[],
  modelRowsList: readonly (readonly ModelRow[])[],
): { start: string; end: string } | null {
  if (marketRows.length === 0 || modelRowsList.length === 0) return null;
  const sets = [
    new Set(marketRows.map((row) => row.date)),
    ...modelRowsList.map((rows) => new Set(rows.map((row) => row.date))),
  ];
  const common = [...sets[0]].filter((date) => sets.every((set) => set.has(date))).sort();
  if (common.length === 0) return null;
  return { start: common[0], end: common[common.length - 1] };
}

export function computeDefaultChartWindow(
  marketRows: readonly MarketRow[],
  chartModelRows: readonly ModelRow[],
): { start: string; end: string } | null {
  return computeCommonWindow(marketRows, [chartModelRows]);
}

export function chooseDefaultChartProfile(
  manifest: Manifest,
  models: Record<string, ModelPayload>,
): string {
  const preferred = [
    manifest.default_chart_profile,
    ...manifest.default_comparison_profiles,
    ...manifest.available_profiles,
  ].filter((profile): profile is string => typeof profile === 'string' && profile in models);
  const runnable = preferred.find((profile) => models[profile].runnable);
  return runnable ?? preferred[0] ?? manifest.default_baseline_profile;
}

export function chartRangeRows(
  models: Record<string, ModelPayload>,
  chartProfile: string,
  marketRows: readonly MarketRow[],
): readonly { date: string }[] {
  const rows = models[chartProfile]?.rows;
  return rows && rows.length ? rows : marketRows;
}

export function clipRowsByDate<T extends { date: string }>(rows: readonly T[], start: string, end: string): T[] {
  return rows.filter((row) => row.date >= start && row.date <= end);
}

export type DateRangePreset = '1M' | '3M' | 'YTD' | 'ALL';
export type KeyboardRangeAction = 'zoom_in' | 'zoom_out' | 'pan_left' | 'pan_right' | 'reset';

function sortedDates(rows: readonly { date: string }[]): string[] {
  return [...new Set(rows.map((row) => row.date))].sort();
}

function addMonths(date: Date, months: number): Date {
  const result = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  result.setUTCMonth(result.getUTCMonth() + months);
  return result;
}

function formatDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function snapStart(dates: readonly string[], target: string): string {
  return dates.find((date) => date >= target) ?? dates[dates.length - 1];
}

function snapStartAtOrBefore(dates: readonly string[], target: string): string {
  return [...dates].reverse().find((date) => date <= target) ?? dates[0];
}

function nearestIndex(dates: readonly string[], date: string, fallback: number): number {
  const exact = dates.indexOf(date);
  if (exact >= 0) return exact;
  const next = dates.findIndex((item) => item >= date);
  return next >= 0 ? next : fallback;
}

export function computeDateRangePreset(
  rows: readonly { date: string }[],
  preset: DateRangePreset,
): { start: string; end: string } | null {
  const dates = sortedDates(rows);
  if (dates.length === 0) return null;
  const end = dates[dates.length - 1];
  if (preset === 'ALL') return { start: dates[0], end };
  const endDate = new Date(`${end}T00:00:00Z`);
  const target =
    preset === 'YTD'
      ? `${endDate.getUTCFullYear()}-01-01`
      : formatDate(addMonths(endDate, preset === '1M' ? -1 : -3));
  return { start: preset === 'YTD' ? snapStart(dates, target) : snapStartAtOrBefore(dates, target), end };
}

export function computeKeyboardDateRange(
  rows: readonly { date: string }[],
  start: string,
  end: string,
  action: KeyboardRangeAction,
): { start: string; end: string } | null {
  const dates = sortedDates(rows);
  if (dates.length === 0) return null;
  if (action === 'reset') return { start: dates[0], end: dates[dates.length - 1] };

  const startIndex = nearestIndex(dates, start, 0);
  const endIndex = nearestIndex(dates, end, dates.length - 1);
  const width = Math.max(1, endIndex - startIndex + 1);
  if (action === 'zoom_in' || action === 'zoom_out') {
    const center = Math.round((startIndex + endIndex) / 2);
    const nextWidth = action === 'zoom_in' ? Math.max(2, Math.ceil(width / 2)) : Math.min(dates.length, width * 2);
    const nextStart = Math.max(0, Math.min(dates.length - nextWidth, center - Math.floor(nextWidth / 2)));
    return { start: dates[nextStart], end: dates[nextStart + nextWidth - 1] };
  }

  const shift = action === 'pan_left' ? -Math.max(1, Math.floor(width * 0.25)) : Math.max(1, Math.floor(width * 0.25));
  const nextStart = Math.max(0, Math.min(dates.length - width, startIndex + shift));
  return { start: dates[nextStart], end: dates[nextStart + width - 1] };
}

export function buildActionMarkers(
  events: readonly ModelEvent[],
  options: { showText?: boolean } = {},
): ActionMarker[] {
  const showText = options.showText ?? true;
  return events
    .filter((event) => event.kind === 'BUY' || event.kind === 'SELL')
    .map((event) => {
      const text = event.kind === 'BUY' ? 'BUY' : 'SELL';
      return {
        time: event.date,
        position: text === 'BUY' ? 'belowBar' : 'aboveBar',
        color: text === 'BUY' ? '#0f766e' : '#b91c1c',
        shape: text === 'BUY' ? 'arrowUp' : 'arrowDown',
        text: showText ? text : '',
      };
    });
}

function marketRowsByDate(rows: readonly MarketRow[]): Map<string, MarketRow> {
  return new Map(rows.map((row) => [row.date, row]));
}

function eventSnapshotFor(model: ModelPayload, date: string): Record<string, string | number | boolean | null> {
  const event = model.events.find((item) => item.date === date && (item.kind === 'BUY' || item.kind === 'SELL'));
  return event?.feature_snapshot ?? {};
}

function snapshotFromMarket(
  marketRow: MarketRow | undefined,
  columns: readonly string[],
): Record<string, string | number | boolean | null> {
  if (!marketRow) return {};
  const result: Record<string, string | number | boolean | null> = {};
  for (const column of columns) {
    if (marketRow[column] !== undefined) result[column] = marketRow[column];
  }
  return result;
}

export function buildActionFeatureRows(input: {
  date: string;
  models: Record<string, ModelPayload>;
  marketRows: readonly MarketRow[];
}): ActionFeatureRow[] {
  const market = marketRowsByDate(input.marketRows);
  return Object.entries(input.models)
    .map(([profile, model]) => {
      const modelRow = model.rows.find((row) => row.date === input.date);
      if (!modelRow) return null;
      const eventSnapshot = eventSnapshotFor(model, input.date);
      const marketSnapshot = snapshotFromMarket(market.get(input.date), model.snapshot_columns);
      const rowSnapshot = modelRow.feature_snapshot ?? {};
      return {
        profile,
        display_name: model.display_name || profile,
        action: modelRow.action,
        position: modelRow.position,
        is_trade: modelRow.action === 'BUY' || modelRow.action === 'SELL',
        source_kind: model.source_kind,
        feature_packs: model.feature_packs,
        feature_columns: model.feature_columns,
        features: Object.keys(rowSnapshot).length
          ? rowSnapshot
          : Object.keys(eventSnapshot).length
            ? eventSnapshot
            : marketSnapshot,
      };
    })
    .filter((row): row is ActionFeatureRow => row !== null);
}

export function computeVisibleComparison(
  comparison: ComparisonPayload,
  start: string,
  end: string,
): ComparisonRow[] {
  return comparison.rows.filter((row) => row.date >= start && row.date <= end);
}

export interface InspectorLookup {
  marketRowsByDate: Map<string, MarketRow>;
  comparisonRowsByDate: Map<string, ComparisonRow>;
  modelRowsByProfileAndDate: Record<string, Map<string, ModelRow>>;
}

function indexRowsByDate<T extends { date: string }>(rows: readonly T[]): Map<string, T> {
  return new Map(rows.map((row) => [row.date, row]));
}

export function buildInspectorLookup(input: {
  market: MarketPayload;
  models: Record<string, ModelPayload>;
  comparison: ComparisonPayload;
}): InspectorLookup {
  return {
    marketRowsByDate: indexRowsByDate(input.market.rows),
    comparisonRowsByDate: indexRowsByDate(input.comparison.rows),
    modelRowsByProfileAndDate: Object.fromEntries(
      Object.entries(input.models).map(([profile, model]) => [profile, indexRowsByDate(model.rows)]),
    ),
  };
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function formatPercent(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : 'n/a';
}

function rowToneForAction(action: string): InspectorRow['tone'] {
  if (action === 'BUY') return 'buy';
  if (action === 'SELL') return 'sell';
  return 'muted';
}

export function computeInspectorRows(input: {
  date: string;
  manifest: Manifest;
  market: MarketPayload;
  models: Record<string, ModelPayload>;
  comparison: ComparisonPayload;
}): InspectorRow[] {
  return computeInspectorRowsFromLookup({
    date: input.date,
    manifest: input.manifest,
    lookup: buildInspectorLookup({
      market: input.market,
      models: input.models,
      comparison: input.comparison,
    }),
  });
}

export function computeInspectorRowsFromLookup(input: {
  date: string;
  manifest: Manifest;
  lookup: InspectorLookup;
}): InspectorRow[] {
  const marketRow = input.lookup.marketRowsByDate.get(input.date);
  const comparisonRow = input.lookup.comparisonRowsByDate.get(input.date);
  const rows: InspectorRow[] = [
    { label: 'date', value: input.date },
    { label: 'safe end', value: input.manifest.safe_backtest_end, tone: 'muted' },
  ];

  if (marketRow) {
    rows.push({ label: 'SOXL close', value: formatNumber(Number(marketRow.close)) });
    rows.push({ label: 'volume', value: formatNumber(Number(marketRow.volume), 0), tone: 'muted' });
    for (const key of ['vix', 'rsi', 'rsi6', 'volatility', 'spread', 'vix_zscore', 'panic_score']) {
      if (marketRow[key] !== undefined && marketRow[key] !== null) {
        rows.push({ label: key, value: String(marketRow[key]) });
      }
    }
  }

  for (const [profile, modelRowsByDate] of Object.entries(input.lookup.modelRowsByProfileAndDate)) {
    const modelRow = modelRowsByDate.get(input.date);
    const comparisonValues = comparisonRow?.profiles[profile];
    if (modelRow) {
      rows.push({ label: `${profile} action`, value: modelRow.action, tone: rowToneForAction(modelRow.action) });
      rows.push({ label: `${profile} position`, value: formatNumber(modelRow.position, 3) });
      rows.push({ label: `${profile} NAV`, value: formatNumber(modelRow.strategy_nav, 2) });
      rows.push({ label: `${profile} BuyHold NAV`, value: formatNumber(modelRow.buyhold_nav, 2), tone: 'muted' });
    }
    if (comparisonValues) {
      rows.push({ label: `${profile} return`, value: formatPercent(comparisonValues.strategy_return) });
      rows.push({ label: `${profile} BuyHold return`, value: formatPercent(comparisonValues.buyhold_return), tone: 'muted' });
      rows.push({
        label: `${profile} vs BuyHold`,
        value: formatPercent(comparisonValues.return_delta_vs_buyhold),
        tone: comparisonValues.return_delta_vs_buyhold >= 0 ? 'buy' : 'warning',
      });
    }
  }

  if (comparisonRow) {
    rows.push({
      label: 'action mismatches',
      value: String(comparisonRow.action_mismatch_count),
      tone: comparisonRow.action_mismatch_count > 0 ? 'warning' : 'muted',
    });
  }
  return rows;
}
