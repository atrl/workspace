import type {
  ActionMarker,
  ActionMismatchRow,
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

const DIFF_MARKER_COLOR = '#22c7a9';
const MATERIAL_POSITION_DELTA = 0.05;
const DEFAULT_VISIBLE_TRADING_DAYS = 90;

export function formatProfileLabel(profile: string): string {
  if (profile === 'soxl_backtest_policy_reimplementation_v1') return 'SOXLBacktest';
  if (profile === 'mode_conditioned_rebound_reward_train_v2') return 'Mode rebound v2';
  if (profile === 'state_conditioned_drawdown_rebound_reward_train_v1') return 'State drawdown v1';
  if (profile === 'drawdown_rebound_reward_train_v1') return 'DD rebound v1';
  if (profile === 'balanced_rule_residual_imitation_v4') return 'BRR imitation v4';
  if (profile === 'balanced_macro_v1') return 'Balanced macro v1';
  const balanced = profile.match(/^balanced_rule_residual_(.+)$/);
  if (balanced) return `BRR ${balanced[1].replaceAll('_', ' ')}`;
  return profile;
}

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
  return filled.slice(0, 6);
}

export function selectDefaultChartProfile(
  selectedProfiles: readonly string[],
  models: Record<string, ModelPayload>,
  fallbackProfile: string,
): string {
  if (selectedProfiles.includes(fallbackProfile) && models[fallbackProfile]) return fallbackProfile;
  const actorProfile = selectedProfiles.find((profile) => models[profile]?.status !== 'reference');
  if (actorProfile) return actorProfile;
  return selectedProfiles[0] ?? fallbackProfile;
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

export function computeDefaultDateRange(input: {
  manifest: Manifest;
  marketRows: readonly MarketRow[];
  models: Record<string, ModelPayload>;
  comparison: ComparisonPayload;
}): { start: string; end: string } | null {
  const modelList = Object.values(input.models);
  const replayableModels = modelList.filter((model) => model.status !== 'reference' && model.rows.length > 0);
  const modelsForWindow = replayableModels.length ? replayableModels : modelList;
  const common = computeCommonWindow(
    input.marketRows,
    modelsForWindow.map((model) => model.rows),
  );
  if (!common) {
    return input.comparison.rows.length
      ? { start: input.comparison.common_start, end: input.comparison.common_end }
      : null;
  }
  const marketDates = sortedDates(input.marketRows);
  const end = marketDates.includes(input.manifest.latest_market_date)
    ? input.manifest.latest_market_date
    : marketDates[marketDates.length - 1] ?? common.end;
  const endYear = end.slice(0, 4);
  const ytdDates = marketDates.filter((date) => date >= common.start && date <= end);
  const ytdStart = ytdDates.length ? snapStart(ytdDates, `${endYear}-01-01`) : common.start;
  const visibleDates = marketDates.filter((date) => date >= ytdStart && date <= end);
  const cappedStart =
    visibleDates.length > DEFAULT_VISIBLE_TRADING_DAYS
      ? visibleDates[visibleDates.length - DEFAULT_VISIBLE_TRADING_DAYS]
      : ytdStart;
  return { start: cappedStart <= end ? cappedStart : common.start, end };
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

export function computeFocusedDateRange(
  rows: readonly { date: string }[],
  targetDate: string,
  radius = 10,
): { start: string; end: string } | null {
  const dates = sortedDates(rows);
  if (dates.length === 0) return null;
  const fallback = targetDate < dates[0] ? 0 : dates.length - 1;
  const targetIndex = nearestIndex(dates, targetDate, fallback);
  const width = Math.max(0, Math.floor(radius));
  const startIndex = Math.max(0, targetIndex - width);
  const endIndex = Math.min(dates.length - 1, targetIndex + width);
  return { start: dates[startIndex], end: dates[endIndex] };
}

export function computeDateRangeFromLogicalRange(
  rows: readonly { date: string }[],
  range: { from: number; to: number } | null,
): { start: string; end: string } | null {
  if (!range) return null;
  const dates = sortedDates(rows);
  if (dates.length === 0) return null;
  const startIndex = Math.max(0, Math.min(dates.length - 1, Math.floor(range.from)));
  const endIndex = Math.max(0, Math.min(dates.length - 1, Math.ceil(range.to)));
  const start = Math.min(startIndex, endIndex);
  const end = Math.max(startIndex, endIndex);
  return { start: dates[start], end: dates[end] };
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

function markerPositionForAction(action: string): ActionMarker['position'] {
  return action === 'BUY' ? 'belowBar' : 'aboveBar';
}

function isMaterialPositionDifference(positionDelta: number | undefined): boolean {
  return typeof positionDelta === 'number' && Number.isFinite(positionDelta) && Math.abs(positionDelta) >= MATERIAL_POSITION_DELTA;
}

export function buildModelDifferenceMarkers(
  comparison: ComparisonPayload,
  profiles: readonly string[],
  start: string,
  end: string,
  limit = 8,
): ActionMarker[] {
  const selectedProfiles = profiles.filter(
    (profile, index) =>
      profile !== comparison.baseline_profile &&
      profiles.indexOf(profile) === index &&
      comparison.compared_profiles.includes(profile),
  );
  const grouped = new Map<
    string,
    {
      time: string;
      action: string;
      count: number;
    }
  >();
  const previousByProfile = new Map<string, { active: boolean; action: string; delta: number }>();
  for (const row of comparison.rows) {
    if (row.date < start || row.date > end) continue;
    const baseline = row.profiles[comparison.baseline_profile];
    if (!baseline) continue;
    for (const profile of selectedProfiles) {
      const value = row.profiles[profile];
      if (!value) continue;
      const delta = Number(value.position_delta_vs_baseline);
      const active = Boolean(value.action_mismatch);
      if (!active) {
        previousByProfile.set(profile, { active: false, action: value.action, delta });
        continue;
      }
      const markerAction = value.action;
      const previous = previousByProfile.get(profile);
      const shouldMark =
        !previous?.active ||
        previous.action !== markerAction;
      previousByProfile.set(profile, { active: true, action: markerAction, delta });
      if (!shouldMark) continue;
      const key = `${row.date}:${markerAction}`;
      const current = grouped.get(key);
      if (current) {
        current.count += 1;
      } else {
        grouped.set(key, { time: row.date, action: markerAction, count: 1 });
      }
    }
  }
  return [...grouped.values()]
    .sort((left, right) => {
      const dateOrder = left.time.localeCompare(right.time);
      if (dateOrder !== 0) return dateOrder;
      return left.action.localeCompare(right.action);
    })
    .map((marker) => ({
      time: marker.time,
      position: markerPositionForAction(marker.action),
      color: DIFF_MARKER_COLOR,
      shape: 'circle' as const,
      text: '',
    }))
    .slice(-limit);
}

export function computeVisibleComparison(
  comparison: ComparisonPayload,
  start: string,
  end: string,
): ComparisonRow[] {
  return comparison.rows.filter((row) => row.date >= start && row.date <= end);
}

const MISMATCH_FEATURE_KEYS = ['rsi6', 'spread', 'vix', 'vix_zscore', 'panic_score', 'volatility'] as const;

export function computeActionMismatchRows(
  comparison: ComparisonPayload,
  market: MarketPayload,
  start: string,
  end: string,
  limit = 40,
): ActionMismatchRow[] {
  const visible = computeVisibleComparison(comparison, start, end);
  const marketByDate = new Map(market.rows.map((row) => [row.date, row]));
  const rows: ActionMismatchRow[] = [];
  for (const row of visible) {
    const baseline = row.profiles[comparison.baseline_profile];
    if (!baseline) continue;
    const marketRow = marketByDate.get(row.date);
    const features = Object.fromEntries(
      MISMATCH_FEATURE_KEYS.flatMap((key) => {
        const value = marketRow?.[key];
        return value === undefined || value === null ? [] : [[key, String(value)]];
      }),
    );
    for (const profile of comparison.compared_profiles) {
      if (profile === comparison.baseline_profile) continue;
      const value = row.profiles[profile];
      if (!value?.action_mismatch && !isMaterialPositionDifference(value?.position_delta_vs_baseline)) continue;
      rows.push({
        date: row.date,
        profile,
        model_action: value.action,
        baseline_action: baseline.action,
        model_position: value.position,
        baseline_position: baseline.position,
        position_delta_vs_baseline: value.position_delta_vs_baseline,
        nav_delta_vs_baseline: value.nav_delta_vs_baseline,
        close: typeof marketRow?.close === 'number' ? marketRow.close : null,
        features,
      });
    }
  }
  return rows.slice(-limit);
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function latestModelRowAtOrBefore(rows: readonly ModelRow[], date: string): ModelRow | undefined {
  return [...rows].filter((row) => row.date <= date).sort((a, b) => a.date.localeCompare(b.date)).at(-1);
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
  const marketRow = input.market.rows.find((row) => row.date === input.date);
  const comparisonRow = input.comparison.rows.find((row) => row.date === input.date);
  const rows: InspectorRow[] = [
    { label: 'date', value: input.date },
    { label: 'safe end', value: input.manifest.safe_backtest_end, tone: 'muted' },
  ];

  if (marketRow) {
    rows.push({ label: 'SOXL open', value: formatNumber(Number(marketRow.open)) });
    rows.push({ label: 'SOXL high', value: formatNumber(Number(marketRow.high)) });
    rows.push({ label: 'SOXL low', value: formatNumber(Number(marketRow.low)) });
    rows.push({ label: 'SOXL close', value: formatNumber(Number(marketRow.close)) });
    rows.push({ label: 'volume', value: formatNumber(Number(marketRow.volume), 0), tone: 'muted' });
    for (const key of ['vix', 'rsi', 'rsi6', 'volatility', 'spread', 'vix_zscore', 'panic_score']) {
      if (marketRow[key] !== undefined && marketRow[key] !== null) {
        rows.push({ label: key, value: String(marketRow[key]) });
      }
    }
  }

  for (const [profile, model] of Object.entries(input.models)) {
    const label = formatProfileLabel(profile);
    const signal = model.latest_signal?.date === input.date ? model.latest_signal : null;
    const modelRow = signal ?? model.rows.find((row) => row.date === input.date) ?? latestModelRowAtOrBefore(model.rows, input.date);
    if (modelRow) {
      if (signal) {
        rows.push({ label: `${label} signal`, value: signal.source, tone: 'warning' });
        rows.push({ label: `${label} replay as-of`, value: signal.replay_asof, tone: 'muted' });
      } else if (modelRow.date !== input.date) {
        rows.push({ label: `${label} as-of`, value: modelRow.date, tone: 'muted' });
      }
      rows.push({ label: `${label} action`, value: modelRow.action, tone: rowToneForAction(modelRow.action) });
      rows.push({ label: `${label} position`, value: formatNumber(modelRow.position, 3) });
      if (!signal && 'strategy_nav' in modelRow) {
        rows.push({ label: `${label} NAV`, value: formatNumber(modelRow.strategy_nav, 2) });
      }
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
