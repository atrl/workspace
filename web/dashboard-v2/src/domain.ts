import type {
  ActionMarker,
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

export function clipRowsByDate<T extends { date: string }>(rows: readonly T[], start: string, end: string): T[] {
  return rows.filter((row) => row.date >= start && row.date <= end);
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

export function computeVisibleComparison(
  comparison: ComparisonPayload,
  start: string,
  end: string,
): ComparisonRow[] {
  return comparison.rows.filter((row) => row.date >= start && row.date <= end);
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
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
    rows.push({ label: 'SOXL close', value: formatNumber(Number(marketRow.close)) });
    rows.push({ label: 'volume', value: formatNumber(Number(marketRow.volume), 0), tone: 'muted' });
    for (const key of ['vix', 'rsi', 'rsi6', 'volatility', 'spread', 'vix_zscore', 'panic_score']) {
      if (marketRow[key] !== undefined && marketRow[key] !== null) {
        rows.push({ label: key, value: String(marketRow[key]) });
      }
    }
  }

  for (const [profile, model] of Object.entries(input.models)) {
    const modelRow = model.rows.find((row) => row.date === input.date);
    if (modelRow) {
      rows.push({ label: `${profile} action`, value: modelRow.action, tone: rowToneForAction(modelRow.action) });
      rows.push({ label: `${profile} position`, value: formatNumber(modelRow.position, 3) });
      rows.push({ label: `${profile} NAV`, value: formatNumber(modelRow.strategy_nav, 2) });
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
