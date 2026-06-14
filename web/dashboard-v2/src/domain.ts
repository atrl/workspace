import type {
  ActionMarker,
  ActionFeatureRow,
  BaselineFeatureContrastRow,
  BuySellFeatureBaselineContrastRow,
  ComparisonPayload,
  ComparisonRow,
  DecisionFeatureCard,
  FeatureBaselineSchemaRow,
  FeatureActionSummaryRow,
  FeatureSchemaOverlapRow,
  FeatureDeltaValue,
  FeatureNameValue,
  InspectorRow,
  Manifest,
  MarketPayload,
  MarketRow,
  ModelEvent,
  ModelPayload,
  ModelRow,
  FeatureValue,
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
  const runnable = preferred.find((profile) => canRunSelectedWindow(models[profile]));
  return runnable ?? preferred[0] ?? manifest.default_baseline_profile;
}

export function runnableProfileReason(model: ModelPayload | undefined): string {
  if (!model) return 'model data not loaded';
  if (model.runnable) return 'ready for selected-window inference';
  if (model.runnable_reason === 'reference_baseline' || model.source_kind === 'soxlbacktest_qlib_buysell') {
    return 'deterministic baseline selected-window replay';
  }
  if (model.runnable_reason === 'evidence_only_retained_dashboard_export') {
    return 'retained export only';
  }
  if (model.runnable_reason === 'missing_actor_artifact' || model.runnable_reason === 'missing_actor') {
    return 'missing actor artifact';
  }
  return model.runnable_reason ? model.runnable_reason.replaceAll('_', ' ') : 'not runnable';
}

export function canRunSelectedWindow(model: ModelPayload | undefined): boolean {
  if (!model) return false;
  if (model.runnable) return true;
  return model.runnable_reason === 'reference_baseline' || model.source_kind === 'soxlbacktest_qlib_buysell';
}

export function chartRangeRows(
  models: Record<string, ModelPayload>,
  chartProfile: string,
  marketRows: readonly MarketRow[],
): readonly { date: string }[] {
  const rows = models[chartProfile]?.rows;
  return rows && rows.length ? rows : marketRows;
}

export function selectableChartProfiles(
  models: Record<string, ModelPayload>,
  selectedProfiles: readonly string[],
): string[] {
  return selectedProfiles.filter((profile) => Boolean(models[profile]?.rows.length));
}

export function selectableRunProfiles(
  models: Record<string, ModelPayload>,
  availableProfiles: readonly string[] = Object.keys(models),
): string[] {
  return availableProfiles.filter((profile) => canRunSelectedWindow(models[profile]));
}

export function chooseDefaultRunProfile(
  manifest: Manifest,
  models: Record<string, ModelPayload>,
  chartProfile?: string,
): string {
  const preferred = [
    chartProfile,
    manifest.default_chart_profile,
    manifest.default_baseline_profile,
    ...manifest.default_comparison_profiles,
    ...manifest.available_profiles,
  ].filter((profile): profile is string => typeof profile === 'string' && profile in models);
  return preferred.find((profile) => canRunSelectedWindow(models[profile])) ?? selectableRunProfiles(models, manifest.available_profiles)[0] ?? '';
}

export function clipRowsByDate<T extends { date: string }>(rows: readonly T[], start: string, end: string): T[] {
  return rows.filter((row) => row.date >= start && row.date <= end);
}

export type DateRangePreset = '1M' | '3M' | '6M' | 'YTD' | '1Y' | 'ALL';
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

function snapStartNearest(dates: readonly string[], target: string): string {
  const targetTime = Date.parse(`${target}T00:00:00Z`);
  if (!Number.isFinite(targetTime)) return dates[0];
  let best = dates[0];
  let bestDistance = Math.abs(Date.parse(`${best}T00:00:00Z`) - targetTime);
  for (const date of dates.slice(1)) {
    const distance = Math.abs(Date.parse(`${date}T00:00:00Z`) - targetTime);
    if (distance < bestDistance) {
      best = date;
      bestDistance = distance;
    }
  }
  return best;
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
      : formatDate(addMonths(endDate, { '1M': -1, '3M': -3, '6M': -6, '1Y': -12 }[preset]));
  return { start: preset === 'YTD' ? snapStart(dates, target) : snapStartNearest(dates, target), end };
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
      const rowSnapshot = modelRow.feature_snapshot ?? {};
      const eventSnapshot = eventSnapshotFor(model, input.date);
      const marketSnapshot = snapshotFromMarket(market.get(input.date), model.snapshot_columns);
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

function modelRowsByDate(rows: readonly ModelRow[]): Map<string, ModelRow> {
  return new Map(rows.map((row) => [row.date, row]));
}

function numericFeatureValue(value: FeatureValue | undefined): number | null {
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value;
}

function featureDelta(value: FeatureValue | undefined, baselineValue: FeatureValue | undefined): number | null {
  const left = numericFeatureValue(value);
  const right = numericFeatureValue(baselineValue);
  return left !== null && right !== null ? left - right : null;
}

function featureEntries(
  features: Record<string, FeatureValue>,
  featureNames: readonly string[],
  limit: number,
): FeatureNameValue[] {
  return featureNames
    .filter((feature) => features[feature] !== undefined)
    .sort()
    .slice(0, limit)
    .map((feature) => ({ feature, value: features[feature] ?? null }));
}

function topFeatureDeltas(
  features: Record<string, FeatureValue>,
  baselineFeatures: Record<string, FeatureValue>,
  limit: number,
): FeatureDeltaValue[] {
  return Object.keys(features)
    .filter((feature) => baselineFeatures[feature] !== undefined)
    .map((feature) => ({
      feature,
      value: features[feature] ?? null,
      baseline_value: baselineFeatures[feature] ?? null,
      delta: featureDelta(features[feature], baselineFeatures[feature]),
    }))
    .sort((left, right) => {
      const leftMissing = left.delta === null ? 1 : 0;
      const rightMissing = right.delta === null ? 1 : 0;
      return (
        leftMissing - rightMissing ||
        Math.abs(right.delta ?? 0) - Math.abs(left.delta ?? 0) ||
        left.feature.localeCompare(right.feature)
      );
    })
    .slice(0, limit);
}

function modelRowAtOrBefore(rows: readonly ModelRow[], date: string): ModelRow | undefined {
  const sorted = [...rows].sort((left, right) => left.date.localeCompare(right.date));
  return [...sorted].reverse().find((row) => row.date <= date) ?? sorted[0];
}

export function buildSelectedDateDecisionFeatureCards(input: {
  date: string;
  models: Record<string, ModelPayload>;
  marketRows: readonly MarketRow[];
  baselineProfile: string;
  modelOnlyLimit?: number;
  deltaLimit?: number;
  buySellLimit?: number;
}): DecisionFeatureCard[] {
  const modelOnlyLimit = input.modelOnlyLimit ?? 4;
  const deltaLimit = input.deltaLimit ?? 4;
  const buySellLimit = input.buySellLimit ?? 4;
  const market = marketRowsByDate(input.marketRows);
  const rows = Object.entries(input.models)
    .map(([profile, model]) => {
      const modelRow = modelRowAtOrBefore(model.rows, input.date);
      if (!modelRow) return null;
      const rowSnapshot = modelRow.feature_snapshot ?? {};
      const marketSnapshot = snapshotFromMarket(market.get(modelRow.date), model.snapshot_columns);
      return {
        profile,
        display_name: model.display_name || profile,
        date: modelRow.date,
        action: modelRow.action,
        position: modelRow.position,
        is_trade: modelRow.action === 'BUY' || modelRow.action === 'SELL',
        source_kind: model.source_kind,
        feature_packs: model.feature_packs,
        feature_columns: model.feature_columns,
        features: Object.keys(rowSnapshot).length ? rowSnapshot : marketSnapshot,
      };
    })
    .filter((row): row is ActionFeatureRow & { date: string } => row !== null);
  const rowsByProfile = new Map(rows.map((row) => [row.profile, row]));
  const baselineRow = rowsByProfile.get(input.baselineProfile);
  const baselineFeatures = baselineRow?.features ?? {};
  const featureSets = new Map(
    Object.entries(input.models).map(([profile, model]) => [
      profile,
      new Set(model.feature_columns.length ? model.feature_columns : model.snapshot_columns),
    ]),
  );

  return rows.map((row) => {
    const model = input.models[row.profile];
    const featureSet = featureSets.get(row.profile) ?? new Set<string>();
    const otherFeatures = new Set<string>();
    for (const [otherProfile, otherSet] of featureSets.entries()) {
      if (otherProfile === row.profile) continue;
      for (const feature of otherSet) otherFeatures.add(feature);
    }
    const snapshotFeatureNames = Object.keys(row.features);
    const modelOnlyNames = snapshotFeatureNames.filter((feature) => {
      const schemaOnly = featureSet.has(feature) && !otherFeatures.has(feature);
      const absentFromBaseline = row.profile !== input.baselineProfile && baselineFeatures[feature] === undefined;
      return schemaOnly || absentFromBaseline;
    });
    const topBuySell = [...(model.feature_action_summary ?? [])]
      .sort((left, right) => {
        const leftScore = Math.abs(left.buy_minus_sell ?? 0);
        const rightScore = Math.abs(right.buy_minus_sell ?? 0);
        return rightScore - leftScore || left.feature.localeCompare(right.feature);
      })
      .slice(0, buySellLimit)
      .map((item) => item.feature);

    return {
      profile: row.profile,
      display_name: row.display_name,
      date: row.date,
      is_exact_date: row.date === input.date,
      action: row.action,
      position: row.position,
      baseline_action: baselineRow?.action ?? 'n/a',
      position_delta_vs_baseline:
        baselineRow === undefined ? null : row.position - baselineRow.position,
      is_trade: row.is_trade,
      source_kind: row.source_kind,
      model_only_features: featureEntries(row.features, modelOnlyNames, modelOnlyLimit),
      shared_feature_deltas:
        row.profile === input.baselineProfile
          ? []
          : topFeatureDeltas(row.features, baselineFeatures, deltaLimit),
      buy_sell_features: topBuySell,
    } satisfies DecisionFeatureCard;
  });
}

export function buildBaselineFeatureContrastRows(input: {
  models: Record<string, ModelPayload>;
  baselineProfile: string;
  limitPerEvent?: number;
  maxRows?: number;
}): BaselineFeatureContrastRow[] {
  const baseline = input.models[input.baselineProfile];
  if (!baseline) return [];
  const limitPerEvent = input.limitPerEvent ?? 5;
  const maxRows = input.maxRows ?? 48;
  const baselineRows = modelRowsByDate(baseline.rows);
  const contrastRows: BaselineFeatureContrastRow[] = [];

  for (const [profile, model] of Object.entries(input.models)) {
    if (profile === input.baselineProfile) continue;
    for (const row of model.rows) {
      if (row.action !== 'BUY' && row.action !== 'SELL') continue;
      const baselineRow = baselineRows.get(row.date);
      const snapshot = row.feature_snapshot ?? {};
      const baselineSnapshot = baselineRow?.feature_snapshot ?? {};
      const featureNames = [...new Set([...Object.keys(snapshot), ...Object.keys(baselineSnapshot)])];
      const featureRows = featureNames
        .map((feature) => {
          const value = snapshot[feature] ?? null;
          const baselineValue = baselineSnapshot[feature] ?? null;
          return {
            date: row.date,
            profile,
            display_name: model.display_name || profile,
            action: row.action,
            baseline_action: baselineRow?.action ?? 'n/a',
            position_delta_vs_baseline:
              baselineRow === undefined ? null : row.position - baselineRow.position,
            feature,
            value,
            baseline_value: baselineValue,
            delta: featureDelta(value, baselineValue),
          } satisfies BaselineFeatureContrastRow;
        })
        .sort((left, right) => {
          const leftScore = Math.abs(left.delta ?? 0);
          const rightScore = Math.abs(right.delta ?? 0);
          const missingDeltaOrder = Number(left.delta === null) - Number(right.delta === null);
          return missingDeltaOrder || rightScore - leftScore || left.feature.localeCompare(right.feature);
        })
        .slice(0, limitPerEvent);
      contrastRows.push(...featureRows);
    }
  }

  return contrastRows
    .sort((left, right) => right.date.localeCompare(left.date) || left.display_name.localeCompare(right.display_name))
    .slice(0, maxRows);
}

function modelRuntimeLabel(model: ModelPayload): string {
  if (model.runnable) return 'runnable';
  if (model.runnable_reason === 'reference_baseline' || model.source_kind === 'soxlbacktest_qlib_buysell') {
    return 'reference baseline';
  }
  return model.runnable_reason ? model.runnable_reason.replaceAll('_', ' ') : 'evidence only';
}

function topBuySellFeatureLabels(model: ModelPayload, limit: number): string[] {
  return [...(model.feature_action_summary ?? [])]
    .sort((left, right) => {
      const leftScore = Math.abs(left.buy_minus_sell ?? 0);
      const rightScore = Math.abs(right.buy_minus_sell ?? 0);
      return rightScore - leftScore || left.feature.localeCompare(right.feature);
    })
    .slice(0, limit)
    .map((row) => row.feature);
}

function featureSetFor(model: ModelPayload | undefined): Set<string> {
  if (!model) return new Set();
  return new Set(model.feature_columns.length ? model.feature_columns : model.snapshot_columns);
}

function topBuySellFeatureLabelsWithValues(model: ModelPayload, limit: number): string[] {
  return [...(model.feature_action_summary ?? [])]
    .sort((left, right) => {
      const leftScore = Math.abs(left.buy_minus_sell ?? 0);
      const rightScore = Math.abs(right.buy_minus_sell ?? 0);
      return rightScore - leftScore || left.feature.localeCompare(right.feature);
    })
    .slice(0, limit)
    .map((row) => {
      const delta = row.buy_minus_sell === null ? 'n/a' : row.buy_minus_sell.toFixed(2);
      return `${row.feature} ${delta}`;
    });
}

export function buildFeatureBaselineSchemaRows(
  models: Record<string, ModelPayload>,
  baselineProfile: string,
  options: { featureNameLimit?: number; topBuySellLimit?: number } = {},
): FeatureBaselineSchemaRow[] {
  const entries = Object.entries(models);
  const baseline = models[baselineProfile] ?? entries[0]?.[1];
  if (!baseline) return [];
  const baselineSet = featureSetFor(baseline);
  const featureNameLimit = options.featureNameLimit ?? 6;
  const topBuySellLimit = options.topBuySellLimit ?? 4;

  return entries.map(([profile, model]) => {
    const featureSet = featureSetFor(model);
    const shared = [...featureSet].filter((feature) => baselineSet.has(feature)).sort();
    const modelOnly = [...featureSet].filter((feature) => !baselineSet.has(feature)).sort();
    const baselineOnly = profile === baselineProfile
      ? []
      : [...baselineSet].filter((feature) => !featureSet.has(feature)).sort();
    return {
      profile,
      display_name: model.display_name || profile,
      runtime: modelRuntimeLabel(model),
      feature_schema_signature: model.feature_schema_signature ?? 'n/a',
      feature_schema_source: model.feature_schema_source ?? 'n/a',
      feature_count: featureSet.size,
      snapshot_count: model.snapshot_columns.length,
      shared_count: shared.length,
      model_only_count: modelOnly.length,
      baseline_only_count: baselineOnly.length,
      shared_features: shared.slice(0, featureNameLimit),
      model_only_features: modelOnly.slice(0, featureNameLimit),
      baseline_only_features: baselineOnly.slice(0, featureNameLimit),
      top_buy_sell_features: topBuySellFeatureLabelsWithValues(model, topBuySellLimit),
    };
  });
}

function summaryByFeature(model: ModelPayload | undefined): Map<string, FeatureActionSummaryRow> {
  return new Map((model?.feature_action_summary ?? []).map((row) => [row.feature, row]));
}

function topSummaryFeatureNames(model: ModelPayload | undefined, limit: number): string[] {
  return [...(model?.feature_action_summary ?? [])]
    .sort((left, right) => {
      const leftScore = Math.abs(left.buy_minus_sell ?? 0);
      const rightScore = Math.abs(right.buy_minus_sell ?? 0);
      const missingOrder = Number(left.buy_minus_sell === null) - Number(right.buy_minus_sell === null);
      return missingOrder || rightScore - leftScore || left.feature.localeCompare(right.feature);
    })
    .slice(0, limit)
    .map((row) => row.feature);
}

function nullableDelta(left: number | null | undefined, right: number | null | undefined): number | null {
  return typeof left === 'number' && Number.isFinite(left) && typeof right === 'number' && Number.isFinite(right)
    ? left - right
    : null;
}

export function buildBuySellFeatureBaselineContrastRows(
  models: Record<string, ModelPayload>,
  baselineProfile: string,
  options: { limitPerModel?: number; baselineFeatureLimit?: number } = {},
): BuySellFeatureBaselineContrastRow[] {
  const entries = Object.entries(models);
  const baseline = models[baselineProfile] ?? entries[0]?.[1];
  if (!baseline) return [];
  const limitPerModel = options.limitPerModel ?? 6;
  const baselineFeatureLimit = options.baselineFeatureLimit ?? 4;
  const baselineSet = featureSetFor(baseline);
  const baselineSummary = summaryByFeature(baseline);
  const baselineTop = topSummaryFeatureNames(baseline, baselineFeatureLimit);
  const rows: BuySellFeatureBaselineContrastRow[] = [];

  for (const [profile, model] of entries) {
    if (profile === baselineProfile) continue;
    const modelSet = featureSetFor(model);
    const modelSummary = summaryByFeature(model);
    const featureNames = [...new Set([...topSummaryFeatureNames(model, limitPerModel), ...baselineTop])];
    for (const feature of featureNames) {
      const modelRow = modelSummary.get(feature);
      const baselineRow = baselineSummary.get(feature);
      const inModel = modelSet.has(feature);
      const inBaseline = baselineSet.has(feature);
      const presence = inModel && inBaseline ? 'shared' : inModel ? 'model_only' : 'baseline_only';
      rows.push({
        profile,
        display_name: model.display_name || profile,
        feature,
        presence,
        model_buy_count: modelRow?.buy_count ?? 0,
        model_sell_count: modelRow?.sell_count ?? 0,
        model_buy_mean: modelRow?.buy_mean ?? null,
        model_sell_mean: modelRow?.sell_mean ?? null,
        model_buy_minus_sell: modelRow?.buy_minus_sell ?? null,
        baseline_buy_count: baselineRow?.buy_count ?? 0,
        baseline_sell_count: baselineRow?.sell_count ?? 0,
        baseline_buy_mean: baselineRow?.buy_mean ?? null,
        baseline_sell_mean: baselineRow?.sell_mean ?? null,
        baseline_buy_minus_sell: baselineRow?.buy_minus_sell ?? null,
        buy_mean_delta: nullableDelta(modelRow?.buy_mean, baselineRow?.buy_mean),
        sell_mean_delta: nullableDelta(modelRow?.sell_mean, baselineRow?.sell_mean),
        driver_delta_vs_baseline: nullableDelta(modelRow?.buy_minus_sell, baselineRow?.buy_minus_sell),
      });
    }
  }

  return rows.sort((left, right) => {
    const leftScore = Math.abs(left.driver_delta_vs_baseline ?? left.model_buy_minus_sell ?? 0);
    const rightScore = Math.abs(right.driver_delta_vs_baseline ?? right.model_buy_minus_sell ?? 0);
    const presenceOrder = { model_only: 0, shared: 1, baseline_only: 2 } as const;
    return (
      left.display_name.localeCompare(right.display_name) ||
      presenceOrder[left.presence] - presenceOrder[right.presence] ||
      rightScore - leftScore ||
      left.feature.localeCompare(right.feature)
    );
  });
}

export function buildFeatureSchemaOverlapRows(
  models: Record<string, ModelPayload>,
  options: { featureNameLimit?: number; topBuySellLimit?: number } = {},
): FeatureSchemaOverlapRow[] {
  const entries = Object.entries(models);
  const featureNameLimit = options.featureNameLimit ?? 8;
  const topBuySellLimit = options.topBuySellLimit ?? 4;
  const featureSets = new Map(
    entries.map(([profile, model]) => [profile, new Set(model.feature_columns.length ? model.feature_columns : model.snapshot_columns)]),
  );

  return entries.map(([profile, model]) => {
    const featureSet = featureSets.get(profile) ?? new Set<string>();
    const otherFeatures = new Set<string>();
    for (const [otherProfile, otherSet] of featureSets.entries()) {
      if (otherProfile === profile) continue;
      for (const feature of otherSet) otherFeatures.add(feature);
    }
    const shared = [...featureSet].filter((feature) => otherFeatures.has(feature)).sort();
    const modelOnly = [...featureSet].filter((feature) => !otherFeatures.has(feature)).sort();
    return {
      profile,
      display_name: model.display_name || profile,
      source_kind: model.source_kind,
      runtime: modelRuntimeLabel(model),
      feature_schema_signature: model.feature_schema_signature ?? 'n/a',
      feature_schema_source: model.feature_schema_source ?? 'n/a',
      feature_count: featureSet.size,
      snapshot_count: model.snapshot_columns.length,
      shared_count: shared.length,
      model_only_count: modelOnly.length,
      shared_features: shared.slice(0, featureNameLimit),
      model_only_features: modelOnly.slice(0, featureNameLimit),
      top_buy_sell_features: topBuySellFeatureLabels(model, topBuySellLimit),
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

function formatPercent(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : 'n/a';
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
