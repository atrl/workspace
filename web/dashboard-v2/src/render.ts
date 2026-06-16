import { computeVisibleComparison, formatProfileLabel } from './domain';
import type {
  ActionMismatchRow,
  ComparisonPayload,
  InspectorRow,
  LatestSignalDifference,
  Manifest,
  ModelPayload,
} from './types';

const MODEL_COLORS = ['#39b7c5', '#a597ff', '#e1aa3e', '#1fb981', '#f472b6', '#f97316'];
const MATERIAL_POSITION_DELTA = 0.05;

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatPct(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)}%` : 'n/a';
}

function formatCompactPct(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${formatSigned(value, 2)}%` : 'n/a';
}

function formatSigned(value: number, digits: number): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(digits)}`;
}

function isMaterialPositionDifference(value: number | undefined): boolean {
  return typeof value === 'number' && Number.isFinite(value) && Math.abs(value) >= MATERIAL_POSITION_DELTA;
}

function readableProfile(profile: string): string {
  return formatProfileLabel(profile).replaceAll('_', ' ');
}

function profileLabel(profile: string, manifest?: Manifest): string {
  return manifest?.profile_labels?.[profile] ?? formatProfileLabel(profile);
}

function profileSourceKind(profile: string, manifest?: Manifest): string {
  return manifest?.profile_source_kinds?.[profile] ?? manifest?.profile_summaries?.[profile]?.source_kind ?? '';
}

function maxDate(values: readonly (string | undefined | null)[]): string | null {
  const dates = values.filter((value): value is string => Boolean(value)).sort();
  return dates.at(-1) ?? null;
}

function freshnessCard(label: string, value: string, detail?: string, tone?: 'latest' | 'safe' | 'stale'): HTMLElement {
  const card = element('div', 'freshness-card');
  if (tone) card.dataset.freshness = tone;
  card.appendChild(element('span', 'freshness-card-label', label));
  card.appendChild(element('strong', 'freshness-card-value', value));
  if (detail) card.appendChild(element('span', 'freshness-card-detail', detail));
  return card;
}

function freshnessLabel(replayEnd: string | undefined, commonEnd: string, manifest?: Manifest): {
  label: string;
  status: 'latest' | 'safe' | 'stale';
  gapLabel?: string;
} {
  const latestMarketDate = manifest?.latest_market_date;
  if (replayEnd && latestMarketDate && replayEnd < latestMarketDate) {
    const isSafeCut = manifest?.safe_backtest_end && replayEnd <= manifest.safe_backtest_end;
    return {
      label: `${isSafeCut ? 'Cut' : 'Stale'} ${replayEnd}`,
      status: 'stale',
      gapLabel: `Price gap ${latestMarketDate}`,
    };
  }
  if (replayEnd && replayEnd > commonEnd) {
    return { label: `Latest ${replayEnd}`, status: 'latest' };
  }
  if (replayEnd && latestMarketDate && replayEnd >= latestMarketDate) {
    return { label: `Latest ${replayEnd}`, status: 'latest' };
  }
  return { label: `Common ${commonEnd}`, status: 'safe' };
}

function latestMismatchForProfile(
  mismatchRows: readonly ActionMismatchRow[],
  profile: string,
): ActionMismatchRow | undefined {
  return [...mismatchRows].reverse().find((row) => row.profile === profile);
}

function latestModelRowAtOrBefore(model: ModelPayload | undefined, date: string): ModelPayload['rows'][number] | null {
  if (!model) return null;
  return [...model.rows].filter((row) => row.date <= date).sort((a, b) => a.date.localeCompare(b.date)).at(-1) ?? null;
}

function visibleLatestSignalDifferences(
  comparison: ComparisonPayload,
  profiles: readonly string[],
  start: string,
  end: string,
): LatestSignalDifference[] {
  const allowed = new Set(profiles);
  return (comparison.latest_signal_differences ?? []).filter(
    (row) =>
      row.date >= start &&
      row.date <= end &&
      allowed.has(row.profile) &&
      row.profile !== comparison.baseline_profile,
  );
}

function latestSignalDifferenceRows(
  comparison: ComparisonPayload,
  models: Record<string, ModelPayload>,
  start: string,
  end: string,
): {
  date: string;
  profile: string;
  baselineProfile: string;
  modelAction: string;
  baselineAction: string;
  modelPosition: number;
  baselinePosition: number;
  positionDelta: number;
  replayAsof: string;
}[] {
  const baseline = models[comparison.baseline_profile];
  return comparison.compared_profiles.flatMap((profile) => {
    if (profile === comparison.baseline_profile) return [];
    const signal = models[profile]?.latest_signal;
    if (!signal || signal.date < start || signal.date > end) return [];
    const baselineRow = latestModelRowAtOrBefore(baseline, signal.date);
    if (!baselineRow) return [];
    return [
      {
        date: signal.date,
        profile,
        baselineProfile: comparison.baseline_profile,
        modelAction: signal.action,
        baselineAction: baselineRow.action,
        modelPosition: signal.position,
        baselinePosition: baselineRow.position,
        positionDelta: signal.position - baselineRow.position,
        replayAsof: signal.replay_asof,
      },
    ];
  });
}

interface LatestDecisionRow {
  targetDate: string;
  profile: string;
  isBaseline: boolean;
  action: string;
  position: number;
  positionDelta: number;
  actionDiff: boolean;
  positionDiff: boolean;
  sourceLabel: string;
}

function latestDecisionRows(
  comparison: ComparisonPayload,
  models: Record<string, ModelPayload>,
  profiles: readonly string[],
  start: string,
  end: string,
  manifest?: Manifest,
): LatestDecisionRow[] {
  const targetDate = manifest?.latest_market_date;
  if (!targetDate || targetDate < start || targetDate > end) return [];
  const baseline = models[comparison.baseline_profile];
  const baselineRow = latestModelRowAtOrBefore(baseline, targetDate);
  if (!baselineRow) return [];
  const ordered = [
    comparison.baseline_profile,
    ...profiles.filter(
      (profile, index) =>
        profile !== comparison.baseline_profile &&
        profiles.indexOf(profile) === index &&
        comparison.compared_profiles.includes(profile),
    ),
  ];
  return ordered.flatMap((profile) => {
    const model = models[profile];
    if (!model) return [];
    const signal = model.latest_signal?.date === targetDate ? model.latest_signal : null;
    const row = signal ?? latestModelRowAtOrBefore(model, targetDate);
    if (!row) return [];
    const position = Number(row.position);
    const positionDelta = position - Number(baselineRow.position);
    const isBaseline = profile === comparison.baseline_profile;
    return [
      {
        targetDate,
        profile,
        isBaseline,
        action: row.action,
        position,
        positionDelta,
        actionDiff: !isBaseline && row.action !== baselineRow.action,
        positionDiff: !isBaseline && Math.abs(positionDelta) > 0.0005,
        sourceLabel: signal
          ? `signal as-of ${signal.replay_asof}`
          : row.date === targetDate
            ? 'latest row'
            : `as-of ${row.date}`,
      },
    ];
  });
}

function mismatchFeatureLabel(row: ActionMismatchRow): string {
  const features = Object.entries(row.features)
    .slice(0, 3)
    .map(([key, value]) => `${key} ${formatFeatureValue(value)}`)
    .join(' | ');
  const prefix =
    `Last diff ${row.date} ${row.model_action} vs ${row.baseline_action}` +
    ` | Δpos ${formatSigned(row.position_delta_vs_baseline, 3)}`;
  return features ? `${prefix} | ${features}` : prefix;
}

function formatFeatureValue(value: string): string {
  const number = Number(value);
  if (!Number.isFinite(number)) return value;
  if (Math.abs(number - Math.round(number)) < 0.005) return String(Math.round(number));
  return number.toFixed(2);
}

export function renderDataSourcePanel(container: HTMLElement, manifest: Manifest, warnings: string[] = []): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Data Status'));
  const blockedCount =
    manifest.retention_filter?.blocked_profile_count ?? Object.keys(manifest.blocked_profiles ?? {}).length;
  const visibleActorCount = manifest.default_comparison_profiles.filter(
    (profile) => manifest.profile_summaries?.[profile]?.status !== 'reference',
  ).length;
  const visibleDiagnosticCount = manifest.default_comparison_profiles.filter((profile) =>
    manifest.diagnostic_profiles?.includes(profile),
  ).length;
  if (visibleDiagnosticCount > 0) {
    const banner = element('div', 'gate-banner');
    banner.appendChild(element('strong', undefined, 'Diagnostic actor comparisons'));
    banner.appendChild(element('span', undefined, 'Shown for model-difference review; actors are not final-retained.'));
    container.appendChild(banner);
  } else if (blockedCount > 0 && visibleActorCount === 0) {
    const banner = element('div', 'gate-banner');
    banner.appendChild(element('strong', undefined, 'No final-retained actor'));
    banner.appendChild(element('span', undefined, 'Showing SOXLBacktest reference baseline only.'));
    container.appendChild(banner);
  }
  const profileSummaries = manifest.profile_summaries ?? {};
  const replayFreshness = manifest.replay_freshness;
  const baselineEnd = profileSummaries[manifest.default_baseline_profile]?.replay_end ?? null;
  const baselineLatest = manifest.baseline_latest;
  const actorEnd = maxDate(
    manifest.default_comparison_profiles
      .filter((profile) => profileSummaries[profile]?.status !== 'reference')
      .map((profile) => profileSummaries[profile]?.replay_end),
  );
  const actorSignalEnd = maxDate(
    manifest.default_comparison_profiles
      .filter((profile) => profileSummaries[profile]?.status !== 'reference')
      .map((profile) => profileSummaries[profile]?.latest_signal_date),
  );
  const strip = element('div', 'status-strip');
  const rows: [string, string][] = [
    ['SOXL K-line', manifest.latest_market_date],
    ['Qlib safe replay', manifest.safe_backtest_end],
    ['Baseline', formatProfileLabel(manifest.default_baseline_profile)],
    ['Blocked', String(blockedCount)],
  ];
  if (baselineEnd) {
    const baselineWindow = profileSummaries[manifest.default_baseline_profile]?.window;
    const baselineValue =
      baselineLatest?.date === baselineEnd
        ? `${baselineWindow ? `${baselineWindow} ` : ''}${baselineEnd} ${formatCompactPct(baselineLatest.return_pct)} pos ${
            typeof baselineLatest.position === 'number' ? baselineLatest.position.toFixed(3) : 'n/a'
          }`
        : `${baselineWindow ? `${baselineWindow} ` : ''}${baselineEnd}`;
    rows.splice(3, 0, ['Baseline latest', baselineValue]);
  }
  if (actorEnd) {
    rows.splice(baselineEnd ? 4 : 3, 0, ['Actor replay', actorEnd]);
  }
  if (actorSignalEnd && actorSignalEnd > (actorEnd ?? '')) {
    const blockedIndex = rows.findIndex(([label]) => label === 'Blocked');
    rows.splice(blockedIndex >= 0 ? blockedIndex : rows.length, 0, ['Actor latest signal', actorSignalEnd]);
  }
  if (manifest.canonical_data_date && manifest.canonical_data_date !== manifest.latest_market_date) {
    rows.splice(1, 0, ['Canonical', manifest.canonical_data_date]);
  }
  if (manifest.strict_retention_gate) {
    rows.push([
      'Strict Gate',
      `${manifest.strict_retention_gate.passing_profile_count}/${manifest.strict_retention_gate.profiles_scanned}`,
    ]);
  }
  if (manifest.strict_candidate_retention_filter) {
    rows.push(['Final Retain', String(manifest.strict_candidate_retention_filter.actor_retention_allowed_count)]);
  }
  if (manifest.final_walkforward_retention_contract) {
    rows.push([
      'WF Retain',
      `${manifest.final_walkforward_retention_contract.final_retained_count}/${manifest.final_walkforward_retention_contract.strict_passing_profile_count}`,
    ]);
  }
  if (manifest.latest_signal) {
    rows.push(['Signals', `${manifest.latest_signal.available_count}@${manifest.latest_signal.date}`]);
  }
  if (replayFreshness) {
    const counts = replayFreshness.profile_status_counts;
    rows.push([
      'Replay freshness',
      `latest ${counts.latest} | signal ${counts.latest_signal} | safe ${counts.safe_replay} | stale ${counts.stale}`,
    ]);
  }
  for (const [label, value] of rows) {
    const item = element('div', 'status-pill');
    if (label === 'Blocked' && blockedCount > 0) item.dataset.tone = 'warning';
    if (label === 'Strict Gate' && manifest.strict_retention_gate?.status === 'blocked') item.dataset.tone = 'warning';
    if (label === 'Final Retain' && manifest.strict_candidate_retention_filter?.status === 'blocked') {
      item.dataset.tone = 'warning';
    }
    if (label === 'WF Retain' && manifest.final_walkforward_retention_contract?.status === 'blocked') {
      item.dataset.tone = 'warning';
    }
    if (label === 'Signals' && manifest.latest_signal && !manifest.latest_signal.comparison_included) {
      item.dataset.tone = 'warning';
    }
    if (label === 'Replay freshness' && replayFreshness?.profile_status_counts.stale) item.dataset.tone = 'warning';
    if (label === 'Actor latest signal') item.dataset.tone = 'warning';
    item.appendChild(element('span', 'source-label', label));
    item.appendChild(element('span', 'source-value', value));
    strip.appendChild(item);
  }
  const board = element('div', 'freshness-board');
  const baselineProfile = manifest.default_baseline_profile;
  const baselineLabel = profileLabel(baselineProfile, manifest);
  const baselineDate = baselineLatest?.date ?? baselineEnd ?? 'n/a';
  const baselineDetailParts = [
    baselineLatest?.action ?? null,
    typeof baselineLatest?.position === 'number' ? `pos ${baselineLatest.position.toFixed(3)}` : null,
    typeof baselineLatest?.return_pct === 'number' ? `ret ${formatCompactPct(baselineLatest.return_pct)}` : null,
  ].filter(Boolean);
  board.appendChild(
    freshnessCard(
      'K-line latest',
      manifest.latest_market_date,
      manifest.canonical_data_date && manifest.canonical_data_date !== manifest.latest_market_date
        ? `canonical ${manifest.canonical_data_date}`
        : undefined,
      'latest',
    ),
  );
  board.appendChild(
    freshnessCard(
      'Baseline latest',
      `${baselineLabel} ${baselineDate}`,
      baselineDetailParts.length ? baselineDetailParts.join(' | ') : undefined,
      baselineEnd && baselineEnd < manifest.latest_market_date ? 'stale' : 'latest',
    ),
  );
  board.appendChild(
    freshnessCard(
      'Actor replay',
      actorEnd ?? 'n/a',
      `safe ${manifest.safe_backtest_end}`,
      actorEnd && actorEnd < manifest.latest_market_date ? 'stale' : 'safe',
    ),
  );
  board.appendChild(
    freshnessCard(
      'Actor latest signal',
      actorSignalEnd ?? 'n/a',
      manifest.latest_signal?.available_count
        ? `${manifest.latest_signal.available_count} signals`
        : actorSignalEnd && actorEnd && actorSignalEnd > actorEnd
          ? 'stateless signal'
          : undefined,
      actorSignalEnd && actorSignalEnd >= manifest.latest_market_date ? 'latest' : 'safe',
    ),
  );
  container.appendChild(board);
  container.appendChild(strip);
  const freshnessRows = manifest.default_comparison_profiles
    .map(
      (profile) =>
        [
          profile,
          profileSummaries[profile]?.replay_end,
          profileSummaries[profile]?.latest_signal_date,
        ] as const,
    )
    .filter(([, replayEnd]) => replayEnd);
  if (freshnessRows.length > 0) {
    const freshnessDetails = element('details', 'freshness-details') as HTMLDetailsElement;
    const summaryParts = [
      `Model dates`,
      baselineEnd ? `baseline ${baselineEnd}` : null,
      actorEnd ? `actors ${actorEnd}` : null,
      actorSignalEnd && actorSignalEnd > (actorEnd ?? '') ? `signals ${actorSignalEnd}` : null,
    ].filter(Boolean);
    freshnessDetails.appendChild(element('summary', undefined, summaryParts.join(' | ')));
    const freshness = element('div', 'freshness-strip');
    for (const [profile, replayEnd, signalDate] of freshnessRows) {
      const item = element('span', 'freshness-pill');
      item.dataset.profile = profile;
      if (signalDate && signalDate > String(replayEnd)) {
        item.dataset.freshness = 'cut';
      } else if (replayEnd === manifest.latest_market_date) {
        item.dataset.freshness = 'latest';
      } else {
        item.dataset.freshness = 'common';
      }
      item.appendChild(element('span', 'freshness-name', profileLabel(profile, manifest)));
      const sourceKind = profileSourceKind(profile, manifest);
      if (sourceKind) item.appendChild(element('span', 'freshness-kind', sourceKind.replaceAll('_', ' ')));
      item.appendChild(element('span', 'freshness-date', `replay ${String(replayEnd)}`));
      const replayStatus = manifest.replay_freshness?.profiles?.[profile]?.replay_status;
      const replayLag = manifest.replay_freshness?.profiles?.[profile]?.replay_lag_trading_days;
      if (replayStatus) {
        item.appendChild(
          element(
            'span',
            'freshness-status',
            replayLag ? `${replayStatus.replaceAll('_', ' ')} +${replayLag}d` : replayStatus.replaceAll('_', ' '),
          ),
        );
      }
      if (signalDate && signalDate > String(replayEnd)) {
        item.appendChild(element('span', 'freshness-signal', `signal ${signalDate}`));
      }
      freshness.appendChild(item);
    }
    freshnessDetails.appendChild(freshness);
    container.appendChild(freshnessDetails);
  }
  if (
    (actorEnd && actorEnd < manifest.latest_market_date) ||
    (baselineEnd && baselineEnd < manifest.latest_market_date) ||
    manifest.safe_backtest_end < manifest.latest_market_date
  ) {
    const alert = element('div', 'freshness-alert');
    alert.appendChild(element('strong', undefined, 'Replay split'));
    alert.appendChild(
      element(
        'span',
        undefined,
        [
          `Market ${manifest.latest_market_date}`,
          actorEnd ? `Actors ${actorEnd}` : null,
          actorSignalEnd && actorSignalEnd > (actorEnd ?? '') ? `Actor signals ${actorSignalEnd}` : null,
          baselineEnd ? `Baseline ${baselineEnd}` : null,
          `Safe ${manifest.safe_backtest_end}`,
        ]
          .filter(Boolean)
          .join(' | '),
      ),
    );
    container.appendChild(alert);
  }
  const gate = manifest.strict_retention_gate;
  const gateWarnings = gate
    ? [
        [
          `Strict retention gate: ${gate.passing_profile_count}/${gate.profiles_scanned} passing; ` +
            `requires all WF windows and MaxDD < ${formatPct(gate.max_walkforward_drawdown_pct)}; ` +
            `required return >= ${formatPct(gate.required_best_return_pct)}.`,
        ],
        gate.closest_wf_drawdown_candidate
          ? [
              `Closest WF drawdown: ${readableProfile(gate.closest_wf_drawdown_candidate.profile)} ` +
                `${formatPct(gate.closest_wf_drawdown_candidate.worst_wf_max_drawdown_pct)} ` +
                `(gap ${formatPct(gate.closest_wf_drawdown_candidate.wf_drawdown_gap_pct)}).`,
            ]
          : [],
        gate.closest_ytd_return_candidate
          ? [
              `Closest return alignment: ${readableProfile(gate.closest_ytd_return_candidate.profile)} ` +
                `${formatPct(gate.closest_ytd_return_candidate.ytd_return_pct)}.`,
            ]
          : [],
      ].flat()
    : [];
  const candidateFilter = manifest.strict_candidate_retention_filter;
  const candidateWarnings = candidateFilter
    ? [
        [
          `Candidate retention filter: ${candidateFilter.promotable_profile_count} promotable, ` +
            `${candidateFilter.actor_retention_allowed_count} actor-retained; ` +
            `decision ${candidateFilter.decision ?? candidateFilter.status}.`,
        ],
        candidateFilter.next_mechanism ? [`Next mechanism: ${candidateFilter.next_mechanism}.`] : [],
        [`PPO allowed next: ${String(candidateFilter.ppo_allowed_next)}.`],
      ].flat()
    : [];
  const finalContract = manifest.final_walkforward_retention_contract;
  const finalContractWarnings = finalContract
    ? [
        [
          `Final retention contract: ${finalContract.final_retained_count} retained, ` +
            `reference current ${String(finalContract.reference_is_current)}, ` +
            `baseline ${formatPct(finalContract.latest_reference_return_pct ?? finalContract.reference_return_pct)}; ` +
            `decision ${finalContract.decision ?? finalContract.status}.`,
        ],
        finalContract.global_blockers.length
          ? [`Final contract blockers: ${finalContract.global_blockers.join('; ')}.`]
          : [],
      ].flat()
    : [];
  const blockedWarnings = Object.entries(manifest.blocked_profiles ?? {}).map(
    ([profile, status]) => `${profile}: ${status.reason}`,
  );
  const allWarnings = [...warnings, ...gateWarnings, ...candidateWarnings, ...finalContractWarnings, ...blockedWarnings];
  if (allWarnings.length) {
    const diagnostics = element('details', 'source-diagnostics') as HTMLDetailsElement;
    diagnostics.appendChild(element('summary', undefined, `Diagnostics (${allWarnings.length})`));
    const warningList = element('ul', 'warning-list');
    for (const warning of allWarnings) {
      warningList.appendChild(element('li', undefined, warning));
    }
    diagnostics.appendChild(warningList);
    container.appendChild(diagnostics);
  }
}

export function renderModelDeltaStrip(
  container: HTMLElement,
  comparison: ComparisonPayload,
  start: string,
  end: string,
  profiles: readonly string[] = comparison.compared_profiles,
  manifest?: Manifest,
  mismatchRows: readonly ActionMismatchRow[] = [],
  models: Record<string, ModelPayload> = {},
): void {
  container.replaceChildren();
  const visible = computeVisibleComparison(comparison, start, end);
  const last = visible.at(-1);
  if (!last) {
    container.appendChild(element('span', 'empty-note', 'No comparison rows'));
    return;
  }
  const first = visible[0];
  const summary = element('div', 'model-delta-summary');
  summary.textContent = [
    `${profiles.length} models`,
    `${visible.length} common rows`,
    `Common ${first.date} to ${last.date}`,
    end > last.date ? `Price view to ${end}` : null,
  ]
    .filter(Boolean)
    .join(' | ');
  container.appendChild(summary);
  const caption = element('div', 'model-delta-caption');
  const captionParts = [`Common comparison ${last.date}`];
  const latestMarketDate = comparison.latest_market_date ?? manifest?.latest_market_date;
  const safeBacktestEnd = comparison.safe_backtest_end ?? manifest?.safe_backtest_end;
  if (latestMarketDate && latestMarketDate > last.date) {
    captionParts.push(`SOXL K-line ${latestMarketDate}`);
    caption.dataset.tone = 'warning';
  }
  const baselineEnd =
    comparison.baseline_replay_end ?? manifest?.profile_summaries?.[comparison.baseline_profile]?.replay_end;
  const baselineWindow = manifest?.profile_summaries?.[comparison.baseline_profile]?.window;
  const actorEnd = maxDate(
    [
      comparison.actor_replay_end,
      ...profiles
        .filter((profile) => manifest?.profile_summaries?.[profile]?.status !== 'reference')
        .map((profile) => manifest?.profile_summaries?.[profile]?.replay_end),
    ],
  );
  captionParts.push(
    `Baseline ${profileLabel(comparison.baseline_profile, manifest)}${
      baselineWindow ? ` ${baselineWindow}` : ''
    }${baselineEnd ? ` latest ${baselineEnd}` : ''}`,
  );
  if (actorEnd) captionParts.push(`Qlib actors safe replay ${actorEnd}`);
  if (safeBacktestEnd) captionParts.push(`safe end ${safeBacktestEnd}`);
  if (baselineEnd && baselineEnd > last.date) {
    caption.dataset.tone = 'warning';
  }
  caption.textContent = captionParts.join(' | ');
  container.appendChild(caption);
  if (latestMarketDate && latestMarketDate > last.date && end > last.date) {
    const tail = element('div', 'latest-tail-banner');
    tail.dataset.tone = 'warning';
    tail.appendChild(element('strong', undefined, 'Latest tail'));
    tail.appendChild(element('span', undefined, `SOXL ${latestMarketDate}`));
    const baselineLatest = manifest?.baseline_latest;
    if (baselineLatest?.date) {
      const position =
        typeof baselineLatest.position === 'number' ? ` pos ${baselineLatest.position.toFixed(3)}` : '';
      const action = baselineLatest.action ? ` ${baselineLatest.action}` : '';
      const ret =
        typeof baselineLatest.return_pct === 'number' ? ` ret ${formatCompactPct(baselineLatest.return_pct)}` : '';
      tail.appendChild(
        element(
          'span',
          undefined,
          `baseline ${profileLabel(baselineLatest.profile, manifest)}${action}${position}${ret}`,
        ),
      );
    } else {
      tail.appendChild(
        element('span', undefined, `baseline ${profileLabel(comparison.baseline_profile, manifest)}`),
      );
    }
    if (actorEnd) tail.appendChild(element('span', undefined, `actors replay ${actorEnd}`));
    if (manifest?.latest_signal?.available_count) {
      tail.appendChild(
        element(
          'span',
          undefined,
          `actor signals ${manifest.latest_signal.available_count}@${manifest.latest_signal.date}`,
        ),
      );
    }
    container.appendChild(tail);
  }
  const decisionRows = latestDecisionRows(comparison, models, profiles, start, end, manifest);
  const hasLatestDecisionMatrix = decisionRows.length > 1;
  if (hasLatestDecisionMatrix) {
    const decision = element('div', 'latest-decision-matrix');
    const baselineRow = decisionRows.find((row) => row.isBaseline);
    const header = element('div', 'latest-decision-header');
    header.appendChild(element('strong', undefined, 'Latest model signals'));
    header.appendChild(element('span', undefined, `Latest ${decisionRows[0].targetDate}`));
    if (baselineRow) {
      header.appendChild(
        element(
          'span',
          undefined,
          `baseline ${profileLabel(baselineRow.profile, manifest)} ${baselineRow.action} ${baselineRow.position.toFixed(3)}`,
        ),
      );
    }
    decision.appendChild(header);
    const grid = element('div', 'latest-decision-grid');
    for (const row of decisionRows) {
      const item = element('article', 'latest-decision-card');
      item.dataset.tone = row.action === 'BUY' ? 'buy' : row.action === 'SELL' ? 'sell' : 'muted';
      item.dataset.diff = String(row.actionDiff || row.positionDiff);
      if (row.isBaseline) item.dataset.role = 'baseline';
      item.appendChild(element('strong', undefined, profileLabel(row.profile, manifest)));
      item.appendChild(element('span', 'latest-decision-action', `${row.action} ${row.position.toFixed(3)}`));
      if (!row.isBaseline) {
        item.appendChild(element('span', 'latest-decision-delta', `Δpos ${formatSigned(row.positionDelta, 3)}`));
      }
      item.appendChild(element('span', 'latest-decision-source', row.sourceLabel));
      grid.appendChild(item);
    }
    decision.appendChild(grid);
    container.appendChild(decision);
  }
  const latestSignalDiffs = visibleLatestSignalDifferences(comparison, profiles, start, end);
  const latestSignalDiffByProfile = new Map(latestSignalDiffs.map((row) => [row.profile, row]));
  if (latestSignalDiffs.length > 0 && !hasLatestDecisionMatrix) {
    const latestDiffStrip = element('div', 'latest-signal-difference-strip');
    const firstDiff = latestSignalDiffs[0];
    latestDiffStrip.appendChild(
      element(
        'span',
        'latest-signal-difference-title',
        `Latest signal delta ${firstDiff.date} vs ${profileLabel(firstDiff.baseline_profile, manifest)} ` +
          `${firstDiff.baseline_action} ${firstDiff.baseline_position.toFixed(3)}`,
      ),
    );
    for (const row of latestSignalDiffs) {
      const item = element('span', 'latest-signal-difference-pill');
      item.dataset.tone = row.model_action === 'BUY' ? 'buy' : row.model_action === 'SELL' ? 'sell' : 'muted';
      item.textContent =
        `${profileLabel(row.profile, manifest)} ${row.model_action} ${row.model_position.toFixed(3)} ` +
        `Δpos ${formatSigned(row.position_delta_vs_baseline, 3)} replay ${row.replay_asof ?? 'n/a'}`;
      latestDiffStrip.appendChild(item);
    }
    container.appendChild(latestDiffStrip);
  }
  const latestSignalRows = hasLatestDecisionMatrix ? [] : latestSignalDifferenceRows(comparison, models, start, end);
  if (latestSignalRows.length > 0) {
    const latestStrip = element('div', 'latest-signal-strip');
    const firstSignal = latestSignalRows[0];
    const comparableBaseline = latestSignalRows.every(
      (row) =>
        row.date === firstSignal.date &&
        row.baselineProfile === firstSignal.baselineProfile &&
        row.baselineAction === firstSignal.baselineAction &&
        row.baselinePosition === firstSignal.baselinePosition,
    );
    latestStrip.appendChild(
      element(
        'span',
        'latest-signal-title',
        comparableBaseline
          ? `Latest signal vs ${profileLabel(firstSignal.baselineProfile, manifest)} ` +
              `${firstSignal.baselineAction} ${firstSignal.baselinePosition.toFixed(3)} @${firstSignal.date}`
          : 'Latest signal differences',
      ),
    );
    for (const row of latestSignalRows) {
      const item = element('span', 'latest-signal-pill');
      item.dataset.tone = row.modelAction === 'BUY' ? 'buy' : row.modelAction === 'SELL' ? 'sell' : 'muted';
      item.textContent =
        `${profileLabel(row.profile, manifest)} ${row.modelAction} ${row.modelPosition.toFixed(3)} ` +
        `| Δpos ${formatSigned(row.positionDelta, 3)} | replay ${row.replayAsof}`;
      latestStrip.appendChild(item);
    }
    container.appendChild(latestStrip);
  }
  const available = profiles.filter((profile) => comparison.compared_profiles.includes(profile));
  const deltaProfiles = available.filter((profile) => profile !== comparison.baseline_profile);
  const selected = deltaProfiles.length ? deltaProfiles : available;
  const rankingByProfile = new Map((comparison.rankings ?? []).map((ranking) => [ranking.profile, ranking]));
  const differenceStats = selected
    .map((profile, index) => {
      const profileRows = visible.filter((row) => row.profiles[profile]);
      const actionMismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
      const positionDiffCount = profileRows.filter((row) =>
        isMaterialPositionDifference(row.profiles[profile].position_delta_vs_baseline),
      ).length;
      const values = last.profiles[profile];
      return { profile, index, actionMismatchCount, positionDiffCount, values };
    })
    .filter((item) => item.values)
    .sort((left, right) => {
      const leftLatestDelta = Math.abs(latestSignalDiffByProfile.get(left.profile)?.position_delta_vs_baseline ?? 0);
      const rightLatestDelta = Math.abs(latestSignalDiffByProfile.get(right.profile)?.position_delta_vs_baseline ?? 0);
      if (rightLatestDelta !== leftLatestDelta) return rightLatestDelta - leftLatestDelta;
      const leftScore = left.actionMismatchCount * 3 + left.positionDiffCount;
      const rightScore = right.actionMismatchCount * 3 + right.positionDiffCount;
      if (rightScore !== leftScore) return rightScore - leftScore;
      return profileLabel(left.profile, manifest).localeCompare(profileLabel(right.profile, manifest));
    });
  if (differenceStats.length > 0 && !hasLatestDecisionMatrix) {
    const quick = element('div', 'model-delta-quick');
    quick.textContent = differenceStats
      .map(({ profile, values }) => {
        if (!values) return '';
        const latestDiff = latestSignalDiffByProfile.get(profile);
        if (latestDiff) {
          return (
            `${profileLabel(profile, manifest)} signal ΔPos ${formatSigned(latestDiff.position_delta_vs_baseline, 3)} ` +
            `@${latestDiff.date.slice(5)}`
          );
        }
        return (
          `${profileLabel(profile, manifest)} ΔNAV ${formatSigned(values.nav_delta_vs_baseline, 2)} ` +
          `ΔPos ${formatSigned(values.position_delta_vs_baseline, 3)}`
        );
      })
      .filter(Boolean)
      .join(' | ');
    container.appendChild(quick);
  }
  const cards = element('div', 'model-difference-cards');
  for (const { profile, index, actionMismatchCount, positionDiffCount, values } of differenceStats) {
    if (!values) continue;
    const replayEnd = manifest?.profile_summaries?.[profile]?.replay_end;
    const replayWindow = manifest?.profile_summaries?.[profile]?.window;
    const ranking = rankingByProfile.get(profile);
    const freshness = freshnessLabel(replayEnd, last.date, manifest);
    const latestSignal = models[profile]?.latest_signal;
    const latestMismatch = latestMismatchForProfile(mismatchRows, profile);
    const latestDiff = latestSignalDiffByProfile.get(profile);
    const replayState =
      freshness.status === 'latest'
        ? 'latest'
        : freshness.status === 'stale'
          ? 'cut'
          : 'common';
    const card = element('article', 'model-difference-card');
    card.dataset.profile = profile;
    card.dataset.freshness = freshness.status;
    card.dataset.diff = String(actionMismatchCount > 0 || positionDiffCount > 0);
    card.style.borderLeftColor = MODEL_COLORS[index % MODEL_COLORS.length];
    const title = element('strong', 'model-difference-card-title', profileLabel(profile, manifest));
    card.appendChild(title);
    const sourceKind = profileSourceKind(profile, manifest);
    if (sourceKind) {
      card.appendChild(element('span', 'model-difference-card-kind', sourceKind.replaceAll('_', ' ')));
    }
    const replayLine = element(
      'span',
      'model-difference-card-replay',
      `${replayWindow ? `${replayWindow} ` : ''}replay ${replayEnd ?? 'n/a'} ${replayState}`,
    );
    if (latestSignal) {
      replayLine.appendChild(
        element(
          'span',
          'model-difference-card-signal',
          ` signal ${latestSignal.action} ${latestSignal.position.toFixed(3)} @${latestSignal.date.slice(5)}`,
        ),
      );
    } else if (freshness.gapLabel) {
      replayLine.appendChild(element('span', 'model-difference-card-gap', ` ${freshness.gapLabel}`));
    }
    card.appendChild(replayLine);
    const metrics = element('div', 'model-difference-card-metrics');
    for (const label of [
      latestDiff ? `Signal Δpos ${formatSigned(latestDiff.position_delta_vs_baseline, 3)} @${latestDiff.date.slice(5)}` : null,
      `ΔPos ${formatSigned(values.position_delta_vs_baseline, 3)}`,
      `ΔNAV ${formatSigned(values.nav_delta_vs_baseline, 2)}`,
      ranking
        ? `Ret ${formatPct(ranking.return * 100)}`
        : `Ret ${formatSigned((values.strategy_nav_rebased ?? values.strategy_nav) - 100, 2)}%`,
      ranking ? `MaxDD ${formatPct(ranking.max_drawdown * 100)}` : 'MaxDD n/a',
      ranking ? `AvgPos ${ranking.avg_position.toFixed(3)}` : 'AvgPos n/a',
      `Diff ${actionMismatchCount}/${positionDiffCount}`,
    ].filter((label): label is string => Boolean(label))) {
      metrics.appendChild(element('span', undefined, label));
    }
    card.appendChild(metrics);
    const feature = element('span', 'model-difference-card-feature');
    if (latestMismatch) {
      feature.title = mismatchFeatureLabel(latestMismatch);
      feature.textContent = mismatchFeatureLabel(latestMismatch);
      card.title = feature.title;
    } else {
      feature.textContent = '-';
    }
    card.appendChild(feature);
    cards.appendChild(card);
  }
  if (cards.childElementCount > 0) {
    container.appendChild(cards);
  }
  const tableWrap = element('div', 'model-diff-table-wrap');
  const table = element('table', 'model-diff-table');
  const thead = element('thead');
  const header = element('tr');
  for (const label of ['Model', 'Replay', 'Latest signal', 'Ret', 'MaxDD', 'AvgPos', 'ΔNAV', 'ΔPos', 'ActionDiff', 'PosDiff', 'Action', 'Last feature diff']) {
    header.appendChild(element('th', undefined, label));
  }
  thead.appendChild(header);
  table.appendChild(thead);
  const tbody = element('tbody');
  for (const [index, profile] of selected.entries()) {
    const values = last.profiles[profile];
    if (!values) continue;
    const profileRows = visible.filter((row) => row.profiles[profile]);
    const actionMismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
    const positionDiffCount = profileRows.filter((row) =>
      isMaterialPositionDifference(row.profiles[profile].position_delta_vs_baseline),
    ).length;
    const rebasedNav = values.strategy_nav_rebased ?? values.strategy_nav;
    const replayEnd = manifest?.profile_summaries?.[profile]?.replay_end;
    const ranking = rankingByProfile.get(profile);
    const freshness = freshnessLabel(replayEnd, last.date, manifest);
    const latestSignal = models[profile]?.latest_signal;
    const row = element('tr', 'model-difference-row');
    row.dataset.freshness = freshness.status;
    row.dataset.profileStatus = manifest?.profile_summaries?.[profile]?.status ?? '';
    row.title = profile;
    row.style.borderLeftColor = MODEL_COLORS[index % MODEL_COLORS.length];
    row.style.borderLeftWidth = '4px';
    const name = element('td');
    name.appendChild(element('strong', 'model-difference-name', profileLabel(profile, manifest)));
    row.appendChild(name);
    const replayState =
      freshness.status === 'latest'
        ? 'latest'
        : freshness.status === 'stale'
          ? 'cut'
          : 'common';
    const replay = element('td', 'model-difference-date', `${replayEnd ?? 'n/a'} ${replayState}`);
    row.appendChild(replay);
    const signal = element('td', latestSignal ? 'model-difference-signal' : 'model-difference-gap');
    if (latestSignal) {
      signal.textContent = `Signal ${latestSignal.action} ${latestSignal.position.toFixed(3)} @${latestSignal.date.slice(5)}`;
    } else {
      signal.textContent = freshness.gapLabel ?? values.action;
    }
    row.appendChild(signal);
    row.appendChild(
      element('td', undefined, ranking ? `Ret ${formatPct(ranking.return * 100)}` : `Ret ${formatSigned(rebasedNav - 100, 2)}%`),
    );
    row.appendChild(element('td', undefined, ranking ? formatPct(ranking.max_drawdown * 100) : 'n/a'));
    row.appendChild(element('td', undefined, ranking ? ranking.avg_position.toFixed(3) : 'n/a'));
    row.appendChild(element('td', undefined, formatSigned(values.nav_delta_vs_baseline, 2)));
    row.appendChild(element('td', undefined, formatSigned(values.position_delta_vs_baseline, 3)));
    row.appendChild(element('td', undefined, String(actionMismatchCount)));
    row.appendChild(element('td', undefined, String(positionDiffCount)));
    row.appendChild(element('td', undefined, values.action));
    const latestMismatch = latestMismatchForProfile(mismatchRows, profile);
    const feature = element('td', 'model-difference-feature');
    if (latestMismatch) {
      feature.title = mismatchFeatureLabel(latestMismatch);
      feature.textContent = mismatchFeatureLabel(latestMismatch);
    } else {
      feature.textContent = '-';
    }
    row.appendChild(feature);
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  tableWrap.appendChild(table);
  const history = element('details', 'model-diff-history') as HTMLDetailsElement;
  history.appendChild(element('summary', undefined, `Historical differences (${selected.length})`));
  history.appendChild(tableWrap);
  container.appendChild(history);
}

export function renderActionLegend(container: HTMLElement): void {
  container.replaceChildren();
  const buy = element('span', 'legend-item legend-buy', 'BUY');
  const sell = element('span', 'legend-item legend-sell', 'SELL');
  const diff = element('span', 'legend-item legend-diff', 'DIFF');
  const position = element('span', 'legend-item legend-position', 'POS');
  container.append(buy, sell, diff, position);
}

export function renderInspectorPanel(container: HTMLElement, rows: InspectorRow[]): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Inspector'));
  const list = element('div', 'inspector-list');
  for (const row of rows) {
    const item = element('div', 'inspector-row');
    if (row.tone) item.dataset.tone = row.tone;
    item.appendChild(element('span', 'inspector-label', row.label));
    item.appendChild(element('span', 'inspector-value', row.value));
    list.appendChild(item);
  }
  container.appendChild(list);
}

export function renderComparisonPanel(
  container: HTMLElement,
  comparison: ComparisonPayload,
  start: string,
  end: string,
  mismatchRows: readonly ActionMismatchRow[] = [],
  onDate?: (date: string) => void,
): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Model Snapshot'));
  const visible = computeVisibleComparison(comparison, start, end);
  const last = visible[visible.length - 1];
  const header = element(
    'div',
    'comparison-summary',
    `${visible.length} comparison rows | baseline ${comparison.baseline_profile} | selected ${start} to ${end}`,
  );
  container.appendChild(header);

  const rankings = comparison.rankings?.filter((item) => comparison.compared_profiles.includes(item.profile)) ?? [];
  if (rankings.length > 0) {
    const rankingStrip = element('div', 'ranking-strip');
    for (const [index, ranking] of rankings.entries()) {
      const item = element('article', 'ranking-item');
      item.style.borderLeftColor = MODEL_COLORS[index % MODEL_COLORS.length];
      item.appendChild(element('strong', undefined, formatProfileLabel(ranking.profile)));
      item.appendChild(element('span', undefined, `Ret ${formatPct(ranking.return * 100)}`));
      item.appendChild(element('span', undefined, `MaxDD ${formatPct(ranking.max_drawdown * 100)}`));
      item.appendChild(element('span', undefined, `AvgPos ${ranking.avg_position.toFixed(3)}`));
      rankingStrip.appendChild(item);
    }
    container.appendChild(rankingStrip);
  }

  const cards = element('div', 'comparison-cards');
  for (const profile of comparison.compared_profiles) {
    const profileRows = visible.filter((row) => row.profiles[profile]);
    const actionMismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
    const positionDiffCount = profileRows.filter((row) =>
      isMaterialPositionDifference(row.profiles[profile].position_delta_vs_baseline),
    ).length;
    const values = last?.profiles[profile];
    const card = element('article', 'comparison-card');
    card.appendChild(element('div', 'comparison-card-title', formatProfileLabel(profile)));
    card.appendChild(element('span', 'metric-label', 'Rebased NAV delta'));
    card.appendChild(element('strong', undefined, values ? values.nav_delta_vs_baseline.toFixed(2) : 'n/a'));
    card.appendChild(element('span', 'metric-label', 'Position delta'));
    card.appendChild(element('strong', undefined, values ? values.position_delta_vs_baseline.toFixed(3) : 'n/a'));
    card.appendChild(element('span', 'metric-label', 'ActionDiff Days'));
    card.appendChild(element('strong', undefined, String(actionMismatchCount)));
    card.appendChild(element('span', 'metric-label', 'PosDiff Days'));
    card.appendChild(element('strong', undefined, String(positionDiffCount)));
    cards.appendChild(card);
  }
  container.appendChild(cards);

  const mismatchSection = element('details', 'mismatch-section comparison-diagnostics') as HTMLDetailsElement;
  mismatchSection.appendChild(element('summary', undefined, `Action/position differences (${mismatchRows.length})`));
  const visibleMismatchRows = mismatchRows.slice(-12);
  if (visibleMismatchRows.length === 0) {
    mismatchSection.appendChild(element('p', 'empty-note', 'No visible buy/sell or material position differences.'));
  } else {
    if (mismatchRows.length > visibleMismatchRows.length) {
      mismatchSection.appendChild(
        element('p', 'empty-note', `Showing recent ${visibleMismatchRows.length} of ${mismatchRows.length}.`),
      );
    }
    const list = element('div', 'mismatch-list');
    for (const row of visibleMismatchRows) {
      const item = element('article', 'mismatch-row');
      const dateButton = element('button', 'mismatch-date-button', row.date);
      dateButton.type = 'button';
      dateButton.addEventListener('click', () => onDate?.(row.date));
      const header = element('div', 'mismatch-row-header');
      header.append(
        dateButton,
        element('span', 'mismatch-profile', formatProfileLabel(row.profile)),
        element('span', 'mismatch-actions', `${row.model_action} vs ${row.baseline_action}`),
      );
      const metrics = element(
        'div',
        'mismatch-metrics',
        [
          `pos ${row.model_position.toFixed(3)} vs ${row.baseline_position.toFixed(3)}`,
          `delta ${row.position_delta_vs_baseline.toFixed(3)}`,
          `nav ${row.nav_delta_vs_baseline.toFixed(2)}`,
          row.close === null ? 'close n/a' : `close ${row.close.toFixed(2)}`,
        ].join(' | '),
      );
      const featureLine = element(
        'div',
        'mismatch-features',
        Object.entries(row.features)
          .slice(0, 4)
          .map(([key, value]) => `${key} ${value}`)
          .join(' | '),
      );
      item.append(header, metrics);
      if (featureLine.textContent) item.appendChild(featureLine);
      list.appendChild(item);
    }
    mismatchSection.appendChild(list);
  }
  container.appendChild(mismatchSection);
}
