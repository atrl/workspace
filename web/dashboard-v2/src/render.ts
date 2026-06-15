import { computeVisibleComparison, formatProfileLabel } from './domain';
import type { ActionMismatchRow, ComparisonPayload, InspectorRow, Manifest } from './types';

const MODEL_COLORS = ['#39b7c5', '#a597ff', '#e1aa3e', '#1fb981', '#f472b6', '#f97316'];

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

function formatSigned(value: number, digits: number): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(digits)}`;
}

function readableProfile(profile: string): string {
  return formatProfileLabel(profile).replaceAll('_', ' ');
}

function maxDate(values: readonly (string | undefined | null)[]): string | null {
  const dates = values.filter((value): value is string => Boolean(value)).sort();
  return dates.at(-1) ?? null;
}

function freshnessLabel(replayEnd: string | undefined, commonEnd: string, manifest?: Manifest): {
  label: string;
  status: 'latest' | 'safe';
} {
  if (replayEnd && replayEnd > commonEnd) {
    return { label: `Latest ${replayEnd}`, status: 'latest' };
  }
  if (replayEnd && manifest?.latest_market_date && replayEnd >= manifest.latest_market_date) {
    return { label: `Latest ${replayEnd}`, status: 'latest' };
  }
  return { label: `Common ${commonEnd}`, status: 'safe' };
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
  const strip = element('div', 'status-strip');
  const rows: [string, string][] = [
    ['Price', manifest.latest_market_date],
    ['Replay', manifest.safe_backtest_end],
    ['Baseline', formatProfileLabel(manifest.default_baseline_profile)],
    ['Blocked', String(blockedCount)],
  ];
  if (manifest.canonical_data_date && manifest.canonical_data_date !== manifest.latest_market_date) {
    rows.splice(1, 0, ['Canonical', manifest.canonical_data_date]);
  }
  if (manifest.strict_retention_gate) {
    rows.push([
      'Strict Gate',
      `${manifest.strict_retention_gate.passing_profile_count}/${manifest.strict_retention_gate.profiles_scanned}`,
    ]);
  }
  for (const [label, value] of rows) {
    const item = element('div', 'status-pill');
    if (label === 'Blocked' && blockedCount > 0) item.dataset.tone = 'warning';
    if (label === 'Strict Gate' && manifest.strict_retention_gate?.status === 'blocked') item.dataset.tone = 'warning';
    item.appendChild(element('span', 'source-label', label));
    item.appendChild(element('span', 'source-value', value));
    strip.appendChild(item);
  }
  container.appendChild(strip);
  const profileSummaries = manifest.profile_summaries ?? {};
  const freshnessRows = manifest.default_comparison_profiles
    .map((profile) => [profile, profileSummaries[profile]?.replay_end] as const)
    .filter(([, replayEnd]) => replayEnd);
  if (freshnessRows.length > 0) {
    const freshness = element('div', 'freshness-strip');
    freshness.appendChild(element('span', 'source-label', 'Model dates'));
    for (const [profile, replayEnd] of freshnessRows) {
      const item = element('span', 'freshness-pill');
      item.appendChild(element('span', 'freshness-name', formatProfileLabel(profile)));
      item.appendChild(element('span', 'freshness-date', String(replayEnd)));
      freshness.appendChild(item);
    }
    container.appendChild(freshness);
  }
  const baselineEnd = profileSummaries[manifest.default_baseline_profile]?.replay_end ?? null;
  const actorEnd = maxDate(
    manifest.default_comparison_profiles
      .filter((profile) => profileSummaries[profile]?.status !== 'reference')
      .map((profile) => profileSummaries[profile]?.replay_end),
  );
  if (
    (actorEnd && actorEnd < manifest.latest_market_date) ||
    (baselineEnd && baselineEnd < manifest.latest_market_date) ||
    manifest.safe_backtest_end < manifest.latest_market_date
  ) {
    const alert = element('div', 'freshness-alert');
    alert.appendChild(element('strong', undefined, 'Latest gap'));
    alert.appendChild(
      element(
        'span',
        undefined,
        [
          `Market ${manifest.latest_market_date}`,
          actorEnd ? `Actors ${actorEnd}` : null,
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
  const blockedWarnings = Object.entries(manifest.blocked_profiles ?? {}).map(
    ([profile, status]) => `${profile}: ${status.reason}`,
  );
  const allWarnings = [...warnings, ...gateWarnings, ...blockedWarnings];
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
): void {
  container.replaceChildren();
  const visible = computeVisibleComparison(comparison, start, end);
  const last = visible.at(-1);
  if (!last) {
    container.appendChild(element('span', 'empty-note', 'No comparison rows'));
    return;
  }
  const caption = element('div', 'model-delta-caption');
  const captionParts = [`Compare through ${last.date}`];
  if (manifest?.latest_market_date && manifest.latest_market_date > last.date) {
    captionParts.push(`Price ${manifest.latest_market_date}`);
    caption.dataset.tone = 'warning';
  }
  const baselineEnd = manifest?.profile_summaries?.[comparison.baseline_profile]?.replay_end;
  if (baselineEnd && baselineEnd > last.date) {
    captionParts.push(`Baseline ${baselineEnd}`);
    caption.dataset.tone = 'warning';
  }
  caption.textContent = captionParts.join(' | ');
  container.appendChild(caption);
  const selected = profiles.filter((profile) => comparison.compared_profiles.includes(profile));
  const rankingByProfile = new Map((comparison.rankings ?? []).map((ranking) => [ranking.profile, ranking]));
  const grid = element('div', 'model-diff-grid');
  for (const [index, profile] of selected.entries()) {
    const values = last.profiles[profile];
    if (!values) continue;
    const profileRows = visible.filter((row) => row.profiles[profile]);
    const mismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
    const rebasedNav = values.strategy_nav_rebased ?? values.strategy_nav;
    const replayEnd = manifest?.profile_summaries?.[profile]?.replay_end;
    const ranking = rankingByProfile.get(profile);
    const freshness = freshnessLabel(replayEnd, last.date, manifest);
    const pill = element('article', 'model-delta-pill model-difference-row');
    pill.dataset.freshness = freshness.status;
    pill.dataset.profileStatus = manifest?.profile_summaries?.[profile]?.status ?? '';
    pill.title = profile;
    pill.style.borderLeftColor = MODEL_COLORS[index % MODEL_COLORS.length];
    pill.style.borderLeftWidth = '4px';
    pill.appendChild(element('strong', 'model-difference-name', formatProfileLabel(profile)));
    if (replayEnd) pill.appendChild(element('span', undefined, `End ${replayEnd}`));
    pill.appendChild(element('span', 'model-difference-date', freshness.label));
    pill.appendChild(
      element(
        'span',
        undefined,
        ranking ? `Ret ${formatPct(ranking.return * 100)}` : `Ret ${formatSigned(rebasedNav - 100, 2)}%`,
      ),
    );
    if (ranking) {
      pill.appendChild(element('span', undefined, `MaxDD ${formatPct(ranking.max_drawdown * 100)}`));
      pill.appendChild(element('span', undefined, `AvgPos ${ranking.avg_position.toFixed(3)}`));
    }
    pill.appendChild(element('span', undefined, `ΔNAV ${formatSigned(values.nav_delta_vs_baseline, 2)}`));
    pill.appendChild(element('span', undefined, `ΔPos ${formatSigned(values.position_delta_vs_baseline, 3)}`));
    pill.appendChild(element('span', undefined, `Mismatch ${mismatchCount}`));
    pill.appendChild(element('span', undefined, `Action ${values.action}`));
    grid.appendChild(pill);
  }
  container.appendChild(grid);
}

export function renderActionLegend(container: HTMLElement): void {
  container.replaceChildren();
  const buy = element('span', 'legend-item legend-buy', 'BUY');
  const sell = element('span', 'legend-item legend-sell', 'SELL');
  container.append(buy, sell);
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
    const mismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
    const values = last?.profiles[profile];
    const card = element('article', 'comparison-card');
    card.appendChild(element('div', 'comparison-card-title', formatProfileLabel(profile)));
    card.appendChild(element('span', 'metric-label', 'Rebased NAV delta'));
    card.appendChild(element('strong', undefined, values ? values.nav_delta_vs_baseline.toFixed(2) : 'n/a'));
    card.appendChild(element('span', 'metric-label', 'Position delta'));
    card.appendChild(element('strong', undefined, values ? values.position_delta_vs_baseline.toFixed(3) : 'n/a'));
    card.appendChild(element('span', 'metric-label', 'Mismatch Days'));
    card.appendChild(element('strong', undefined, String(mismatchCount)));
    cards.appendChild(card);
  }
  container.appendChild(cards);

  const mismatchSection = element('details', 'mismatch-section comparison-diagnostics') as HTMLDetailsElement;
  mismatchSection.appendChild(element('summary', undefined, `Action mismatches (${mismatchRows.length})`));
  const visibleMismatchRows = mismatchRows.slice(-12);
  if (visibleMismatchRows.length === 0) {
    mismatchSection.appendChild(element('p', 'empty-note', 'No visible buy/sell action differences.'));
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
