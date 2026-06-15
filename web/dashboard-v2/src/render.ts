import { computeVisibleComparison, formatProfileLabel } from './domain';
import type { ActionMismatchRow, ComparisonPayload, InspectorRow, Manifest } from './types';

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

function readableProfile(profile: string): string {
  return formatProfileLabel(profile).replaceAll('_', ' ');
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
    ['Market', manifest.latest_market_date],
    ['Replay', manifest.safe_backtest_end],
    ['Baseline', formatProfileLabel(manifest.default_baseline_profile)],
    ['Blocked', String(blockedCount)],
  ];
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

export function renderActionLegend(container: HTMLElement): void {
  container.replaceChildren();
  const buy = element('span', 'legend-item legend-buy', 'BUY');
  const sell = element('span', 'legend-item legend-sell', 'SELL');
  const position = element('span', 'legend-item legend-position', 'Position');
  container.append(buy, sell, position);
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

  const cards = element('div', 'comparison-cards');
  for (const profile of comparison.compared_profiles) {
    const profileRows = visible.filter((row) => row.profiles[profile]);
    const mismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
    const values = last?.profiles[profile];
    const card = element('article', 'comparison-card');
    card.appendChild(element('div', 'comparison-card-title', formatProfileLabel(profile)));
    card.appendChild(element('span', 'metric-label', 'NAV delta'));
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
