import {
  buildActionFeatureRows,
  buildBaselineFeatureContrastRows,
  buildFeatureSchemaOverlapRows,
  buildSelectedDateDecisionFeatureCards,
  computeVisibleComparison,
} from './domain';
import type { ChartRangeCommand } from './chart';
import type {
  ComparisonPayload,
  DecisionFeatureCard,
  FeatureActionSummaryRow,
  FeatureDeltaValue,
  FeatureNameValue,
  FeatureValue,
  InspectorRow,
  Manifest,
  MarketRow,
  ModelPayload,
  ModelRow,
} from './types';

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

export function renderDataSourcePanel(container: HTMLElement, manifest: Manifest, warnings: string[] = []): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Data Sources'));
  const grid = element('div', 'source-grid');
  const groups = manifest.profile_groups;
  const rows: [string, string][] = [
    ['safe_backtest_end', manifest.safe_backtest_end],
    ['latest_market_date', manifest.latest_market_date],
    ['generated_at', manifest.generated_at],
    ['baseline', manifest.default_baseline_profile],
    ['models', manifest.default_comparison_profiles.join(', ')],
    ['runnable profiles', groups?.runnable?.join(', ') || 'none'],
    ['reference profiles', groups?.reference?.join(', ') || 'none'],
    ['evidence-only profiles', groups?.evidence_only?.join(', ') || 'none'],
    ['unavailable profiles', String(manifest.unavailable_profiles?.length ?? 0)],
    ...Object.entries(manifest.source_paths),
  ];
  for (const [label, value] of rows) {
    const item = element('div', 'source-row');
    item.appendChild(element('span', 'source-label', label));
    item.appendChild(element('span', 'source-value', value));
    grid.appendChild(item);
  }
  container.appendChild(grid);
  const auto = manifest.auto_update;
  if (auto) {
    const autoSection = element('section', 'auto-update-block');
    autoSection.appendChild(element('h3', undefined, 'Auto Update'));
    const taskList = element('div', 'source-grid');
    const autoRows: [string, string][] = [
      ['configured', auto.configured ? 'yes' : 'no'],
      ['timezone', auto.timezone],
      ['freshness safe end', auto.model_replay_freshness?.safe_backtest_end ?? 'n/a'],
      [
        'runnable replay profiles',
        `${auto.model_replay_freshness?.ready_count ?? 0} / ${auto.model_replay_freshness?.profile_count ?? 0}`,
      ],
      [
        'replay refresh counts',
        Object.entries(auto.model_replay_freshness?.counts ?? {})
          .map(([key, value]) => `${key}=${value}`)
          .join(' ') || 'n/a',
      ],
      ['freshness source', auto.model_replay_freshness?.source_path ?? 'n/a'],
      ['install auto update', auto.install_command ?? 'n/a'],
      ['run refresh now', auto.run_now_command ?? 'n/a'],
      ['manual refresh', auto.manual_refresh_command ?? 'n/a'],
      ['refresh pipeline', auto.refresh_pipeline?.join(' -> ') ?? 'n/a'],
      ['remote deploy required', auto.remote_deploy_required ? 'yes' : 'no'],
    ];
    for (const [label, value] of autoRows) {
      const item = element('div', 'source-row');
      item.appendChild(element('span', 'source-label', label));
      item.appendChild(element('span', 'source-value', value));
      taskList.appendChild(item);
    }
    for (const task of auto.tasks) {
      const stateBits = [
        task.enabled ? 'enabled' : 'disabled',
        task.configured === undefined ? null : task.configured ? 'configured' : 'not configured',
        task.installed === undefined ? null : task.installed ? 'installed' : 'not installed',
        task.loaded === undefined || task.loaded === null ? null : task.loaded ? 'loaded' : 'not loaded',
      ].filter((value): value is string => Boolean(value));
      const detailBits = [
        task.schedule,
        task.command_path ? `cmd ${task.command_path}` : null,
        task.log_path ? `log ${task.log_path}` : null,
      ].filter((value): value is string => Boolean(value));
      const item = element('div', 'source-row');
      item.appendChild(element('span', 'source-label', task.name));
      item.appendChild(element('span', 'source-value', [...stateBits, ...detailBits].join(' · ')));
      taskList.appendChild(item);
    }
    autoSection.appendChild(taskList);
    container.appendChild(autoSection);
  }
  renderUnavailableProfileTable(container, manifest);
  const retention = manifest.model_retention_gate;
  if (retention) {
    const retentionSection = element('section', 'auto-update-block');
    retentionSection.appendChild(element('h3', undefined, 'Retention Gate'));
    const gateRows: [string, string][] = [
      [
        'promotable',
        `${retention.passing_profile_count ?? 0} / ${retention.profiles_scanned ?? 0} promotable`,
      ],
      [
        'walk-forward MaxDD',
        retention.max_walkforward_drawdown_pct !== undefined
          ? `MaxDD < ${retention.max_walkforward_drawdown_pct.toFixed(2)}%`
          : 'n/a',
      ],
      ['return floor', retention.required_best_return_pct !== undefined ? `${retention.required_best_return_pct.toFixed(2)}%` : 'n/a'],
      ['windows', retention.required_wf_windows?.join(', ') || 'n/a'],
      [
        'keep_actor strict violations',
        retention.registry_keep_actor_count !== undefined
          ? `${retention.registry_keep_actor_strict_violation_count ?? 0} / ${retention.registry_keep_actor_count} blocked`
          : 'n/a',
      ],
      ['source', retention.source_path ?? 'n/a'],
    ];
    const gateGrid = element('div', 'source-grid');
    for (const [label, value] of gateRows) {
      const item = element('div', 'source-row');
      item.appendChild(element('span', 'source-label', label));
      item.appendChild(element('span', 'source-value', value));
      gateGrid.appendChild(item);
    }
    retentionSection.appendChild(gateGrid);
    container.appendChild(retentionSection);
  }
  const frontier = manifest.model_retention_frontier;
  if (frontier) {
    const frontierSection = element('section', 'auto-update-block');
    frontierSection.appendChild(element('h3', undefined, 'Training Frontier'));
    const frontierRows: [string, string][] = [
      ['recommended source', frontier.recommended_next_source ?? 'n/a'],
      ['recommended action', frontier.recommended_next_action ?? 'n/a'],
      ['primary blocker', frontier.recommended_primary_blocker ?? 'n/a'],
      ['rows scanned', String(frontier.rows_scanned ?? 0)],
      ['completed handoffs', String(frontier.completed_handoff_count ?? 0)],
      ['source', frontier.source_path ?? 'n/a'],
    ];
    const frontierGrid = element('div', 'source-grid');
    for (const [label, value] of frontierRows) {
      const item = element('div', 'source-row');
      item.appendChild(element('span', 'source-label', label));
      item.appendChild(element('span', 'source-value', value));
      frontierGrid.appendChild(item);
    }
    frontierSection.appendChild(frontierGrid);
    container.appendChild(frontierSection);
  }
  if (warnings.length) {
    const warningList = element('ul', 'warning-list');
    for (const warning of warnings) {
      warningList.appendChild(element('li', undefined, warning));
    }
    container.appendChild(warningList);
  }
}

function renderUnavailableProfileTable(container: HTMLElement, manifest: Manifest): void {
  const unavailable = manifest.unavailable_profiles ?? [];
  if (!unavailable.length) return;
  const section = element('section', 'auto-update-block unavailable-models');
  section.appendChild(element('h3', undefined, 'Unavailable Model Replays'));
  const table = element('table', 'comparison-table compact-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Result', 'Reason', 'Actor/Profile', 'Output']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);
  const tbody = element('tbody');
  for (const item of unavailable) {
    const tr = element('tr');
    tr.dataset.tone = 'warning';
    tr.appendChild(element('td', undefined, item.profile));
    tr.appendChild(element('td', undefined, item.result));
    tr.appendChild(element('td', undefined, item.reason));
    tr.appendChild(element('td', 'feature-snapshot-cell', [item.actor, item.profile_yaml].filter(Boolean).join(' · ') || 'n/a'));
    tr.appendChild(element('td', 'feature-snapshot-cell', item.output ?? 'n/a'));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  section.appendChild(tableScroll);
  container.appendChild(section);
}

export function renderOperationsPanel(container: HTMLElement, manifest: Manifest, warnings: string[] = []): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Operations'));
  const auto = manifest.auto_update;
  if (!auto) {
    container.appendChild(element('p', 'empty-state', 'No auto update status in manifest.'));
    return;
  }

  const counts = auto.model_replay_freshness?.counts ?? {};
  const readyCount = auto.model_replay_freshness?.ready_count ?? 0;
  const profileCount = auto.model_replay_freshness?.profile_count ?? 0;
  const runnableCount = manifest.profile_groups?.runnable?.length ?? 0;
  const availableProfileCount = manifest.available_profiles?.length ?? 0;
  const grid = element('div', 'source-grid operations-grid');
  const rows: [string, string][] = [
    ['auto update', auto.configured ? 'configured' : 'not configured'],
    ['replay ready', `${readyCount} / ${profileCount}`],
    ['runnable inference', `${runnableCount} / ${availableProfileCount}`],
    [
      'refresh counts',
      Object.entries(counts)
        .map(([key, value]) => `${key}=${value}`)
        .join(' ') || 'n/a',
    ],
    ['safe replay end', auto.model_replay_freshness?.safe_backtest_end ?? 'n/a'],
    ['install auto update', auto.install_command ?? 'n/a'],
    ['run refresh now', auto.run_now_command ?? 'n/a'],
    ['manual refresh', auto.manual_refresh_command ?? 'n/a'],
    ['pipeline', auto.refresh_pipeline?.join(' -> ') ?? 'n/a'],
    ['next training', manifest.model_retention_frontier?.recommended_next_source ?? 'n/a'],
    ['next action', manifest.model_retention_frontier?.recommended_next_action ?? 'n/a'],
    ['training blocker', manifest.model_retention_frontier?.recommended_primary_blocker ?? 'n/a'],
  ];
  for (const [label, value] of rows) {
    const item = element('div', 'source-row');
    item.appendChild(element('span', 'source-label', label));
    item.appendChild(element('span', 'source-value', value));
    grid.appendChild(item);
  }
  container.appendChild(grid);

  const taskList = element('div', 'task-list');
  for (const task of auto.tasks) {
    const item = element('div', 'task-card');
    item.dataset.enabled = task.enabled ? 'true' : 'false';
    const configured = task.configured === undefined ? 'configured unknown' : task.configured ? 'configured' : 'not configured';
    const installed = task.installed === undefined ? 'install unknown' : task.installed ? 'installed' : 'not installed';
    const loaded = task.loaded === undefined || task.loaded === null ? 'loaded unknown' : task.loaded ? 'loaded' : 'not loaded';
    item.appendChild(element('strong', undefined, task.name));
    item.appendChild(element('span', undefined, [task.label, configured, installed, loaded].filter(Boolean).join(' · ')));
    item.appendChild(element('span', undefined, task.schedule ?? 'n/a'));
    item.appendChild(element('code', undefined, task.command_path ?? 'n/a'));
    item.appendChild(element('code', undefined, task.log_path ?? 'n/a'));
    taskList.appendChild(item);
  }
  container.appendChild(taskList);
  renderUnavailableProfileTable(container, manifest);

  const critical = warnings.filter((warning) => warning.includes('Replay') || warning.includes('runnable') || warning.includes('not runnable'));
  if (critical.length) {
    const warningList = element('ul', 'warning-list');
    for (const warning of critical.slice(0, 4)) {
      warningList.appendChild(element('li', undefined, warning));
    }
    container.appendChild(warningList);
  }
}

export function renderActionLegend(container: HTMLElement): void {
  container.replaceChildren();
  const buy = element('span', 'legend-item legend-buy', 'BUY');
  const sell = element('span', 'legend-item legend-sell', 'SELL');
  const position = element('span', 'legend-item legend-position', 'Position');
  container.append(buy, sell, position);
}

const CHART_COMMANDS: readonly { command: ChartRangeCommand; label: string; icon: string }[] = [
  { command: 'zoom_in', label: 'Zoom in', icon: '+' },
  { command: 'zoom_out', label: 'Zoom out', icon: '-' },
  { command: 'pan_left', label: 'Pan left', icon: '<' },
  { command: 'pan_right', label: 'Pan right', icon: '>' },
  { command: 'reset', label: 'Reset range', icon: 'R' },
  { command: 'latest', label: 'Follow latest', icon: '>|' },
];

export function renderChartCommandBar(
  container: HTMLElement,
  onCommand: (command: ChartRangeCommand) => void,
): void {
  container.replaceChildren();
  for (const item of CHART_COMMANDS) {
    const button = element('button', 'chart-command-button', item.icon);
    button.type = 'button';
    button.title = item.label;
    button.setAttribute('aria-label', item.label);
    button.addEventListener('click', () => onCommand(item.command));
    container.appendChild(button);
  }
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
  selectedProfiles: readonly string[] = comparison.compared_profiles,
  models: Record<string, ModelPayload> = {},
  baselineProfile = comparison.baseline_profile,
): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Model Difference'));
  const visible = computeVisibleComparison(comparison, start, end);
  const header = element(
    'div',
    'comparison-summary',
    `${visible.length} rows | baseline ${baselineProfile} | ${start} to ${end}`,
  );
  container.appendChild(header);

  const table = element('table', 'comparison-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Return', 'BuyHold', 'Vs BuyHold', 'Last Pos Delta', 'Action mismatches', 'Retention']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  const last = visible[visible.length - 1];
  for (const profile of selectedProfiles) {
    const profileRows = visible.filter((row) => row.profiles[profile]);
    const fallback = fallbackComparisonFromModelRows(profile, models, baselineProfile, start, end);
    const useStaticComparison = baselineProfile === comparison.baseline_profile && profileRows.length > 0;
    const mismatchCount = useStaticComparison
      ? profileRows.filter((row) => row.profiles[profile].action_mismatch).length
      : (fallback?.actionMismatchCount ?? 0);
    const values = (useStaticComparison ? last?.profiles[profile] : undefined) ?? fallback?.values;
    const tr = element('tr');
    tr.appendChild(element('td', undefined, models[profile]?.display_name || profile));
    tr.appendChild(element('td', undefined, values ? `${(values.strategy_return * 100).toFixed(2)}%` : 'n/a'));
    tr.appendChild(element('td', undefined, values ? `${(values.buyhold_return * 100).toFixed(2)}%` : 'n/a'));
    tr.appendChild(element('td', undefined, values ? `${(values.return_delta_vs_buyhold * 100).toFixed(2)}pp` : 'n/a'));
    tr.appendChild(element('td', undefined, values ? values.position_delta_vs_baseline.toFixed(3) : 'n/a'));
    tr.appendChild(element('td', undefined, String(mismatchCount)));
    const model = models[profile];
    const hasRetention = model?.retention_gate_pass !== undefined || model?.retention_decision !== undefined;
    const retentionLabel = model?.retention_decision === 'keep_reference_baseline'
      ? 'baseline'
      : model?.retention_gate_pass
        ? 'promotable'
        : 'evidence-only';
    const retentionText = model && hasRetention
      ? [retentionLabel, ...(model.retention_failures ?? []).slice(0, 2)].join(' · ')
      : 'n/a';
    tr.appendChild(element('td', undefined, retentionText));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

function rowsInRange(rows: readonly ModelRow[], start: string, end: string): ModelRow[] {
  return rows.filter((row) => row.date >= start && row.date <= end);
}

function fallbackComparisonFromModelRows(
  profile: string,
  models: Record<string, ModelPayload>,
  baselineProfile: string,
  start: string,
  end: string,
):
  | {
      values: {
        strategy_return: number;
        buyhold_return: number;
        return_delta_vs_buyhold: number;
        position_delta_vs_baseline: number;
      };
      actionMismatchCount: number;
    }
  | null {
  const modelRows = rowsInRange(models[profile]?.rows ?? [], start, end);
  if (!modelRows.length) return null;
  const first = modelRows[0];
  const last = modelRows[modelRows.length - 1];
  const strategyReturn = first.strategy_nav ? last.strategy_nav / first.strategy_nav - 1 : 0;
  const buyholdReturn = first.buyhold_nav ? last.buyhold_nav / first.buyhold_nav - 1 : 0;
  const baselineRowsByDate = new Map(rowsInRange(models[baselineProfile]?.rows ?? [], start, end).map((row) => [row.date, row]));
  const baselineLast = baselineRowsByDate.get(last.date);
  const actionMismatchCount =
    profile === baselineProfile
      ? 0
      : modelRows.filter((row) => {
          const baselineRow = baselineRowsByDate.get(row.date);
          return baselineRow !== undefined && row.action !== baselineRow.action;
        }).length;
  return {
    values: {
      strategy_return: strategyReturn,
      buyhold_return: buyholdReturn,
      return_delta_vs_buyhold: strategyReturn - buyholdReturn,
      position_delta_vs_baseline: baselineLast ? last.position - baselineLast.position : 0,
    },
    actionMismatchCount,
  };
}

function formatFeatureValue(value: FeatureValue): string {
  if (typeof value === 'number' && Number.isFinite(value)) return value.toFixed(2);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return value === null || value === undefined ? 'n/a' : String(value);
}

const FEATURE_LIST_LIMIT = 12;

function renderFeatureList(features: Record<string, FeatureValue>, limit = FEATURE_LIST_LIMIT): string {
  const entries = Object.entries(features);
  if (!entries.length) return 'n/a';
  const visibleEntries = entries.slice(0, Math.max(1, limit));
  const suffix = entries.length > visibleEntries.length ? ` · +${entries.length - visibleEntries.length} more` : '';
  return `${visibleEntries.map(([key, value]) => `${key} ${formatFeatureValue(value)}`).join(' · ')}${suffix}`;
}

function formatSignedFeatureDelta(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
  return value > 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function formatFeatureChip(row: FeatureNameValue): string {
  return `${row.feature} ${formatFeatureValue(row.value)}`;
}

function formatFeatureDeltaChip(row: FeatureDeltaValue): string {
  const delta = formatSignedFeatureDelta(row.delta);
  return `${row.feature} ${formatFeatureValue(row.value)} vs ${formatFeatureValue(row.baseline_value)} (${delta})`;
}

function renderChipList(
  container: HTMLElement,
  rows: readonly string[],
  emptyText: string,
): void {
  const list = element('div', 'feature-chip-list');
  if (rows.length) {
    for (const text of rows) {
      list.appendChild(element('span', 'feature-chip', text));
    }
  } else {
    list.appendChild(element('span', 'feature-chip feature-chip-empty', emptyText));
  }
  container.appendChild(list);
}

function renderDecisionFeatureCards(
  container: HTMLElement,
  input: { date: string; marketRows: readonly MarketRow[]; models: Record<string, ModelPayload>; baselineProfile?: string },
): void {
  container.appendChild(element('h3', undefined, 'Selected Decision Features'));
  const baselineProfile = input.baselineProfile ?? Object.keys(input.models)[0] ?? '';
  const cards = buildSelectedDateDecisionFeatureCards({
    date: input.date,
    marketRows: input.marketRows,
    models: input.models,
    baselineProfile,
  });
  if (!cards.length) {
    container.appendChild(element('p', 'empty-state', 'No selected-date decision rows.'));
    return;
  }
  const grid = element('div', 'decision-card-grid');
  for (const card of cards) {
    grid.appendChild(renderDecisionFeatureCard(card));
  }
  container.appendChild(grid);
}

function renderDecisionFeatureCard(card: DecisionFeatureCard): HTMLElement {
  const node = element('article', 'decision-card');
  node.dataset.action = card.action;
  node.dataset.trade = card.is_trade ? 'true' : 'false';
  const header = element('div', 'decision-card-header');
  header.appendChild(element('strong', undefined, card.display_name));
  header.appendChild(element('span', 'decision-action', `${card.action} · pos ${card.position.toFixed(3)}`));
  node.appendChild(header);

  const baselineText =
    card.position_delta_vs_baseline === null
      ? `baseline ${card.baseline_action}`
      : `baseline ${card.baseline_action} · pos delta ${formatSignedFeatureDelta(card.position_delta_vs_baseline)}`;
  const dateText = card.is_exact_date ? `as of ${card.date}` : `latest row ${card.date}`;
  node.appendChild(element('div', 'decision-card-meta', `${dateText} · ${card.source_kind} · ${baselineText}`));

  node.appendChild(element('div', 'decision-card-label', 'Model-only'));
  renderChipList(node, card.model_only_features.map(formatFeatureChip), 'none on selected date');
  node.appendChild(element('div', 'decision-card-label', 'Shared vs baseline'));
  renderChipList(node, card.shared_feature_deltas.map(formatFeatureDeltaChip), 'baseline row or shared deltas unavailable');
  node.appendChild(element('div', 'decision-card-label', 'BUY/SELL drivers'));
  renderChipList(node, card.buy_sell_features, 'no trade feature summary');
  return node;
}

function topFeatureSummaryRows(model: ModelPayload, limit = 4): FeatureActionSummaryRow[] {
  return [...(model.feature_action_summary ?? [])]
    .sort((left, right) => {
      const leftScore = Math.abs(left.buy_minus_sell ?? 0);
      const rightScore = Math.abs(right.buy_minus_sell ?? 0);
      return rightScore - leftScore || left.feature.localeCompare(right.feature);
    })
    .slice(0, limit);
}

function renderFeatureActionSummaryTable(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
): void {
  container.appendChild(element('h3', undefined, 'Buy/Sell Feature Summary'));
  const rows = Object.entries(models).flatMap(([profile, model]) =>
    topFeatureSummaryRows(model).map((summary) => ({
      profile: model.display_name || profile,
      feature: summary.feature,
      buyCount: summary.buy_count,
      sellCount: summary.sell_count,
      buyMean: summary.buy_mean,
      sellMean: summary.sell_mean,
      delta: summary.buy_minus_sell,
    })),
  );
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No buy/sell feature summary for selected models.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Feature', 'BUY n', 'BUY mean', 'SELL n', 'SELL mean', 'BUY-SELL']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.appendChild(element('td', undefined, row.profile));
    tr.appendChild(element('td', undefined, row.feature));
    tr.appendChild(element('td', undefined, String(row.buyCount)));
    tr.appendChild(element('td', undefined, formatFeatureValue(row.buyMean)));
    tr.appendChild(element('td', undefined, String(row.sellCount)));
    tr.appendChild(element('td', undefined, formatFeatureValue(row.sellMean)));
    tr.appendChild(element('td', undefined, formatSignedFeatureDelta(row.delta)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

function renderBaselineFeatureContrastTable(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
  baselineProfile: string,
): void {
  container.appendChild(element('h3', undefined, 'Trade Feature vs Baseline'));
  const rows = buildBaselineFeatureContrastRows({
    models,
    baselineProfile,
    limitPerEvent: 5,
    maxRows: 40,
  });
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No buy/sell feature contrast against the selected baseline.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table baseline-feature-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Date', 'Model', 'Action', 'Baseline', 'Pos Delta', 'Feature', 'Value', 'Baseline Value', 'Delta']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.dataset.trade = 'true';
    tr.appendChild(element('td', undefined, row.date));
    tr.appendChild(element('td', undefined, row.display_name));
    tr.appendChild(element('td', undefined, row.action));
    tr.appendChild(element('td', undefined, `baseline ${row.baseline_action}`));
    tr.appendChild(
      element(
        'td',
        undefined,
        row.position_delta_vs_baseline === null ? 'n/a' : formatSignedFeatureDelta(row.position_delta_vs_baseline),
      ),
    );
    tr.appendChild(element('td', undefined, row.feature));
    tr.appendChild(element('td', undefined, formatFeatureValue(row.value)));
    tr.appendChild(element('td', undefined, formatFeatureValue(row.baseline_value)));
    tr.appendChild(element('td', undefined, formatSignedFeatureDelta(row.delta)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

function renderFeatureSchemaTable(container: HTMLElement, models: Record<string, ModelPayload>): void {
  container.appendChild(element('h3', undefined, 'Model Feature Schemas'));
  const rows = buildFeatureSchemaOverlapRows(models);
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No selected model schemas.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table schema-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Runtime', 'Schema', 'Cols', 'Shared', 'Model-only', 'Top BUY/SELL']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.appendChild(element('td', undefined, row.display_name));
    tr.appendChild(element('td', undefined, `${row.source_kind} · ${row.runtime}`));
    tr.appendChild(element('td', undefined, `${row.feature_schema_signature} · ${row.feature_schema_source}`));
    tr.appendChild(element('td', undefined, `${row.feature_count} schema / ${row.snapshot_count} snapshot`));
    tr.appendChild(element('td', 'feature-snapshot-cell', `${row.shared_count}: ${row.shared_features.join(', ') || 'n/a'}`));
    tr.appendChild(
      element('td', 'feature-snapshot-cell', `${row.model_only_count}: ${row.model_only_features.join(', ') || 'n/a'}`),
    );
    tr.appendChild(element('td', 'feature-snapshot-cell', row.top_buy_sell_features.join(', ') || 'n/a'));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

function renderRecentEventFeatureTable(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
): void {
  const rows = Object.entries(models)
    .flatMap(([profile, model]) =>
      model.events
        .filter((event) => event.kind === 'BUY' || event.kind === 'SELL')
        .map((event) => ({
          profile: model.display_name || profile,
          date: event.date,
          action: event.kind,
          position: event.position,
          features: event.feature_snapshot ?? {},
          featureCount: model.feature_columns.length,
        })),
    )
    .sort((left, right) => right.date.localeCompare(left.date))
    .slice(0, 24);

  container.appendChild(element('h3', undefined, 'Recent Buy/Sell Events'));
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No buy/sell events in selected models.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Date', 'Model', 'Action', 'Position', 'Snapshot', 'Schema cols']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.dataset.trade = 'true';
    tr.appendChild(element('td', undefined, row.date));
    tr.appendChild(element('td', undefined, row.profile));
    tr.appendChild(element('td', undefined, row.action));
    tr.appendChild(element('td', undefined, row.position.toFixed(3)));
    tr.appendChild(element('td', 'feature-snapshot-cell', renderFeatureList(row.features)));
    tr.appendChild(element('td', undefined, String(row.featureCount)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

export function renderFeatureSnapshotPanel(
  container: HTMLElement,
  input: { date: string; marketRows: readonly MarketRow[]; models: Record<string, ModelPayload> },
): void {
  container.replaceChildren();
  const rows = buildActionFeatureRows(input);
  container.appendChild(element('div', 'comparison-summary', `${input.date} | ${rows.length} selected model rows`));
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No model rows for selected date.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Action', 'Position', 'Feature packs', 'Snapshot', 'Schema cols']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.dataset.trade = row.is_trade ? 'true' : 'false';
    tr.appendChild(element('td', undefined, row.display_name));
    tr.appendChild(element('td', undefined, row.action));
    tr.appendChild(element('td', undefined, row.position.toFixed(3)));
    tr.appendChild(element('td', undefined, row.feature_packs.length ? row.feature_packs.join(', ') : 'core'));
    tr.appendChild(element('td', 'feature-snapshot-cell', renderFeatureList(row.features)));
    tr.appendChild(element('td', undefined, String(row.feature_columns.length)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

export function renderFeatureDateSnapshotPanel(
  container: HTMLElement,
  input: { date: string; marketRows: readonly MarketRow[]; models: Record<string, ModelPayload> },
): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Selected Date Features'));
  const snapshotPanel = element('section', 'feature-date-snapshot');
  renderFeatureSnapshotPanel(snapshotPanel, input);
  container.appendChild(snapshotPanel);
}

export function renderFeatureComparisonPanel(
  container: HTMLElement,
  input: { date: string; marketRows: readonly MarketRow[]; models: Record<string, ModelPayload>; baselineProfile?: string },
): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Buy/Sell Feature Comparison'));
  container.appendChild(
    element(
      'div',
      'comparison-summary',
      `${Object.keys(input.models).length} selected models | feature schemas and BUY/SELL event snapshots`,
    ),
  );
  renderDecisionFeatureCards(container, input);
  renderFeatureActionSummaryTable(container, input.models);
  renderBaselineFeatureContrastTable(container, input.models, input.baselineProfile ?? Object.keys(input.models)[0] ?? '');
  renderRecentEventFeatureTable(container, input.models);
  renderFeatureSchemaTable(container, input.models);
}

export function renderFeatureSpotlightPanel(
  container: HTMLElement,
  input: { date: string; marketRows: readonly MarketRow[]; models: Record<string, ModelPayload>; baselineProfile?: string },
): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Buy/Sell Features'));
  container.appendChild(
    element(
      'div',
      'comparison-summary',
      `${input.date} | selected model snapshots and buy/sell contrasts`,
    ),
  );
  renderDecisionFeatureCards(container, input);
  const snapshotPanel = element('section', 'feature-date-snapshot');
  snapshotPanel.id = 'feature-date-snapshot';
  renderFeatureSnapshotPanel(snapshotPanel, input);
  container.appendChild(snapshotPanel);
  renderFeatureActionSummaryTable(container, input.models);
  renderBaselineFeatureContrastTable(container, input.models, input.baselineProfile ?? Object.keys(input.models)[0] ?? '');
  renderRecentEventFeatureTable(container, input.models);
  renderFeatureSchemaTable(container, input.models);
}
