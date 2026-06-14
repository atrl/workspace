import {
  buildActionFeatureRows,
  buildBuySellFeatureBaselineContrastRows,
  buildFeatureBaselineSchemaRows,
  buildBaselineFeatureContrastRows,
  buildFeatureSchemaOverlapRows,
  buildSelectedDateDecisionFeatureCards,
  computeVisibleComparison,
} from './domain';
import type { ChartRangeCommand } from './chart';
import type { ChartClickMode } from './viewState';
import type {
  ComparisonPayload,
  AutoUpdateStatus,
  AutoUpdateTask,
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
      ['replay candidate note', auto.model_replay_freshness?.note || 'n/a'],
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
  renderModelRuntimeStatusTable(container, manifest);
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

function renderModelRuntimeStatusTable(container: HTMLElement, manifest: Manifest): void {
  const rows = manifest.model_runtime_status ?? [];
  if (!rows.length) return;
  const section = element('section', 'auto-update-block model-runtime-status');
  section.appendChild(element('h3', undefined, 'Model Runtime Status'));
  const table = element('table', 'comparison-table compact-table runtime-status-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Role', 'Run', 'Reason', 'Schema', 'Window', 'Source']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);
  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.dataset.role = row.role;
    const schemaText = `${row.feature_schema_signature ?? 'n/a'} · ${row.feature_count} features · ${row.snapshot_count} snapshots`;
    const windowText =
      row.replay_start && row.replay_end
        ? `${row.replay_start} to ${row.replay_end} · ${row.row_count} rows`
        : `${row.row_count} rows`;
    tr.appendChild(element('td', undefined, row.display_name || row.profile));
    tr.appendChild(element('td', undefined, row.role));
    tr.appendChild(element('td', undefined, row.runnable ? 'yes' : 'no'));
    tr.appendChild(element('td', undefined, row.runnable_reason));
    tr.appendChild(element('td', 'feature-snapshot-cell', schemaText));
    tr.appendChild(element('td', 'feature-snapshot-cell', windowText));
    tr.appendChild(element('td', 'feature-snapshot-cell', row.source_path || row.output || 'n/a'));
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

  container.appendChild(renderAutoUpdateTaskList(auto));

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
    ['replay candidate note', auto.model_replay_freshness?.note || 'n/a'],
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
  renderModelRuntimeStatusTable(container, manifest);
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

function profileCountLabel(count: number, total: number, noun: string): string {
  return `${count} / ${total} ${noun}`;
}

function taskCount(
  tasks: readonly { enabled?: boolean; loaded?: boolean | null; installed?: boolean }[],
  field: 'enabled' | 'loaded' | 'installed',
): number {
  return tasks.filter((task) => task[field] === true).length;
}

function taskStateBits(task: AutoUpdateTask): string[] {
  return [
    task.enabled ? 'enabled' : 'disabled',
    task.configured === undefined ? 'configured unknown' : task.configured ? 'configured' : 'not configured',
    task.installed === undefined ? 'install unknown' : task.installed ? 'installed' : 'not installed',
    task.loaded === undefined || task.loaded === null ? 'loaded unknown' : task.loaded ? 'loaded' : 'not loaded',
  ].filter((value): value is string => Boolean(value));
}

function renderAutoUpdateTaskCard(task: AutoUpdateTask): HTMLElement {
  const item = element('div', 'task-card');
  item.dataset.enabled = task.enabled ? 'true' : 'false';
  const detailBits = [
    task.label,
    ...taskStateBits(task),
  ].filter((value): value is string => Boolean(value));
  item.appendChild(element('strong', undefined, task.name));
  item.appendChild(element('span', undefined, detailBits.join(' · ')));
  item.appendChild(element('span', undefined, task.schedule ?? 'n/a'));
  item.appendChild(element('code', undefined, task.command_path ?? 'n/a'));
  item.appendChild(element('code', undefined, task.log_path ?? 'n/a'));
  return item;
}

function renderAutoUpdateTaskList(auto: AutoUpdateStatus): HTMLElement {
  const section = element('section', 'auto-update-block auto-update-task-section');
  section.appendChild(element('h3', undefined, 'Auto Update Tasks'));
  section.appendChild(
    element(
      'div',
      'comparison-summary',
      `${auto.tasks.length} local tasks | refresh ${auto.run_now_command ?? auto.manual_refresh_command ?? 'n/a'}`,
    ),
  );
  const taskList = element('div', 'task-list');
  for (const task of auto.tasks) {
    taskList.appendChild(renderAutoUpdateTaskCard(task));
  }
  section.appendChild(taskList);
  return section;
}

function statusPill(title: string, value: string, detail: string): HTMLElement {
  const pill = element('div', 'status-pill');
  pill.appendChild(element('strong', undefined, title));
  pill.appendChild(element('span', undefined, value));
  pill.appendChild(element('small', undefined, detail));
  return pill;
}

export function renderDashboardStatusStrip(container: HTMLElement, manifest: Manifest): void {
  container.replaceChildren();
  container.className = 'status-strip';
  const auto = manifest.auto_update;
  const tasks = auto?.tasks ?? [];
  const taskTotal = tasks.length;
  const enabled = taskCount(tasks, 'enabled');
  const loaded = taskCount(tasks, 'loaded');
  const installed = taskCount(tasks, 'installed');
  const taskNames = tasks.map((task) => task.name).join(', ');
  const runnableCount = manifest.profile_groups?.runnable?.length ?? 0;
  const referenceProfiles = manifest.profile_groups?.reference ?? [];
  const evidenceOnlyCount = manifest.profile_groups?.evidence_only?.length ?? 0;
  const profileTotal = manifest.available_profiles.length;
  const baselineLabel = referenceProfiles.includes('soxl_backtest_policy_reimplementation_v1')
    ? 'SOXLBacktest baseline'
    : referenceProfiles.join(', ') || 'no reference baseline';
  const safeEnd = auto?.model_replay_freshness?.safe_backtest_end ?? manifest.safe_backtest_end;
  const latest = auto?.model_replay_freshness?.latest_data_date ?? manifest.latest_market_date;
  const replayReadyCount = auto?.model_replay_freshness?.ready_count ?? 0;
  const replayProfileCount = auto?.model_replay_freshness?.profile_count ?? 0;
  const frontier = manifest.model_retention_frontier;

  container.appendChild(
    statusPill(
      'Auto Update',
      taskTotal ? taskNames : 'not configured',
      taskTotal
        ? `${profileCountLabel(enabled, taskTotal, 'enabled')} · ${profileCountLabel(loaded, taskTotal, 'loaded')} · ${profileCountLabel(installed, taskTotal, 'installed')} · ${profileCountLabel(replayReadyCount, replayProfileCount, 'replay candidates')}`
        : 'no local refresh task in manifest',
    ),
  );
  container.appendChild(
    statusPill(
      'RL Runtime',
      profileCountLabel(runnableCount, profileTotal, 'runnable'),
      `${baselineLabel} · ${evidenceOnlyCount} evidence-only`,
    ),
  );
  container.appendChild(
    statusPill(
      'Replay Window',
      `safe ${safeEnd ?? 'n/a'}`,
      `market latest ${latest ?? 'n/a'}`,
    ),
  );
  container.appendChild(
    statusPill(
      'Training Frontier',
      frontier?.recommended_next_source ?? 'n/a',
      frontier?.recommended_primary_blocker ?? frontier?.recommended_next_action ?? 'n/a',
    ),
  );
  if (tasks.length) {
    const taskRail = element('div', 'status-task-rail');
    for (const task of tasks) {
      const chip = element('span', 'status-task-chip');
      chip.dataset.enabled = task.enabled ? 'true' : 'false';
      chip.textContent = `${task.name} · ${task.schedule ?? 'n/a'} · ${task.loaded ? 'loaded' : 'not loaded'}`;
      taskRail.appendChild(chip);
    }
    container.appendChild(taskRail);
  }
}

export function renderActionLegend(container: HTMLElement): void {
  container.replaceChildren();
  const buy = element('span', 'legend-item legend-buy', 'BUY');
  const sell = element('span', 'legend-item legend-sell', 'SELL');
  const price = element('span', 'legend-item legend-price', 'SOXL Price');
  const position = element('span', 'legend-item legend-position', 'Position');
  container.append(buy, sell, price, position);
}

const CHART_COMMANDS: readonly { command: ChartRangeCommand; label: string; icon: string; shortcut: string }[] = [
  { command: 'zoom_in', label: 'Zoom in', icon: '+', shortcut: '+/=' },
  { command: 'zoom_out', label: 'Zoom out', icon: '-', shortcut: '-/_' },
  { command: 'pan_left', label: 'Pan left', icon: '<', shortcut: 'Left/H' },
  { command: 'pan_right', label: 'Pan right', icon: '>', shortcut: 'Right/L' },
  { command: 'reset', label: 'Reset range', icon: 'R', shortcut: 'R/Home' },
  { command: 'latest', label: 'Follow latest', icon: '>|', shortcut: 'End' },
];

const CHART_BAR_WINDOWS: readonly { command: ChartRangeCommand; label: string; icon: string; shortcut: string }[] = [
  { command: 'last_60', label: 'Show last 60 bars', icon: '60', shortcut: '60' },
  { command: 'last_120', label: 'Show last 120 bars', icon: '120', shortcut: '120' },
  { command: 'last_250', label: 'Show last 250 bars', icon: '250', shortcut: '250' },
];

const CHART_CLICK_MODES: readonly { mode: ChartClickMode; label: string; icon: string; shortcut: string }[] = [
  { mode: 'inspect', label: 'Inspect candles', icon: 'I', shortcut: 'Esc' },
  { mode: 'set_start', label: 'Click chart to set Start', icon: '[', shortcut: 'S' },
  { mode: 'set_end', label: 'Click chart to set End', icon: ']', shortcut: 'E' },
];

export function renderChartCommandBar(
  container: HTMLElement,
  onCommand: (command: ChartRangeCommand) => void,
  activeMode: ChartClickMode = 'inspect',
  onModeChange?: (mode: ChartClickMode) => void,
): void {
  container.replaceChildren();
  const modeGroup = element('div', 'chart-mode-group');
  for (const item of CHART_CLICK_MODES) {
    const button = element('button', 'chart-command-button chart-mode-button', item.icon);
    button.type = 'button';
    button.title = `${item.label} (${item.shortcut})`;
    button.setAttribute('aria-label', item.label);
    button.dataset.active = item.mode === activeMode ? 'true' : 'false';
    button.addEventListener('click', () => onModeChange?.(item.mode));
    modeGroup.appendChild(button);
  }
  container.appendChild(modeGroup);
  const barWindowGroup = element('div', 'chart-range-command-group');
  for (const item of CHART_BAR_WINDOWS) {
    const button = element('button', 'chart-command-button chart-bars-button', item.icon);
    button.type = 'button';
    button.title = `${item.label} (${item.shortcut})`;
    button.setAttribute('aria-label', item.label);
    button.addEventListener('click', () => onCommand(item.command));
    barWindowGroup.appendChild(button);
  }
  container.appendChild(barWindowGroup);
  for (const item of CHART_COMMANDS) {
    const button = element('button', 'chart-command-button', item.icon);
    button.type = 'button';
    button.title = `${item.label} (${item.shortcut})`;
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

function renderFeatureDriverMatrix(container: HTMLElement, models: Record<string, ModelPayload>): void {
  container.appendChild(element('h3', undefined, 'Feature Driver Matrix'));
  const rows = buildFeatureSchemaOverlapRows(models, { featureNameLimit: 5, topBuySellLimit: 5 });
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No selected model feature drivers.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table driver-matrix-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Runtime', 'Schema', 'Top BUY/SELL Drivers', 'Model-only']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.appendChild(element('td', undefined, row.display_name));
    tr.appendChild(element('td', undefined, `${row.source_kind} · ${row.runtime}`));
    tr.appendChild(element('td', undefined, `${row.feature_schema_signature} · ${row.feature_count} cols`));
    tr.appendChild(element('td', 'feature-snapshot-cell', row.top_buy_sell_features.join(', ') || 'n/a'));
    tr.appendChild(element('td', 'feature-snapshot-cell', row.model_only_features.join(', ') || 'n/a'));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

function renderFeatureBaselineSchemaTable(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
  baselineProfile: string,
): void {
  container.appendChild(element('h3', undefined, 'Schema vs Baseline'));
  container.appendChild(element('div', 'comparison-summary', `baseline ${baselineProfile}`));
  const rows = buildFeatureBaselineSchemaRows(models, baselineProfile, {
    featureNameLimit: 6,
    topBuySellLimit: 5,
  });
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No baseline-relative feature schemas.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table schema-baseline-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of [
    'Model',
    'Runtime',
    'Schema',
    'Shared',
    'model-only vs baseline',
    'baseline-only',
    'Top BUY/SELL',
  ]) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.appendChild(element('td', undefined, row.display_name));
    tr.appendChild(element('td', undefined, row.runtime));
    tr.appendChild(
      element(
        'td',
        'feature-snapshot-cell',
        `${row.feature_schema_signature} · ${row.feature_count} schema / ${row.snapshot_count} snapshot · ${row.feature_schema_source}`,
      ),
    );
    tr.appendChild(element('td', 'feature-snapshot-cell', `${row.shared_count}: ${row.shared_features.join(', ') || 'n/a'}`));
    tr.appendChild(
      element('td', 'feature-snapshot-cell', `${row.model_only_count}: ${row.model_only_features.join(', ') || 'n/a'}`),
    );
    tr.appendChild(
      element('td', 'feature-snapshot-cell', `${row.baseline_only_count}: ${row.baseline_only_features.join(', ') || 'n/a'}`),
    );
    tr.appendChild(element('td', 'feature-snapshot-cell', row.top_buy_sell_features.join(', ') || 'n/a'));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

function renderBuySellFeatureBaselineContrastTable(
  container: HTMLElement,
  models: Record<string, ModelPayload>,
  baselineProfile: string,
): void {
  container.appendChild(element('h3', undefined, 'Buy/Sell Drivers vs Baseline'));
  const rows = buildBuySellFeatureBaselineContrastRows(models, baselineProfile, {
    limitPerModel: 6,
    baselineFeatureLimit: 4,
  });
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No baseline-relative buy/sell driver rows.'));
    return;
  }

  const table = element('table', 'comparison-table feature-table buy-sell-baseline-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of [
    'Model',
    'Feature',
    'Presence',
    'model BUY mean',
    'model SELL mean',
    'model BUY-SELL',
    'baseline BUY mean',
    'baseline SELL mean',
    'baseline BUY-SELL',
    'Driver delta',
  ]) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  for (const row of rows) {
    const tr = element('tr');
    tr.dataset.presence = row.presence;
    tr.appendChild(element('td', undefined, row.display_name));
    tr.appendChild(element('td', undefined, row.feature));
    tr.appendChild(element('td', undefined, row.presence));
    tr.appendChild(
      element('td', undefined, `${formatFeatureValue(row.model_buy_mean)} (${row.model_buy_count})`),
    );
    tr.appendChild(
      element('td', undefined, `${formatFeatureValue(row.model_sell_mean)} (${row.model_sell_count})`),
    );
    tr.appendChild(element('td', undefined, formatSignedFeatureDelta(row.model_buy_minus_sell)));
    tr.appendChild(
      element('td', undefined, `${formatFeatureValue(row.baseline_buy_mean)} (${row.baseline_buy_count})`),
    );
    tr.appendChild(
      element('td', undefined, `${formatFeatureValue(row.baseline_sell_mean)} (${row.baseline_sell_count})`),
    );
    tr.appendChild(element('td', undefined, formatSignedFeatureDelta(row.baseline_buy_minus_sell)));
    tr.appendChild(element('td', undefined, formatSignedFeatureDelta(row.driver_delta_vs_baseline)));
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
  const baselineProfile = input.baselineProfile ?? Object.keys(input.models)[0] ?? '';
  container.appendChild(
    element(
      'div',
      'comparison-summary',
      `${Object.keys(input.models).length} selected models | feature schemas and BUY/SELL event snapshots`,
    ),
  );
  renderFeatureBaselineSchemaTable(container, input.models, baselineProfile);
  renderBuySellFeatureBaselineContrastTable(container, input.models, baselineProfile);
  renderFeatureDriverMatrix(container, input.models);
  renderFeatureActionSummaryTable(container, input.models);
  renderBaselineFeatureContrastTable(container, input.models, baselineProfile);
  renderDecisionFeatureCards(container, input);
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
}
