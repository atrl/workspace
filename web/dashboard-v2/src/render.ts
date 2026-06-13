import { buildActionFeatureRows, computeVisibleComparison } from './domain';
import type { ChartRangeCommand } from './chart';
import type { ComparisonPayload, FeatureValue, InspectorRow, Manifest, MarketRow, ModelPayload } from './types';

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
  const rows: [string, string][] = [
    ['safe_backtest_end', manifest.safe_backtest_end],
    ['latest_market_date', manifest.latest_market_date],
    ['generated_at', manifest.generated_at],
    ['baseline', manifest.default_baseline_profile],
    ['models', manifest.default_comparison_profiles.join(', ')],
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
      ['freshness source', auto.model_replay_freshness?.source_path ?? 'n/a'],
      ['remote deploy required', auto.remote_deploy_required ? 'yes' : 'no'],
      ...auto.tasks.map((task) => [
        task.name,
        `${task.enabled ? 'enabled' : 'disabled'}${task.schedule ? ` · ${task.schedule}` : ''}`,
      ] as [string, string]),
    ];
    for (const [label, value] of autoRows) {
      const item = element('div', 'source-row');
      item.appendChild(element('span', 'source-label', label));
      item.appendChild(element('span', 'source-value', value));
      taskList.appendChild(item);
    }
    autoSection.appendChild(taskList);
    container.appendChild(autoSection);
  }
  if (warnings.length) {
    const warningList = element('ul', 'warning-list');
    for (const warning of warnings) {
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
): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Model Difference'));
  const visible = computeVisibleComparison(comparison, start, end);
  const header = element(
    'div',
    'comparison-summary',
    `${visible.length} rows | baseline ${comparison.baseline_profile} | ${start} to ${end}`,
  );
  container.appendChild(header);

  const table = element('table', 'comparison-table');
  const thead = element('thead');
  const headerRow = element('tr');
  for (const title of ['Model', 'Return', 'BuyHold', 'Vs BuyHold', 'Last Pos Delta', 'Action mismatches']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  const last = visible[visible.length - 1];
  for (const profile of selectedProfiles) {
    const profileRows = visible.filter((row) => row.profiles[profile]);
    const mismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
    const values = last?.profiles[profile];
    const tr = element('tr');
    tr.appendChild(element('td', undefined, profile));
    tr.appendChild(element('td', undefined, values ? `${(values.strategy_return * 100).toFixed(2)}%` : 'n/a'));
    tr.appendChild(element('td', undefined, values ? `${(values.buyhold_return * 100).toFixed(2)}%` : 'n/a'));
    tr.appendChild(element('td', undefined, values ? `${(values.return_delta_vs_buyhold * 100).toFixed(2)}pp` : 'n/a'));
    tr.appendChild(element('td', undefined, values ? values.position_delta_vs_baseline.toFixed(3) : 'n/a'));
    tr.appendChild(element('td', undefined, String(mismatchCount)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}

function formatFeatureValue(value: FeatureValue): string {
  if (typeof value === 'number' && Number.isFinite(value)) return value.toFixed(2);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return value === null || value === undefined ? 'n/a' : String(value);
}

function renderFeatureList(features: Record<string, FeatureValue>): string {
  const entries = Object.entries(features);
  if (!entries.length) return 'n/a';
  return entries.map(([key, value]) => `${key} ${formatFeatureValue(value)}`).join(' · ');
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

export function renderFeatureComparisonPanel(
  container: HTMLElement,
  input: { date: string; marketRows: readonly MarketRow[]; models: Record<string, ModelPayload> },
): void {
  container.replaceChildren();
  container.appendChild(element('h2', undefined, 'Buy/Sell Feature Comparison'));
  const rows = buildActionFeatureRows(input);
  container.appendChild(element('div', 'comparison-summary', `${input.date} | ${rows.length} selected model rows`));
  if (!rows.length) {
    container.appendChild(element('p', 'empty-state', 'No model rows for selected date.'));
    renderRecentEventFeatureTable(container, input.models);
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
  renderRecentEventFeatureTable(container, input.models);
}
