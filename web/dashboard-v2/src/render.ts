import { computeVisibleComparison } from './domain';
import type { ComparisonPayload, InspectorRow, Manifest } from './types';

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
  for (const title of ['Model', 'Last NAV Delta', 'Last Position Delta', 'Action mismatches']) {
    headerRow.appendChild(element('th', undefined, title));
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = element('tbody');
  const last = visible[visible.length - 1];
  for (const profile of comparison.compared_profiles) {
    const profileRows = visible.filter((row) => row.profiles[profile]);
    const mismatchCount = profileRows.filter((row) => row.profiles[profile].action_mismatch).length;
    const values = last?.profiles[profile];
    const tr = element('tr');
    tr.appendChild(element('td', undefined, profile));
    tr.appendChild(element('td', undefined, values ? values.nav_delta_vs_baseline.toFixed(2) : 'n/a'));
    tr.appendChild(element('td', undefined, values ? values.position_delta_vs_baseline.toFixed(3) : 'n/a'));
    tr.appendChild(element('td', undefined, String(mismatchCount)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  const tableScroll = element('div', 'table-scroll');
  tableScroll.appendChild(table);
  container.appendChild(tableScroll);
}
