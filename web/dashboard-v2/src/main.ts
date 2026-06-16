import './styles.css';
import {
  renderDeltaChart,
  renderMarketChart,
  renderNavChart,
  renderPositionChart,
  syncCrosshairPositions,
  syncVisibleTimeRanges,
} from './chart';
import {
  clampComparisonProfiles,
  clipRowsByDate,
  computeActionMismatchRows,
  computeCommonWindow,
  computeDateRangePreset,
  computeDateRangeFromLogicalRange,
  computeDefaultDateRange,
  computeFocusedDateRange,
  computeInspectorRows,
  computeKeyboardDateRange,
  formatProfileLabel,
  selectDefaultChartProfile,
  type DateRangePreset,
  type KeyboardRangeAction,
} from './domain';
import { loadDashboardData, type DashboardData } from './data';
import {
  renderActionLegend,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderInspectorPanel,
  renderModelDeltaStrip,
} from './render';
import type { ModelPayload } from './types';
import { viewUpdateRequiresChartRender } from './viewState';

interface ViewState {
  selectedProfiles: string[];
  baselineProfile: string;
  start: string;
  end: string;
  hoveredDate: string;
}

interface UpdateOptions {
  skipChartRender?: boolean;
}

type UpdateState = (next: Partial<ViewState>, options?: UpdateOptions) => void;

function requireElement(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element: ${id}`);
  return node;
}

function createShell(app: HTMLElement): void {
  app.innerHTML = `
    <header class="topbar">
      <section id="source-panel" class="source-panel compact-panel" aria-live="polite"></section>
      <nav id="toolbar" class="toolbar" aria-label="Dashboard controls"></nav>
    </header>
    <main id="workspace" class="workspace">
      <section class="chart-column">
        <div class="chart-title-row">
          <h1>SOXL</h1>
          <span>K-line with buy/sell differences; model position and NAV panes below</span>
          <div id="action-legend" class="action-legend"></div>
        </div>
        <div id="model-delta-strip" class="model-delta-strip"></div>
        <div id="market-chart" class="chart chart-main"></div>
        <div class="lower-charts">
          <section>
            <h2>Model Position</h2>
            <div id="position-chart" class="chart chart-small"></div>
          </section>
          <section>
            <h2>Delta NAV vs Baseline</h2>
            <div id="delta-nav-chart" class="chart chart-small"></div>
          </section>
          <section>
            <h2>Rebased NAV</h2>
            <div id="nav-chart" class="chart chart-small"></div>
          </section>
        </div>
      </section>
      <aside id="inspector" class="side-panel"></aside>
    </main>
    <details class="panel comparison-shell">
      <summary>Model comparison and buy/sell feature differences</summary>
      <section id="comparison-panel"></section>
    </details>
  `;
}

function selectedModels(data: DashboardData, selectedProfiles: string[]): Record<string, ModelPayload> {
  return Object.fromEntries(Object.entries(data.models).filter(([profile]) => selectedProfiles.includes(profile)));
}

function clippedMarket(data: DashboardData, start: string, end: string): DashboardData['market'] {
  return { ...data.market, rows: clipRowsByDate(data.market.rows, start, end) };
}

function clippedModel(model: ModelPayload, start: string, end: string): ModelPayload {
  return {
    ...model,
    rows: clipRowsByDate(model.rows, start, end),
    events: clipRowsByDate(model.events, start, end),
  };
}

function clippedModels(models: Record<string, ModelPayload>, start: string, end: string): Record<string, ModelPayload> {
  return Object.fromEntries(Object.entries(models).map(([profile, model]) => [profile, clippedModel(model, start, end)]));
}

function visibleContext(data: DashboardData, state: ViewState) {
  const models = clippedModels(selectedModels(data, state.selectedProfiles), state.start, state.end);
  const market = clippedMarket(data, state.start, state.end);
  const primary = models[state.baselineProfile] ?? Object.values(models)[0];
  return { models, market, primary };
}

let disposeChartSync: (() => void) | null = null;

function renderToolbar(
  container: HTMLElement,
  data: DashboardData,
  state: ViewState,
  update: UpdateState,
): void {
  container.replaceChildren();

  const profileGroup = document.createElement('div');
  profileGroup.className = 'toolbar-group profile-group';
  profileGroup.append('Models');
  for (const profile of data.manifest.default_comparison_profiles) {
    const label = document.createElement('label');
    label.className = 'check-label';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = state.selectedProfiles.includes(profile);
    input.addEventListener('change', () => {
      const raw = input.checked
        ? [...state.selectedProfiles, profile]
        : state.selectedProfiles.filter((item) => item !== profile);
      update({ selectedProfiles: clampComparisonProfiles(raw, data.manifest.default_comparison_profiles) });
    });
    label.title = profile;
    label.append(input, document.createTextNode(formatProfileLabel(profile)));
    profileGroup.appendChild(label);
  }
  container.appendChild(profileGroup);

  const baselineGroup = document.createElement('label');
  baselineGroup.className = 'toolbar-group';
  baselineGroup.append('Action Model');
  const baseline = document.createElement('select');
  for (const profile of state.selectedProfiles) {
    const option = document.createElement('option');
    option.value = profile;
    option.textContent = formatProfileLabel(profile);
    option.title = profile;
    option.selected = profile === state.baselineProfile;
    baseline.appendChild(option);
  }
  baseline.addEventListener('change', () => update({ baselineProfile: baseline.value }));
  baselineGroup.appendChild(baseline);
  container.appendChild(baselineGroup);

  const rangeGroup = document.createElement('div');
  rangeGroup.className = 'toolbar-group range-group';
  for (const preset of ['1M', '3M', 'YTD', 'ALL'] as const satisfies readonly DateRangePreset[]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'range-button';
    button.textContent = preset;
    button.addEventListener('click', () => {
      const range = computeDateRangePreset(data.market.rows, preset);
      if (range) update({ ...range, hoveredDate: range.end });
    });
    rangeGroup.appendChild(button);
  }
  container.appendChild(rangeGroup);

  for (const [labelText, key] of [
    ['Start', 'start'],
    ['End', 'end'],
  ] as const) {
    const label = document.createElement('label');
    label.className = 'toolbar-group';
    label.append(labelText);
    const input = document.createElement('input');
    input.type = 'date';
    input.value = state[key];
    input.addEventListener('change', () => update({ [key]: input.value }));
    label.appendChild(input);
    container.appendChild(label);
  }
}

function warningsFor(data: DashboardData, state: ViewState): string[] {
  const models = selectedModels(data, state.selectedProfiles);
  const common = computeCommonWindow(data.market.rows, Object.values(models).map((model) => model.rows));
  const warnings: string[] = [];
  if (!common) warnings.push('Selected models have no common overlap with SOXL market rows.');
  if (data.manifest.safe_backtest_end < data.manifest.latest_market_date) {
    warnings.push(`Replay safe end ${data.manifest.safe_backtest_end} is before market latest ${data.manifest.latest_market_date}.`);
  }
  if (data.comparison.common_end < state.end) {
    warnings.push(
      `Common model comparison ends ${data.comparison.common_end}; SOXL price and reference baseline may extend to ${state.end}.`,
    );
  }
  return warnings;
}

function renderInspector(data: DashboardData, state: ViewState): void {
  const inspector = requireElement('inspector');
  const { models, market } = visibleContext(data, state);
  renderInspectorPanel(
    inspector,
    computeInspectorRows({
      date: state.hoveredDate,
      manifest: data.manifest,
      market,
      models,
      comparison: data.comparison,
    }),
  );
}

function renderReadoutPanels(data: DashboardData, state: ViewState, update: UpdateState): void {
  const toolbar = requireElement('toolbar');
  const comparison = requireElement('comparison-panel');
  const source = requireElement('source-panel');
  const deltaStrip = requireElement('model-delta-strip');
  const mismatchRows = computeActionMismatchRows(data.comparison, data.market, state.start, state.end);

  renderToolbar(toolbar, data, state, update);
  renderInspector(data, state);
  renderModelDeltaStrip(
    deltaStrip,
    data.comparison,
    state.start,
    state.end,
    state.selectedProfiles,
    data.manifest,
    mismatchRows,
    data.models,
  );
  renderComparisonPanel(
    comparison,
    data.comparison,
    state.start,
    state.end,
    mismatchRows,
    (date) => {
      const range = computeFocusedDateRange(data.market.rows, date, 10);
      update(range ? { ...range, hoveredDate: date } : { hoveredDate: date });
    },
  );
  renderDataSourcePanel(source, data.manifest, warningsFor(data, state));
}

function renderDashboard(data: DashboardData, state: ViewState, update: UpdateState): void {
  const marketChart = requireElement('market-chart');
  const positionChart = requireElement('position-chart');
  const deltaNavChart = requireElement('delta-nav-chart');
  const navChart = requireElement('nav-chart');
  const actionLegend = requireElement('action-legend');
  const { models, market, primary } = visibleContext(data, state);

  disposeChartSync?.();
  disposeChartSync = null;
  renderActionLegend(actionLegend);
  const marketHandles = renderMarketChart(
    marketChart,
    market,
    primary,
    models,
    (date) => update({ hoveredDate: date }),
    data.comparison,
    state.selectedProfiles,
  );
  const positionHandles = renderPositionChart(
    positionChart,
    models,
    primary.profile,
  );
  const deltaNavHandles = renderDeltaChart(
    deltaNavChart,
    data.comparison,
    state.selectedProfiles,
    'nav_delta_vs_baseline',
    state.start,
    state.end,
  );
  const navHandles = renderNavChart(
    navChart,
    models,
  );
  const timeCleanup = syncVisibleTimeRanges(
    [marketHandles.chart, positionHandles.chart, deltaNavHandles.chart, navHandles.chart],
    (logicalRange) => {
      const range = computeDateRangeFromLogicalRange(market.rows, logicalRange);
      if (!range || (range.start === state.start && range.end === state.end)) return;
      update({ ...range, hoveredDate: range.end }, { skipChartRender: true });
    },
  );
  const crosshairCleanup = syncCrosshairPositions([marketHandles, positionHandles, deltaNavHandles, navHandles]);
  disposeChartSync = () => {
    timeCleanup();
    crosshairCleanup();
  };
  renderReadoutPanels(data, state, update);
}

function keyboardRangeAction(event: KeyboardEvent): KeyboardRangeAction | null {
  if (event.key === '+' || event.key === '=') return 'zoom_in';
  if (event.key === '-' || event.key === '_') return 'zoom_out';
  if (event.key === 'ArrowLeft') return 'pan_left';
  if (event.key === 'ArrowRight') return 'pan_right';
  if (event.key.toLowerCase() === 'r') return 'reset';
  return null;
}

function isEditingTarget(target: EventTarget | null): boolean {
  const element = target instanceof HTMLElement ? target : null;
  return element ? ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(element.tagName) : false;
}

async function main(): Promise<void> {
  const app = requireElement('app');
  createShell(app);
  try {
    const data = await loadDashboardData();
    const models = selectedModels(data, data.manifest.default_comparison_profiles);
    const defaultRange = computeDefaultDateRange({
      manifest: data.manifest,
      marketRows: data.market.rows,
      models,
      comparison: data.comparison,
    });
    const common = defaultRange ?? computeCommonWindow(data.market.rows, Object.values(models).map((model) => model.rows));
    const state: ViewState = {
      selectedProfiles: clampComparisonProfiles(
        data.manifest.default_comparison_profiles,
        data.manifest.default_comparison_profiles,
      ),
      baselineProfile: selectDefaultChartProfile(
        data.manifest.default_comparison_profiles,
        models,
        data.manifest.default_baseline_profile,
      ),
      start: common?.start ?? data.comparison.common_start,
      end: common?.end ?? data.comparison.common_end,
      hoveredDate: common?.end ?? data.comparison.common_end,
    };
    const update: UpdateState = (next, options = {}) => {
      const keys = Object.keys(next);
      const hoverOnly = next.hoveredDate !== undefined && keys.length === 1;
      if (hoverOnly && next.hoveredDate === state.hoveredDate) {
        return;
      }
      const needsChartRender = !options.skipChartRender && viewUpdateRequiresChartRender(next);
      Object.assign(state, next);
      if (!state.selectedProfiles.includes(state.baselineProfile)) {
        state.baselineProfile = selectDefaultChartProfile(
          state.selectedProfiles,
          selectedModels(data, state.selectedProfiles),
          data.manifest.default_baseline_profile,
        );
      }
      if (needsChartRender) {
        renderDashboard(data, state, update);
      } else if (hoverOnly) {
        renderInspector(data, state);
      } else {
        renderReadoutPanels(data, state, update);
      }
    };
    window.addEventListener('keydown', (event) => {
      if (isEditingTarget(event.target)) return;
      const action = keyboardRangeAction(event);
      if (!action) return;
      const range = computeKeyboardDateRange(data.market.rows, state.start, state.end, action);
      if (!range) return;
      event.preventDefault();
      update({ ...range, hoveredDate: range.end });
    });
    renderDashboard(data, state, update);
  } catch (error) {
    app.replaceChildren();
    const message = document.createElement('pre');
    message.className = 'load-error';
    message.textContent = error instanceof Error ? error.message : String(error);
    app.appendChild(message);
  }
}

void main();
