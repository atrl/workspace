import './styles.css';
import {
  applyChartRangeCommand,
  renderMarketChart,
  renderNavChart,
  renderPositionChart,
  syncVisibleLogicalRanges,
  type ChartRangeCommand,
  type ChartRangeTarget,
} from './chart';
import {
  buildInspectorLookup,
  chartRangeRows,
  clampComparisonProfiles,
  clipRowsByDate,
  computeCommonWindow,
  computeDateRangePreset,
  computeDefaultChartWindow,
  computeInspectorRows,
  computeInspectorRowsFromLookup,
  chooseDefaultChartProfile,
  type DateRangePreset,
  type InspectorLookup,
} from './domain';
import { loadDashboardData, type DashboardData } from './data';
import { runBacktest } from './api';
import {
  renderActionLegend,
  renderChartCommandBar,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderFeatureSpotlightPanel,
  renderFeatureSnapshotPanel,
  renderInspectorPanel,
  renderOperationsPanel,
} from './render';
import type { ModelPayload } from './types';
import { viewUpdateRequiresChartRender } from './viewState';

interface ViewState {
  selectedProfiles: string[];
  baselineProfile: string;
  chartProfile: string;
  start: string;
  end: string;
  hoveredDate: string;
}

function requireElement(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element: ${id}`);
  return node;
}

function createShell(app: HTMLElement): void {
  app.innerHTML = `
    <header id="toolbar" class="toolbar"></header>
    <main id="workspace" class="workspace">
      <section class="chart-column">
        <div class="chart-title-row">
          <h1>SOXL</h1>
          <span>TradingView-like replay review</span>
          <div id="chart-command-bar" class="chart-command-bar" aria-label="Chart controls"></div>
          <div id="action-legend" class="action-legend"></div>
        </div>
        <div id="market-chart" class="chart chart-main"></div>
        <div class="lower-charts">
          <section>
            <h2>Position</h2>
            <div id="position-chart" class="chart chart-small"></div>
          </section>
          <section>
            <h2>NAV</h2>
            <div id="nav-chart" class="chart chart-small"></div>
          </section>
        </div>
      </section>
      <aside class="right-stack">
        <section id="inspector" class="side-panel"></section>
        <section id="operations-panel" class="side-panel operations-panel"></section>
        <section id="feature-panel" class="side-panel feature-panel"></section>
      </aside>
    </main>
    <section id="comparison-panel" class="panel"></section>
    <section id="source-panel" class="panel"></section>
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
  const primary = clippedModel(data.models[state.chartProfile] ?? Object.values(data.models)[0], state.start, state.end);
  return { models, market, primary };
}

function selectableChartProfiles(data: DashboardData, selectedProfiles: readonly string[]): string[] {
  const selected = selectedProfiles.filter((profile) => Boolean(data.models[profile]));
  const runnable = selected.filter((profile) => data.models[profile].runnable);
  return runnable.length ? runnable : selected;
}

function profileOptionLabel(model: ModelPayload | undefined, profile: string): string {
  if (!model) return profile;
  const name = model.display_name || profile;
  if (model.runnable) return `${name} [run]`;
  if (model.runnable_reason === 'reference_baseline' || model.source_kind === 'soxlbacktest_qlib_buysell') {
    return `${name} [baseline]`;
  }
  return `${name} [evidence]`;
}

function apiProfileFor(chartProfile: string): string {
  return chartProfile.startsWith('rerun:') ? chartProfile.split(':')[1] : chartProfile;
}

function renderToolbar(
  container: HTMLElement,
  data: DashboardData,
  state: ViewState,
  update: (next: Partial<ViewState>) => void,
  runInteractiveBacktest: () => void,
): void {
  container.replaceChildren();

  const profileGroup = document.createElement('div');
  profileGroup.className = 'toolbar-group profile-group';
  profileGroup.append('Models');
  for (const profile of data.manifest.available_profiles) {
    const label = document.createElement('label');
    label.className = 'check-label';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = state.selectedProfiles.includes(profile);
    input.addEventListener('change', () => {
      const raw = input.checked
        ? [...state.selectedProfiles, profile]
        : state.selectedProfiles.filter((item) => item !== profile);
      update({ selectedProfiles: clampComparisonProfiles(raw, data.manifest.available_profiles) });
    });
    label.append(input, document.createTextNode(profileOptionLabel(data.models[profile], profile)));
    profileGroup.appendChild(label);
  }
  container.appendChild(profileGroup);

  const baselineGroup = document.createElement('label');
  baselineGroup.className = 'toolbar-group';
  baselineGroup.append('Baseline');
  const baseline = document.createElement('select');
  for (const profile of state.selectedProfiles) {
    const option = document.createElement('option');
    option.value = profile;
    option.textContent = profile;
    option.selected = profile === state.baselineProfile;
    baseline.appendChild(option);
  }
  baseline.addEventListener('change', () => update({ baselineProfile: baseline.value }));
  baselineGroup.appendChild(baseline);
  container.appendChild(baselineGroup);

  const chartGroup = document.createElement('label');
  chartGroup.className = 'toolbar-group';
  chartGroup.append('Chart');
  const chartSelect = document.createElement('select');
  for (const profile of selectableChartProfiles(data, state.selectedProfiles)) {
    const option = document.createElement('option');
    option.value = profile;
    option.textContent = profileOptionLabel(data.models[profile], profile);
    option.selected = profile === state.chartProfile;
    chartSelect.appendChild(option);
  }
  chartSelect.addEventListener('change', () => {
    const chartProfile = chartSelect.value;
    const chartWindow =
      computeDateRangePreset(chartRangeRows(data.models, chartProfile, data.market.rows), 'YTD') ??
      computeDefaultChartWindow(data.market.rows, data.models[chartProfile]?.rows ?? []);
    update({ chartProfile, ...(chartWindow ?? {}), hoveredDate: chartWindow?.end ?? state.hoveredDate });
  });
  chartGroup.appendChild(chartSelect);
  container.appendChild(chartGroup);

  const rangeGroup = document.createElement('div');
  rangeGroup.className = 'toolbar-group range-group';
  for (const preset of ['1M', '3M', '6M', 'YTD', '1Y', 'ALL'] as const satisfies readonly DateRangePreset[]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'range-button';
    button.textContent = preset;
    button.addEventListener('click', () => {
      const range = computeDateRangePreset(chartRangeRows(data.models, state.chartProfile, data.market.rows), preset);
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

  const runGroup = document.createElement('div');
  runGroup.className = 'toolbar-group run-group';
  const runButton = document.createElement('button');
  runButton.type = 'button';
  runButton.textContent = data.models[state.chartProfile]?.runnable ? 'Run inference' : 'Evidence only';
  runButton.disabled = !data.models[state.chartProfile]?.runnable;
  runButton.addEventListener('click', runInteractiveBacktest);
  runGroup.appendChild(runButton);
  container.appendChild(runGroup);
}

function warningsFor(data: DashboardData, state: ViewState): string[] {
  const models = selectedModels(data, state.selectedProfiles);
  const common = computeCommonWindow(data.market.rows, Object.values(models).map((model) => model.rows));
  const warnings: string[] = [];
  if (!common) warnings.push('Selected models have no common overlap with SOXL market rows.');
  if (data.manifest.safe_backtest_end < data.manifest.latest_market_date) {
    warnings.push(`Replay safe end ${data.manifest.safe_backtest_end} is before market latest ${data.manifest.latest_market_date}.`);
  }
  const freshness = data.manifest.auto_update?.model_replay_freshness;
  if (freshness && (freshness.ready_count ?? 0) === 0) {
    warnings.push(
      `Replay refresh found 0 runnable model profiles; dashboard is using retained exports/reference rows only.`,
    );
  }
  if (!Object.values(models).some((model) => model.runnable)) {
    warnings.push('No selected model is runnable for interactive inference. Select a profile with a retained actor replay.');
  }
  for (const profile of state.selectedProfiles) {
    const model = data.models[profile];
    if (model && !model.runnable) {
      const reason =
        model.runnable_reason === 'reference_baseline'
          ? 'reference baseline'
          : model.runnable_reason
            ? `not runnable: ${model.runnable_reason}`
            : 'not runnable';
      warnings.push(`${profile} is ${reason}; interactive inference is disabled for this model.`);
    }
  }
  return warnings;
}

function buildCurrentInspectorLookup(data: DashboardData, state: ViewState): InspectorLookup {
  return buildInspectorLookup({
    market: data.market,
    models: selectedModels(data, state.selectedProfiles),
    comparison: data.comparison,
  });
}

function renderInspector(data: DashboardData, state: ViewState, lookup?: InspectorLookup): void {
  const inspector = requireElement('inspector');
  const featurePanel = requireElement('feature-panel');
  const { models } = visibleContext(data, state);
  const rows = lookup
    ? computeInspectorRowsFromLookup({
        date: state.hoveredDate,
        manifest: data.manifest,
        lookup,
      })
    : computeInspectorRows({
        date: state.hoveredDate,
        manifest: data.manifest,
        market: data.market,
        models,
        comparison: data.comparison,
      });
  renderInspectorPanel(
    inspector,
    rows,
  );
  const featureSnapshotPanel = document.getElementById('feature-date-snapshot') ?? featurePanel;
  renderFeatureSnapshotPanel(featureSnapshotPanel, {
    date: state.hoveredDate,
    marketRows: data.market.rows,
    models: selectedModels(data, state.selectedProfiles),
  });
}

interface DashboardRenderResult {
  inspectorLookup: InspectorLookup;
  chartHandles: ChartRangeTarget[];
  chartSyncCleanup: () => void;
}

function renderDashboard(
  data: DashboardData,
  state: ViewState,
  update: (next: Partial<ViewState>) => void,
  onChartCommand: (command: ChartRangeCommand) => void,
  runInteractiveBacktest: () => void,
): DashboardRenderResult {
  const toolbar = requireElement('toolbar');
  const marketChart = requireElement('market-chart');
  const positionChart = requireElement('position-chart');
  const navChart = requireElement('nav-chart');
  const comparison = requireElement('comparison-panel');
  const source = requireElement('source-panel');
  const operations = requireElement('operations-panel');
  const actionLegend = requireElement('action-legend');
  const chartCommandBar = requireElement('chart-command-bar');
  const { models, market, primary } = visibleContext(data, state);
  const inspectorLookup = buildCurrentInspectorLookup(data, state);

  renderToolbar(toolbar, data, state, update, runInteractiveBacktest);
  renderActionLegend(actionLegend);
  renderChartCommandBar(chartCommandBar, onChartCommand);
  const marketHandles = renderMarketChart(marketChart, market, primary, (date) => update({ hoveredDate: date }));
  const positionHandles = renderPositionChart(positionChart, primary, (date) => update({ hoveredDate: date }));
  const navHandles = renderNavChart(navChart, models, state.baselineProfile, (date) => update({ hoveredDate: date }));
  const chartSyncCleanup = syncVisibleLogicalRanges([marketHandles.chart, positionHandles.chart, navHandles]);
  renderComparisonPanel(comparison, data.comparison, state.start, state.end, state.selectedProfiles, data.models);
  renderFeatureSpotlightPanel(requireElement('feature-panel'), {
    date: state.hoveredDate,
    marketRows: market.rows,
    models,
    baselineProfile: state.baselineProfile,
  });
  renderInspector(data, state, inspectorLookup);
  const warnings = warningsFor(data, state);
  renderOperationsPanel(operations, data.manifest, warnings);
  renderDataSourcePanel(source, data.manifest, warnings);
  return {
    inspectorLookup,
    chartHandles: [marketHandles.chart, positionHandles.chart, navHandles],
    chartSyncCleanup,
  };
}

function keyboardRangeAction(event: KeyboardEvent): ChartRangeCommand | null {
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
    const selectedProfiles = clampComparisonProfiles(
      data.manifest.default_comparison_profiles,
      data.manifest.available_profiles,
    );
    const defaultChartProfile = chooseDefaultChartProfile(data.manifest, data.models);
    const chartWindow =
      computeDateRangePreset(chartRangeRows(data.models, defaultChartProfile, data.market.rows), 'YTD') ??
      computeDefaultChartWindow(data.market.rows, data.models[defaultChartProfile]?.rows ?? []);
    const common = computeCommonWindow(data.market.rows, Object.values(models).map((model) => model.rows));
    const state: ViewState = {
      selectedProfiles,
      baselineProfile: data.manifest.default_baseline_profile,
      chartProfile: defaultChartProfile,
      start: chartWindow?.start ?? common?.start ?? data.comparison.common_start,
      end: chartWindow?.end ?? common?.end ?? data.comparison.common_end,
      hoveredDate: chartWindow?.end ?? common?.end ?? data.comparison.common_end,
    };
    let inspectorLookup: InspectorLookup | null = null;
    let chartHandles: ChartRangeTarget[] = [];
    let chartSyncCleanup: (() => void) | null = null;
    const runChartCommand = (command: ChartRangeCommand) => applyChartRangeCommand(chartHandles, command);
    const rerender = () => {
      chartSyncCleanup?.();
      const result = renderDashboard(data, state, update, runChartCommand, runInteractiveBacktest);
      inspectorLookup = result.inspectorLookup;
      chartHandles = result.chartHandles;
      chartSyncCleanup = result.chartSyncCleanup;
    };
    const update = (next: Partial<ViewState>) => {
      if (next.hoveredDate !== undefined && next.hoveredDate === state.hoveredDate && Object.keys(next).length === 1) {
        return;
      }
      const needsChartRender = viewUpdateRequiresChartRender(next);
      Object.assign(state, next);
      if (!state.selectedProfiles.includes(state.baselineProfile)) {
        state.baselineProfile = state.selectedProfiles[0];
      }
      const chartProfiles = selectableChartProfiles(data, state.selectedProfiles);
      if (!chartProfiles.includes(state.chartProfile)) {
        state.chartProfile = chartProfiles[0] ?? state.selectedProfiles[0];
      }
      if (needsChartRender) {
        rerender();
      } else if (inspectorLookup) {
        renderInspector(data, state, inspectorLookup);
      }
    };
    async function runInteractiveBacktest(): Promise<void> {
      const sourceProfile = apiProfileFor(state.chartProfile);
      const rerunModel = await runBacktest({
        profile: sourceProfile,
        startDate: state.start,
        endDate: state.end,
      });
      data.models[rerunModel.profile] = rerunModel;
      data.manifest.available_profiles = [
        rerunModel.profile,
        ...data.manifest.available_profiles.filter((profile) => profile !== rerunModel.profile),
      ];
      state.selectedProfiles = clampComparisonProfiles(
        [rerunModel.profile, ...state.selectedProfiles],
        data.manifest.available_profiles,
      );
      state.chartProfile = rerunModel.profile;
      state.start = rerunModel.replay_start;
      state.end = rerunModel.replay_end;
      state.hoveredDate = rerunModel.replay_end;
      rerender();
    }
    window.addEventListener('keydown', (event) => {
      if (isEditingTarget(event.target)) return;
      const action = keyboardRangeAction(event);
      if (!action) return;
      event.preventDefault();
      runChartCommand(action);
    });
    rerender();
  } catch (error) {
    app.replaceChildren();
    const message = document.createElement('pre');
    message.className = 'load-error';
    message.textContent = error instanceof Error ? error.message : String(error);
    app.appendChild(message);
  }
}

void main();
