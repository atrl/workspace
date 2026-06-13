import './styles.css';
import { renderMarketChart, renderNavChart, renderPositionChart } from './chart';
import { clampComparisonProfiles, clipRowsByDate, computeCommonWindow, computeInspectorRows } from './domain';
import { loadDashboardData, type DashboardData } from './data';
import { renderActionLegend, renderComparisonPanel, renderDataSourcePanel, renderInspectorPanel } from './render';
import type { ModelPayload } from './types';
import { viewUpdateRequiresChartRender } from './viewState';

interface ViewState {
  selectedProfiles: string[];
  baselineProfile: string;
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
      <aside id="inspector" class="side-panel"></aside>
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
  const primary = models[state.baselineProfile] ?? Object.values(models)[0];
  return { models, market, primary };
}

function renderToolbar(
  container: HTMLElement,
  data: DashboardData,
  state: ViewState,
  update: (next: Partial<ViewState>) => void,
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
    label.append(input, document.createTextNode(profile));
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

function renderDashboard(data: DashboardData, state: ViewState, update: (next: Partial<ViewState>) => void): void {
  const toolbar = requireElement('toolbar');
  const marketChart = requireElement('market-chart');
  const positionChart = requireElement('position-chart');
  const navChart = requireElement('nav-chart');
  const comparison = requireElement('comparison-panel');
  const source = requireElement('source-panel');
  const actionLegend = requireElement('action-legend');
  const { models, market, primary } = visibleContext(data, state);

  renderToolbar(toolbar, data, state, update);
  renderActionLegend(actionLegend);
  renderMarketChart(marketChart, market, primary, (date) => update({ hoveredDate: date }));
  renderPositionChart(positionChart, primary);
  renderNavChart(navChart, models);
  renderInspector(data, state);
  renderComparisonPanel(comparison, data.comparison, state.start, state.end);
  renderDataSourcePanel(source, data.manifest, warningsFor(data, state));
}

async function main(): Promise<void> {
  const app = requireElement('app');
  createShell(app);
  try {
    const data = await loadDashboardData();
    const models = selectedModels(data, data.manifest.default_comparison_profiles);
    const common = computeCommonWindow(data.market.rows, Object.values(models).map((model) => model.rows));
    const state: ViewState = {
      selectedProfiles: clampComparisonProfiles(
        data.manifest.default_comparison_profiles,
        data.manifest.default_comparison_profiles,
      ),
      baselineProfile: data.manifest.default_baseline_profile,
      start: common?.start ?? data.comparison.common_start,
      end: common?.end ?? data.comparison.common_end,
      hoveredDate: common?.end ?? data.comparison.common_end,
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
      if (needsChartRender) {
        renderDashboard(data, state, update);
      } else {
        renderInspector(data, state);
      }
    };
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
