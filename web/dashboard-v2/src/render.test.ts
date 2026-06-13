import { describe, expect, test, vi } from 'vitest';
import {
  renderActionLegend,
  renderChartCommandBar,
  renderComparisonPanel,
  renderDataSourcePanel,
  renderFeatureComparisonPanel,
  renderInspectorPanel,
} from './render';
import { comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';

describe('dashboard v2 DOM render helpers', () => {
  test('renders data source provenance and safe replay window', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, manifestFixture);

    expect(panel.textContent).toContain('safe_backtest_end');
    expect(panel.textContent).toContain('2026-01-06');
    expect(panel.textContent).toContain('model_replay_freshness.json');
    expect(panel.textContent).toContain('Auto Update');
    expect(panel.textContent).toContain('postclose');
  });

  test('renders inspector rows with tone classes', () => {
    const panel = document.createElement('aside');

    renderInspectorPanel(panel, [
      { label: 'SOXL close', value: '11.00' },
      { label: 'smoke action', value: 'BUY', tone: 'buy' },
    ]);

    expect(panel.querySelector('[data-tone="buy"]')?.textContent).toContain('BUY');
  });

  test('renders comparison panel with model deltas and action mismatch counts', () => {
    const panel = document.createElement('section');

    renderComparisonPanel(panel, comparisonFixture, '2026-01-02', '2026-01-06');

    expect(panel.querySelector('.table-scroll .comparison-table')).not.toBeNull();
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('Action mismatches');
    expect(panel.textContent).toContain('BuyHold');
    expect(panel.textContent).toContain('Vs BuyHold');
    expect(panel.textContent).toContain('-5.00pp');
    expect(panel.textContent).toContain('1');
  });

  test('labels comparison panel with actual visible comparison row dates', () => {
    const panel = document.createElement('section');

    renderComparisonPanel(panel, comparisonFixture, '2020-01-02', '2026-06-11');

    expect(panel.textContent).toContain('3 rows | baseline balanced_rule_residual_v22 | 2026-01-02 to 2026-01-06');
  });

  test('renders visible buy and sell action labels', () => {
    const panel = document.createElement('div');

    renderActionLegend(panel);

    expect(panel.textContent).toContain('BUY');
    expect(panel.textContent).toContain('SELL');
  });

  test('renders chart command buttons that emit trading view range commands', () => {
    const panel = document.createElement('div');
    const onCommand = vi.fn();

    renderChartCommandBar(panel, onCommand);

    const zoomIn = panel.querySelector<HTMLButtonElement>('button[aria-label="Zoom in"]');
    const panRight = panel.querySelector<HTMLButtonElement>('button[aria-label="Pan right"]');
    const latest = panel.querySelector<HTMLButtonElement>('button[aria-label="Follow latest"]');
    expect(panel.querySelectorAll('button')).toHaveLength(6);
    expect(zoomIn?.textContent).toBe('+');
    expect(panRight?.textContent).toBe('>');
    expect(latest?.textContent).toBe('>|');

    zoomIn?.click();
    panRight?.click();
    latest?.click();

    expect(onCommand).toHaveBeenNthCalledWith(1, 'zoom_in');
    expect(onCommand).toHaveBeenNthCalledWith(2, 'pan_right');
    expect(onCommand).toHaveBeenNthCalledWith(3, 'latest');
  });

  test('renders buy sell feature comparison for selected date', () => {
    const panel = document.createElement('section');

    renderFeatureComparisonPanel(panel, {
      date: '2026-01-05',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture },
    });

    expect(panel.textContent).toContain('Buy/Sell Feature Comparison');
    expect(panel.textContent).toContain('rulebase_signals');
    expect(panel.textContent).toContain('rsi6');
    expect(panel.textContent).toContain('60.00');
  });

  test('renders recent buy sell event feature snapshots without relying on hover date', () => {
    const panel = document.createElement('section');

    renderFeatureComparisonPanel(panel, {
      date: '2026-01-02',
      marketRows: marketFixture.rows,
      models: { smoke: modelFixture },
    });

    expect(panel.textContent).toContain('Recent Buy/Sell Events');
    expect(panel.textContent).toContain('2026-01-05');
    expect(panel.textContent).toContain('BUY');
    expect(panel.textContent).toContain('2026-01-06');
    expect(panel.textContent).toContain('SELL');
    expect(panel.textContent).toContain('58.00');
  });
});
