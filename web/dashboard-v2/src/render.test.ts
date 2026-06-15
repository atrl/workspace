import { describe, expect, test } from 'vitest';
import { renderActionLegend, renderComparisonPanel, renderDataSourcePanel, renderInspectorPanel } from './render';
import { comparisonFixture, manifestFixture } from './testFixtures';

describe('dashboard v2 DOM render helpers', () => {
  test('renders data source provenance and safe replay window', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, manifestFixture);

    expect(panel.querySelector('.status-strip')).not.toBeNull();
    expect(panel.textContent).toContain('Replay safe');
    expect(panel.textContent).toContain('2026-01-06');
    expect(panel.textContent).not.toContain('model_replay_freshness.json');
  });

  test('keeps source warnings inside a collapsed diagnostics block', () => {
    const panel = document.createElement('section');

    renderDataSourcePanel(panel, manifestFixture, ['Replay safe end is before market latest.']);

    const diagnostics = panel.querySelector<HTMLDetailsElement>('details.source-diagnostics');
    expect(diagnostics).not.toBeNull();
    expect(diagnostics?.open).toBe(false);
    expect(diagnostics?.querySelector('summary')?.textContent).toContain('Diagnostics');
    expect(diagnostics?.textContent).toContain('Replay safe end is before market latest.');
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

    expect(panel.querySelector('.comparison-cards')).not.toBeNull();
    expect(panel.querySelector('.table-scroll .comparison-table')).toBeNull();
    expect(panel.textContent).toContain('smoke');
    expect(panel.textContent).toContain('balanced_rule_residual_v22');
    expect(panel.textContent).toContain('Mismatch Days');
    expect(panel.textContent).toContain('1');
  });

  test('renders clickable action mismatch rows with feature context', () => {
    const panel = document.createElement('section');
    const clicked: string[] = [];

    renderComparisonPanel(
      panel,
      comparisonFixture,
      '2026-01-02',
      '2026-01-06',
      [
        {
          date: '2026-01-05',
          profile: 'smoke',
          model_action: 'BUY',
          baseline_action: 'HOLD',
          model_position: 0.7,
          baseline_position: 0,
          position_delta_vs_baseline: 0.7,
          nav_delta_vs_baseline: 5,
          close: 11,
          features: { rsi6: '60', vix: '18' },
        },
      ],
      (date) => clicked.push(date),
    );

    expect(panel.textContent).toContain('Mismatch Days');
    const diagnostics = panel.querySelector<HTMLDetailsElement>('details.comparison-diagnostics');
    expect(diagnostics).not.toBeNull();
    expect(diagnostics?.open).toBe(false);
    expect(diagnostics?.querySelector('summary')?.textContent).toContain('Action mismatches');
    expect(panel.textContent).toContain('BUY vs HOLD');
    expect(panel.textContent).toContain('rsi6 60');
    panel.querySelector<HTMLButtonElement>('.mismatch-date-button')?.click();
    expect(clicked).toEqual(['2026-01-05']);
  });

  test('caps action mismatch rows to a compact recent list', () => {
    const panel = document.createElement('section');
    const mismatchRows = Array.from({ length: 20 }, (_, index) => ({
      date: `2026-01-${String(index + 1).padStart(2, '0')}`,
      profile: 'smoke',
      model_action: 'BUY' as const,
      baseline_action: 'HOLD' as const,
      model_position: 0.7,
      baseline_position: 0,
      position_delta_vs_baseline: 0.7,
      nav_delta_vs_baseline: index,
      close: 10 + index,
      features: { rsi6: String(50 + index) },
    }));

    renderComparisonPanel(panel, comparisonFixture, '2026-01-02', '2026-01-06', mismatchRows);

    expect(panel.querySelectorAll('.mismatch-row')).toHaveLength(12);
    expect(panel.textContent).toContain('2026-01-20');
    expect(panel.textContent).not.toContain('2026-01-01');
  });

  test('renders visible buy and sell action labels', () => {
    const panel = document.createElement('div');

    renderActionLegend(panel);

    expect(panel.textContent).toContain('BUY');
    expect(panel.textContent).toContain('SELL');
  });
});
