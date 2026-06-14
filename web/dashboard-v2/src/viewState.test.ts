import { describe, expect, test } from 'vitest';
import {
  applyChartDateInteraction,
  viewUpdateRequiresChartRender,
  viewUpdateRequiresFeatureRender,
} from './viewState';

describe('dashboard v2 view state interactions', () => {
  test('uses chart clicks to set the rerun start date without leaving range selection armed', () => {
    expect(
      applyChartDateInteraction(
        {
          chartClickMode: 'set_start',
          start: '2026-01-02',
          end: '2026-06-09',
          hoveredDate: '2026-06-09',
          selectedDate: '2026-06-09',
        },
        '2026-03-16',
        'select',
      ),
    ).toEqual({
      start: '2026-03-16',
      hoveredDate: '2026-03-16',
      selectedDate: '2026-03-16',
      chartClickMode: 'inspect',
    });
  });

  test('keeps the selected rerun window valid when chart clicks set endpoints out of order', () => {
    expect(
      applyChartDateInteraction(
        {
          chartClickMode: 'set_end',
          start: '2026-03-16',
          end: '2026-06-09',
          hoveredDate: '2026-06-09',
          selectedDate: '2026-06-09',
        },
        '2026-02-02',
        'select',
      ),
    ).toEqual({
      start: '2026-02-02',
      end: '2026-02-02',
      hoveredDate: '2026-02-02',
      selectedDate: '2026-02-02',
      chartClickMode: 'inspect',
    });
  });

  test('treats interaction-mode changes as chrome-only updates', () => {
    expect(viewUpdateRequiresChartRender({ chartClickMode: 'set_start' })).toBe(false);
    expect(viewUpdateRequiresFeatureRender({ chartClickMode: 'set_end' })).toBe(false);
  });
});
