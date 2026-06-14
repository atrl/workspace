import type { ChartDateInteraction } from './chart';

export type ChartClickMode = 'inspect' | 'set_start' | 'set_end';

export interface ViewUpdate {
  selectedProfiles?: string[];
  baselineProfile?: string;
  chartProfile?: string;
  runProfile?: string;
  start?: string;
  end?: string;
  hoveredDate?: string;
  selectedDate?: string;
  chartClickMode?: ChartClickMode;
}

export interface ChartDateState {
  chartClickMode: ChartClickMode;
  start: string;
  end: string;
  hoveredDate: string;
  selectedDate: string;
}

export function viewUpdateRequiresChartRender(update: ViewUpdate): boolean {
  return (
    update.selectedProfiles !== undefined ||
    update.baselineProfile !== undefined ||
    update.chartProfile !== undefined ||
    update.start !== undefined ||
    update.end !== undefined
  );
}

export function viewUpdateRequiresFeatureRender(update: ViewUpdate): boolean {
  return update.selectedDate !== undefined || viewUpdateRequiresChartRender(update);
}

export function viewUpdateRequiresToolbarRender(update: ViewUpdate): boolean {
  return update.runProfile !== undefined;
}

export function applyChartDateInteraction(
  state: ChartDateState,
  date: string,
  interaction: ChartDateInteraction,
): ViewUpdate {
  if (interaction === 'hover') return { hoveredDate: date };
  if (state.chartClickMode === 'set_start') {
    return {
      start: date,
      ...(date > state.end ? { end: date } : {}),
      hoveredDate: date,
      selectedDate: date,
      chartClickMode: 'inspect',
    };
  }
  if (state.chartClickMode === 'set_end') {
    return {
      ...(date < state.start ? { start: date } : {}),
      end: date,
      hoveredDate: date,
      selectedDate: date,
      chartClickMode: 'inspect',
    };
  }
  return { hoveredDate: date, selectedDate: date };
}
