export interface ViewUpdate {
  selectedProfiles?: string[];
  baselineProfile?: string;
  chartProfile?: string;
  start?: string;
  end?: string;
  hoveredDate?: string;
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
