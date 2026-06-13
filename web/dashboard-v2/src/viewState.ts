export interface ViewUpdate {
  selectedProfiles?: string[];
  baselineProfile?: string;
  start?: string;
  end?: string;
  hoveredDate?: string;
}

export function viewUpdateRequiresChartRender(update: ViewUpdate): boolean {
  return (
    update.selectedProfiles !== undefined ||
    update.baselineProfile !== undefined ||
    update.start !== undefined ||
    update.end !== undefined
  );
}
