import type { ChartRangeCommand } from './chart';
import type { ChartClickMode } from './viewState';

export function chartKeyboardRangeCommand(event: KeyboardEvent): ChartRangeCommand | null {
  const key = event.key;
  const lower = key.toLowerCase();
  if (key === '+' || key === '=') return 'zoom_in';
  if (key === '-' || key === '_') return 'zoom_out';
  if (key === 'ArrowLeft' || lower === 'h') return 'pan_left';
  if (key === 'ArrowRight' || lower === 'l') return 'pan_right';
  if (key === 'Home' || lower === 'r') return 'reset';
  if (key === 'End') return 'latest';
  return null;
}

export function chartKeyboardMode(event: KeyboardEvent): ChartClickMode | null {
  const lower = event.key.toLowerCase();
  if (event.key === 'Escape') return 'inspect';
  if (lower === 's') return 'set_start';
  if (lower === 'e') return 'set_end';
  return null;
}

export function isChartKeyboardEditableTarget(target: EventTarget | null): boolean {
  const element = target instanceof HTMLElement ? target : null;
  if (!element) return false;
  if (element.isContentEditable || element.getAttribute('contenteditable') === 'true') return true;
  return ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(element.tagName);
}
