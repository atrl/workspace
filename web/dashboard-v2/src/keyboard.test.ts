import { describe, expect, test } from 'vitest';
import { chartKeyboardMode, chartKeyboardRangeCommand, isChartKeyboardEditableTarget } from './keyboard';

function keyEvent(key: string): KeyboardEvent {
  return new KeyboardEvent('keydown', { key });
}

describe('dashboard v2 chart keyboard shortcuts', () => {
  test('maps TradingView-style range shortcuts to chart commands', () => {
    expect(chartKeyboardRangeCommand(keyEvent('+'))).toBe('zoom_in');
    expect(chartKeyboardRangeCommand(keyEvent('='))).toBe('zoom_in');
    expect(chartKeyboardRangeCommand(keyEvent('-'))).toBe('zoom_out');
    expect(chartKeyboardRangeCommand(keyEvent('ArrowLeft'))).toBe('pan_left');
    expect(chartKeyboardRangeCommand(keyEvent('h'))).toBe('pan_left');
    expect(chartKeyboardRangeCommand(keyEvent('ArrowRight'))).toBe('pan_right');
    expect(chartKeyboardRangeCommand(keyEvent('l'))).toBe('pan_right');
    expect(chartKeyboardRangeCommand(keyEvent('Home'))).toBe('reset');
    expect(chartKeyboardRangeCommand(keyEvent('r'))).toBe('reset');
    expect(chartKeyboardRangeCommand(keyEvent('End'))).toBe('latest');
    expect(chartKeyboardRangeCommand(keyEvent('x'))).toBeNull();
  });

  test('maps chart mode shortcuts without stealing ordinary editing keys', () => {
    expect(chartKeyboardMode(keyEvent('Escape'))).toBe('inspect');
    expect(chartKeyboardMode(keyEvent('s'))).toBe('set_start');
    expect(chartKeyboardMode(keyEvent('e'))).toBe('set_end');
    expect(chartKeyboardMode(keyEvent('ArrowLeft'))).toBeNull();

    expect(isChartKeyboardEditableTarget(document.createElement('input'))).toBe(true);
    expect(isChartKeyboardEditableTarget(document.createElement('select'))).toBe(true);
    expect(isChartKeyboardEditableTarget(document.createElement('button'))).toBe(true);

    const editable = document.createElement('div');
    editable.setAttribute('contenteditable', 'true');
    expect(isChartKeyboardEditableTarget(editable)).toBe(true);
    expect(isChartKeyboardEditableTarget(document.createElement('div'))).toBe(false);
  });
});
