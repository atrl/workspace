import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';

describe('dashboard v2 layout styles', () => {
  test('bounds the desktop workspace height to the viewport so charts cannot stretch offscreen', () => {
    const css = readFileSync('src/styles.css', 'utf8');
    const workspaceRule = css.match(/\.workspace\s*\{[^}]*\}/)?.[0] ?? '';

    expect(workspaceRule).toContain('height: clamp(');
    expect(workspaceRule).toContain('calc(100vh');
    expect(workspaceRule).not.toContain('height: 100%;');
  });
});
