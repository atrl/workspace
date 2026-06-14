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

  test('prioritizes feature and operations panels above the hover inspector', () => {
    const css = readFileSync('src/styles.css', 'utf8');
    const rightStackRule = css.match(/\.right-stack\s*\{[^}]*\}/)?.[0] ?? '';
    const inspectorRule = css.match(/\.inspector-panel\s*\{[^}]*\}/)?.[0] ?? '';

    expect(rightStackRule).toContain('grid-template-rows: minmax(300px, 1.15fr) minmax(170px, auto) minmax(120px, 0.55fr);');
    expect(inspectorRule).toContain('max-height: 260px;');
  });

  test('keeps desktop layout constrained while model labels wrap inside toolbar', () => {
    const css = readFileSync('src/styles.css', 'utf8');
    const appRule = css.match(/#app\s*\{[^}]*\}/)?.[0] ?? '';
    const toolbarRule = css.match(/\.toolbar\s*\{[^}]*\}/)?.[0] ?? '';
    const profileGroupRule = css.match(/\.profile-group\s*\{[^}]*\}/)?.[0] ?? '';

    expect(appRule).toContain('max-width: 100vw;');
    expect(toolbarRule).toContain('max-width: 100vw;');
    expect(profileGroupRule).toContain('max-width: 100%;');
    expect(profileGroupRule).toContain('flex-wrap: wrap;');
    expect(profileGroupRule).not.toContain('overflow-x: auto;');
  });
});
