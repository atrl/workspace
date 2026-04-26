---
name: dashboard
description: Use when generating, maintaining, debugging, or visually verifying the SOXL interactive backtest dashboard
---

# Dashboard Skill

## Purpose
Generate and maintain the interactive backtest dashboard (`dashboard.html`) for the SOXL harness. The dashboard is a standalone HTML file that uses Plotly.js for interactive charts and requires no server.

## Workflow

### Regenerate Dashboard After Backtest
```bash
cd /Users/rziiiii/.openclaw/workspace
qlib_venv/bin/python scripts/build_backtest_dashboard.py
# Open dashboard.html in browser
```

### Post-Train Default View
After a training run, the harness should:
1. export `runtime/execution/card/<profile>/<profile>_soxl_daily.csv`
2. rebuild `dashboard.html`
3. default the dashboard to the latest generated version, not the historical best-Sharpe version

If the dashboard opens on an older run after fresh training, check:
- whether the latest `*_soxl_daily.csv` was written
- whether `scripts/build_backtest_dashboard.py` was rerun
- whether browser localStorage is pinning a manual old version

### Add a New Subplot
1. In `scripts/build_backtest_dashboard.py`, edit `_build_layout_all()`:
   - Increase `rows` in `make_subplots()`
   - Add a new `subplot_titles` entry
   - Update `row_heights` to allocate space
   - Add axis styling in the loop
2. In the JS `renderChart()` function, add a new trace block:
   - Extract data arrays from `rows`
   - Push a Plotly trace with correct `xaxis`/`yaxis` (e.g. `x5`, `y5`)
3. Regenerate and test.

### Add a Feature Column to Hover Tooltip
1. Add the column name to `FEATURE_COLS` in `build_backtest_dashboard.py`.
2. In the JS `updateInfoPanel()` function, add an info-row element:
   ```html
   <div class="info-row"><span class="info-k">NewFeature</span><span class="info-v" id="infoNewFeature">-</span></div>
   ```
3. In JS, populate it in `updateInfoPanel()`:
   ```js
   document.getElementById('infoNewFeature').textContent = f.new_feature != null ? fmtNum(f.new_feature, 2) : 'n/a';
   ```
4. Regenerate.

### Add a New ETF to Generalization
1. Ensure `runtime/generalization/detailed/{etf}_detailed_trades.csv` exists.
2. Ensure the ETF is listed in `runtime/generalization/detailed/summary.json`.
3. Regenerate dashboard — the ETF will appear in the select dropdown automatically.

## Architecture

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| Data | Python (pandas, csv, json, pickle) | Load backtest results, merge features, scan model versions |
| Layout | Plotly Python (`make_subplots`) | Generate 4-subplot layout JSON (NAV+Trades / Position / Excess Return / Drawdown) |
| Rendering | Plotly.js (CDN) | Browser-side interactive rendering, zoom, pan, hover, crosshair spikes |
| UI | Vanilla HTML5/CSS3/JS | Controls, metric cards, tables, info panel, state persistence |

## Layout

The dashboard uses a **single 4-subplot layout** (All Panels). No view selector.

```
┌────────────────────────────┬─────────────┐
│  NAV + Trades              │  Info Panel │
│  (Strategy + Buy&Hold +    │  (hover     │
│   BUY/SELL markers)        │   features, │
│                            │   position  │
│  Position (bars)           │   dist,     │
│                            │   trades)   │
│  Excess Return %           │             │
│                            │             │
│  Drawdown                  │             │
├────────────────────────────┴─────────────┤
│  Trade Quality (Kelly / Expectancy / PF) │
├──────────────────────────────────────────┤
│  Trade Performance Table (sortable)      │
├──────────────────────────────────────────┤
│  Model Version Comparison (all versions) │
└──────────────────────────────────────────┘
```

## Key Files
- `scripts/build_backtest_dashboard.py` — dashboard generator
- `dashboard.html` — generated output
- `data/features_v2.pkl` — feature data merged into hover tooltips
- `runtime/execution/generalization/detailed/` — per-ETF CSV backtest data
- `runtime/model/card/*/*_summary.json` — model version history for comparison table

## Design System Colours
| Token | Hex | Usage |
|-------|-----|-------|
| Strategy NAV | `#9a3412` | Primary line |
| Buy&Hold NAV | `#0f766e` | Secondary line |
| BUY marker | `#0f766e` | Triangle-up |
| SELL marker | `#b91c1c` | Triangle-down |
| Background | `#f4efe6` | Page gradient |
| Panel | `#fffaf2` | Card backgrounds |
| Grid line | `#d8ccbb` | Chart grid |
| Stress regime | `rgba(185,28,28,0.07)` | VIX z-score > 2 |
| Calm regime | `rgba(15,118,110,0.07)` | VIX z-score < -1 |

## Features

### Mouse / Touch
- **滚轮缩放**: `scrollZoom: true` — 在图表区域滚动鼠标滚轮缩放
- **拖拽平移**: 按住拖拽平移（Plotly `dragmode: 'zoom'`）
- **框选放大**: 框选区域自动 zoom in

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `+` / `=` | Zoom in (halve visible window) |
| `-` / `_` | Zoom out (double visible window) |
| `R` / `r` | Reset zoom |
| `←` | Pan left |
| `→` | Pan right |

### Crosshair + Info Panel
- **十字准线**: `showspikes=True, spikemode='across'` 跨全图显示
- **右侧固定信息面板**: 鼠标 hover 时显示当日价格、仓位、所有特征值
- **仓位分布**: 4 档水平条形图（0 / 低 0-0.3 / 中 0.3-0.7 / 高 0.7-1）
- **最近交易**: 显示最近 5 笔 BUY→SELL 配对

### Regime Shading
- VIX z-score > 2: 淡红色背景（高压力区）
- VIX z-score < -1: 淡青色背景（低波动区）
- 覆盖全部 4 个子图

### Trade Performance Table
- 点击表头排序
- 点击交易行 → zoom 到该交易日期 ±5 天
- 显示: 入场/出场日期、持仓天数、收益率、PnL、累计、Avg VIX、Avg RSI

### Trade Quality Stats (Kelly Formula)
- **凯利比例** (Kelly Ratio): `win_rate - (1-win_rate)/R`
- **期望收益** (Expectancy): `win_rate*avg_win + (1-win_rate)*avg_loss`
- **收益因子** (Profit Factor): `gross_profit / abs(gross_loss)`

### Model Version Comparison
- 自动扫描 `runtime/model/card/*/*_summary.json`
- 展示所有历史版本的 `test_proxy` 和 `valid` 指标
- 绿色高亮每列最优值
- ★ 标记已晋升 (promoted) 的版本

## Troubleshooting

### Dashboard shows blank chart area
- Check browser console for Plotly.js load errors (CDN blocked?)
- Verify `PLOTLY_LAYOUTS` is present in generated HTML

### Features missing in hover
- Check `data/features_v2.pkl` exists and covers the date range
- Verify `_features` key is present in DATA rows (search generated HTML)

### File too large
- The generated HTML embeds all data inline. For >5 ETFs or very long histories, consider splitting into per-ETF pages or lazy-loading data.

### Model versions not showing
- Verify `runtime/model/card/*/*_summary.json` exists and contains `test_proxy` / `valid` fields
- Check browser console for JS errors in `renderVersionTable()`
