# Dashboard Skill

## Purpose
Generate and maintain the interactive backtest dashboard (`runtime/generalization/dashboard.html`) for the SOXL harness. The dashboard is a standalone HTML file that uses Plotly.js for interactive charts and requires no server.

## Workflow

### Regenerate Dashboard After Backtest
```bash
cd /Users/rziiiii/.openclaw/workspace
qlib_venv/bin/python scripts/build_backtest_dashboard.py
# Open runtime/generalization/dashboard.html in browser
```

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
2. In the JS `buildHoverText()` function, add:
   ```js
   if (f.new_feature != null) featParts.push(`NewFeature: ${fmtNum(f.new_feature, 4)}`);
   ```
3. Regenerate.

### Add a New ETF to Generalization
1. Ensure `runtime/generalization/detailed/{etf}_detailed_trades.csv` exists.
2. Ensure the ETF is listed in `runtime/generalization/detailed/summary.json`.
3. Regenerate dashboard — the ETF will appear in the select dropdown automatically.

## Architecture

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| Data | Python (pandas, csv, json, pickle) | Load backtest results, merge features |
| Layout | Plotly Python (`make_subplots`) | Generate 4-subplot (or N-subplot) layout JSON |
| Rendering | Plotly.js (CDN) | Browser-side interactive rendering, zoom, pan, hover |
| UI | Vanilla HTML5/CSS3/JS | Controls, metric cards, tables, state persistence |

## Key Files
- `scripts/build_backtest_dashboard.py` — dashboard generator
- `runtime/generalization/dashboard.html` — generated output
- `data/features_v2.pkl` — feature data merged into hover tooltips
- `runtime/generalization/detailed/` — per-ETF CSV backtest data

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

## Troubleshooting

### Dashboard shows blank chart area
- Check browser console for Plotly.js load errors (CDN blocked?)
- Verify `PLOTLY_LAYOUTS` is present in generated HTML

### Features missing in hover
- Check `data/features_v2.pkl` exists and covers the date range
- Verify `_features` key is present in DATA rows (search generated HTML)

### File too large
- The generated HTML embeds all data inline. For >5 ETFs or very long histories, consider splitting into per-ETF pages or lazy-loading data.
