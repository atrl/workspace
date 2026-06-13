# SOXL Dashboard V2 Design

## Context

The current dashboard is generated as a large standalone Plotly HTML file from
`scripts/build_backtest_dashboard.py`. It is useful as a static artifact, but it
does not provide a trading-review workflow:

- The main chart does not behave like a TradingView-style charting surface.
- Model differences are hard to inspect because each version is selected in
  isolation.
- BUY/SELL decisions, position changes, and feature state are not rendered as
  first-class evidence on the price timeline.
- The generated HTML mixes data shaping, chart rendering, model selection,
  production status, and UI state into one file, making further interaction
  changes fragile.

The user explicitly removed remote static hosting as a constraint. Dashboard v2
should optimize for local and internal research-review use first. Remote
deployment compatibility is out of scope for this design.

## Goals

Dashboard v2 should be a local TradingView-like SOXL research terminal that can
answer:

1. What did SOXL do on the price chart?
2. Where did a selected model buy, sell, hold, or change exposure?
3. How did two to four selected models differ over the same dates?
4. Which market and model features were present when those actions happened?
5. Which replay file, actor/profile, and safe backtest window produced the
   displayed data?

The first implementation should prove the core review loop rather than recreate
every feature in the old dashboard.

## Non-Goals

- Do not optimize for Vercel or any remote static deployment path.
- Do not delete or replace the existing `dashboard.html` in the first pass.
- Do not add broker execution, order placement, or live account operations.
- Do not create another model training or selection experiment.
- Do not revive stale model summary tables unless they are backed by replayable
  daily path evidence.

## Recommended Approach

Build dashboard v2 as a small Vite + TypeScript app backed by generated JSON
artifacts under `dashboard_data/v2/`.

Use TradingView Lightweight Charts as the primary market chart. It is
TradingView-maintained, Apache-2.0 licensed, available through npm, and supports
series markers for event annotations. It is the closest fit for the requested
TradingView-like K-line experience without adopting a closed TradingView
product.

Use custom TypeScript panels for model comparison and feature inspection. Apache
ECharts can be introduced later for dense comparison heatmaps, but the first
version should keep dependencies limited to Lightweight Charts and ordinary
DOM/SVG panels.

KLineCharts remains a fallback candidate if the project later needs built-in
drawing tools and a fuller technical-analysis terminal. Highcharts Stock is not
recommended because its license path adds avoidable cost and ambiguity.

## Architecture

### Data Build Layer

Create a repo-owned data builder:

- `scripts/build_dashboard_v2_data.py`

Responsibilities:

- Read `dashboard_data/model_replay_freshness.json` as the replayable model
  source of truth.
- Read model replay CSVs such as
  `runtime/model/card/<profile>/<profile>_soxl_daily.csv`.
- Read market OHLCV and derived indicators through the same market enrichment
  path currently used by `scripts/build_backtest_dashboard.py`.
- Emit normalized JSON files under `dashboard_data/v2/`.
- Fail closed when a selected replay CSV is missing, malformed, or outside the
  safe replay window.

Do not make the browser infer model lineage or replay freshness from raw file
paths. The data builder should expose that metadata explicitly.

### Data Contract

The first version should write these files:

- `dashboard_data/v2/manifest.json`
- `dashboard_data/v2/market/SOXL.json`
- `dashboard_data/v2/models/<profile>.json`
- `dashboard_data/v2/comparison/default.json`

`manifest.json` should include:

- generated timestamp
- latest market date
- safe backtest end
- available profiles
- default baseline profile
- default comparison profiles
- source paths for market and replay artifacts

`market/SOXL.json` should include one row per trading day:

- `date`
- `open`, `high`, `low`, `close`, `volume`
- indicator fields already available in the harness such as `rsi6`, `ma180`,
  `hv_20`, `vol_coef`, `panic_score`, `market_state`
- feature fields from `data/features_v2.pkl` when available, including `vix`,
  `rsi`, `volatility`, `spread`, `vix_zscore`, `spread_trend`, `ma5_ma20`,
  `vol_strength`

`models/<profile>.json` should include:

- profile name
- status and actor/profile paths when available
- replay start/end and row count
- daily rows with `date`, `strategy_nav`, `buyhold_nav`, `position`,
  `action`, `price`, `excess`, `excess_pct`
- derived events for `BUY`, `SELL`, and material position changes
- metrics for full window and standard slices such as YTD, June, 2020, 2022,
  and 2023-2024 when data exists

`comparison/default.json` should include:

- baseline profile
- compared profiles
- common date window
- per-date model deltas:
  - NAV delta versus baseline
  - position delta versus baseline
  - action mismatch
  - drawdown delta
- summary rankings by return, max drawdown, excess return, and position usage

### Frontend Layer

Create a new app under:

- `web/dashboard-v2/`

The app should compile to a local static bundle, but local development is the
primary workflow.

Core components:

- `App`: coordinates selected profiles, selected date window, and hover date.
- `ChartWorkspace`: owns the TradingView Lightweight Charts instances.
- `MarketChart`: renders SOXL candlesticks, volume, MA line, feature regime
  shading, and BUY/SELL markers.
- `PositionChart`: renders selected model position as a step line and highlights
  material position changes.
- `NavComparisonChart`: renders selected model NAV paths and buy-and-hold.
- `ModelDiffPanel`: shows model delta metrics for the visible window.
- `FeatureInspector`: shows the hovered date's market features, model actions,
  positions, and action mismatches.
- `DataSourcePanel`: shows replay freshness, source paths, generated timestamp,
  and warnings.

Keep the UI dense and operational. This is a review tool, not a landing page.

### Interaction Model

Required interactions:

- Select two to four models for comparison.
- Choose a baseline model.
- Change date range with presets and custom inputs.
- Hover a date and see:
  - OHLCV
  - selected model position and action
  - action mismatch versus baseline
  - feature values
  - source row availability
- Click a BUY/SELL marker to pin the date in the inspector.
- Toggle overlays:
  - BUY/SELL markers
  - position changes
  - feature regime shading
  - model NAV lines
  - action mismatch markers

The visible chart window should use the common date range where selected models
and SOXL market data overlap. If one selected model does not cover the requested
window, the UI should show a clear warning and clip to the common range rather
than silently drawing misleading comparisons.

## Visual Design

The first viewport should show the actual review surface:

- A compact top toolbar for profile selection, baseline, and date range.
- A main price chart occupying most horizontal space.
- A right-side inspector for the hovered or pinned date.
- A lower comparison area for NAV, position, and action-difference panels.

Use restrained colors:

- green for BUY/increase exposure
- red for SELL/decrease exposure
- blue or cyan for selected strategy lines
- gray for baseline and unavailable data
- amber for warnings

Avoid decorative hero sections, marketing layout, nested cards, and broad
single-hue palettes. The tool should feel like a research terminal.

## Error Handling

The data builder should fail with explicit messages for:

- missing `dashboard_data/model_replay_freshness.json`
- missing replay CSV
- missing market OHLCV data
- no overlapping date range between market and model rows
- duplicate or invalid dates
- non-numeric `position`, `strategy_nav`, or price fields

The frontend should show inline warnings for:

- selected models with no common overlap
- model rows outside `safe_backtest_end`
- missing feature values
- action values outside `BUY`, `SELL`, `HOLD`
- source data generated before the latest market date

Warnings should be visible in `DataSourcePanel`; they should not block rendering
when enough data exists to draw an honest partial view.

## Testing And Verification

The implementation plan should require test-first work for each behavior.

Data builder tests:

- Manifest uses `dashboard_data/model_replay_freshness.json` as the replay
  source.
- Market JSON preserves OHLCV and feature fields.
- Model JSON preserves `position`, `strategy_nav`, `buyhold_nav`, `action`, and
  source path.
- Derived position-change and action events are deterministic.
- Comparison JSON clips to the common date window and computes action mismatch
  rows.

Frontend tests:

- Profile selection limits comparison to two to four models.
- Hover date updates the feature inspector.
- Missing overlap renders a warning instead of a blank chart.
- BUY/SELL markers are derived from model event rows, not inferred from price
  movement.
- Baseline changes recompute model deltas.

Visual verification:

- Run the local dev server.
- Open the dashboard in a browser.
- Capture desktop and mobile screenshots.
- Confirm nonblank canvas pixels.
- Confirm visible K-line candles, BUY/SELL markers, position line, NAV
  comparison, and feature inspector content.

Repository verification:

- Do not stage unrelated data, qlib, runtime index, or model report files.
- Keep the old dashboard tests passing unless an implementation step explicitly
  updates the old dashboard contract.
- Add a focused static data contract checker for `dashboard_data/v2/`.

## Rollout

1. Land this design.
2. Write a separate implementation plan with task-level TDD steps.
3. Implement the data builder and v2 frontend behind a new local entrypoint.
4. Keep `dashboard.html` and `scripts/build_backtest_dashboard.py` intact until
   v2 has browser evidence that satisfies the requirements.
5. After v2 is verified, decide whether the old dashboard remains as legacy
   static evidence or becomes a compatibility wrapper that links to v2.

## Open Questions

The first implementation can proceed with conservative defaults:

- Default baseline: `balanced_rule_residual_v22` when available, otherwise the
  first canonical replay profile in the manifest.
- Default comparison set: `smoke`, `balanced_rule_residual_v22`, and current
  production composer if the data builder can resolve it from existing
  artifacts.
- Initial date range: latest six months within the common safe replay window.

These defaults should be visible in the UI and easy to change. They should not
be hard-coded as trading conclusions.
