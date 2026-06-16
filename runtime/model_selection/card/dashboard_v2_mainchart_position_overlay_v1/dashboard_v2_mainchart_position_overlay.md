# Dashboard V2 Main Chart Position Overlay

## Purpose

Make model differences visible directly on the main K-line chart. The main chart
now keeps SOXL candles dominant and adds compact historical position overlays in
the lower lane for selected models. Latest signal rows stay as markers/readouts
and are not appended to the historical main-chart position overlay.

## Evidence

- Desktop screenshot:
  `runtime/model_selection/card/dashboard_v2_mainchart_position_overlay_v1/dashboard_v2_position_overlay_1440.png`
- Mobile screenshot:
  `runtime/model_selection/card/dashboard_v2_mainchart_position_overlay_v1/dashboard_v2_position_overlay_mobile.png`
- DOM checks confirmed:
  - no visible `load-error` node
  - `POSITION` legend is visible
  - baseline latest reads `2026-06-15 +146.04% pos 0.159`
  - latest model signals and replay split are visible
  - market chart canvas is present
- Pixel checks:
  - desktop `1440x1000`, non-dark `18.07%`, cyan-like pixels `8446`
  - mobile `390x900`, non-dark `22.30%`, cyan-like pixels `4074`

## Verification

```bash
npm --prefix web/dashboard-v2 test -- --run
npm --prefix web/dashboard-v2 run build
```

Both commands passed after the runtime price-scale fix.
