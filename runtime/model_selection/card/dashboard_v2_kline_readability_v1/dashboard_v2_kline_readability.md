# Dashboard V2 K-Line Readability

Date: 2026-06-16

## Issue

The Dashboard V2 first viewport mixed three different concepts in the same
visual area:

- SOXL market K-line through `2026-06-15`.
- Qlib actor replay rows clipped to safe replay end `2026-06-12`.
- SOXLBacktest reference baseline rows through `2026-06-15`.

The main chart also overlaid model position lines on the K-line price pane, so
the candles were harder to read. The UI duplicated the latest baseline readout,
which made the stale/fresh split harder to scan.

## Fix

- Main K-line now shows price candles, SOXL close, MA180, volume, BUY/SELL
  markers, and model-difference markers only.
- Model positions remain in the dedicated `Model Position` pane.
- Model differences are shown in the compact strip with explicit common-window
  text:
  `Common 2026-01-02 to 2026-06-12 | Price view to 2026-06-15`.
- Baseline freshness is displayed once in the status strip:
  `Baseline latest 2026-06-15 +146.04% pos 0.159`.
- The duplicate standalone baseline latest readout was removed.

## Evidence

- `dashboard_v2_v2_desktop_final.png`

Pixel check for the final screenshot:

```text
dashboard_v2_v2_desktop_final.png size=3800x1530 non_dark=0.1166
```

Data sanity:

```text
dashboard_v2_check=ok profiles=6 market_rows=4091 comparison_rows=112
baseline latest=2026-06-15
actor replay end=2026-06-12
latest actor signals=2026-06-15
```

## Decision

This is a dashboard readability fix only. It does not promote any actor model.
The strict walk-forward gate remains blocked with zero final-retained actors.
