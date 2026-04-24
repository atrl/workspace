---
date: '2026-04-21'
title: 2026-04-21 existing overlay gap review
kind: decision
profiles:
- soxl
status: active
tags:
- active
- decision
- eval-2022
- execution-overlay
- soxl
- stress-regime
artifacts: []
workpad: memory/index.json
summary: Reviewed the existing fixed-actor execution overlay sweep before starting
  any new overlay work. The repo already validated the generic overlay direction with
  `regime_route_v1`, but on SOXL it did not materially improve the `2022` bear window
  and also gave up recovery return. Any next overlay loop that only retunes stress/bull
  thresholds would be duplicate work; the next valid overlay mechanism must introduce
  an explicit release/re-entry state.
---

# 2026-04-21 existing overlay gap review

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Question

Has the repo already implemented the overlay idea that was being proposed as
the next step?

## Answer

Yes, in the generic sense.

The repository already has a fixed-actor execution overlay study with:

- `base`
- `cta_trend_hold`
- `cta_defensive`
- `fast_reentry`
- `regime_route_v1`

This means another loop that only tweaks threshold multipliers, cooldowns, or
trade-lock settings without introducing a new state machine would be duplicate
work.

## What the current study already proved

- `regime_route_v1` validated regime routing as a direction.
- On average across the 5 ETF sweep, it had the best `bear_2022` outcome.
- But it did **not** dominate overall proxy Sharpe / return.

## Why it still does not solve the SOXL problem

For SOXL specifically:

- `base`: proxy `329.1% / 0.718 / 62.84`, `bear_2022 -37.2 / 45.12`,
  `recovery_2023_2024 60.1 / 36.08`
- `regime_route_v1`: proxy `268.5% / 0.673 / 62.03`, `bear_2022 -37.8 / 45.04`,
  `recovery_2023_2024 48.5 / 37.40`

So the current regime-routing overlay:

- did not materially fix the SOXL bear window
- also gave up recovery return

That is not the desired selective re-entry behavior.

## Implication for the next step

The next overlay experiment is only justified if it introduces a mechanism that
does not already exist in `regime_route_v1`.

The cleanest candidate is:

- an explicit `release / re-entry` state
- separate from `stress`, `bull`, and `base`
- used to allow faster re-entry after stress relief instead of only broad
  stress-time de-risking

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
