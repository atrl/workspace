---
date: 2026-04-21
title: 2026-04-21 release overlay evaluation
kind: failure
profiles:
- balanced_rule_residual_v19
- soxl
status: dropped
tags:
- balanced_rule_residual_v19
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- failure
- soxl
- stress-regime
artifacts:
- runtime/execution/generalization/execution_overlay_results_v19_release.json
- runtime/execution/generalization/execution_overlay_results_v19_release_subset.json
workpad: memory/index.json
summary: The first explicit `release` execution overlay on top of `balanced_rule_residual_v19`
  did not produce a materially new behavior. `stress_release_route_v1` was effectively
  identical to `regime_route_v1` on the key SOXL bear/recovery slices, so this direction
  should not be treated as a successful next-step mechanism.
---

# 2026-04-21 release overlay evaluation

## Context

The repo already concluded that another generic overlay sweep would be duplicate
work unless it introduced an explicit `release / re-entry` state beyond the
existing `stress / bull / base` routing. That state was implemented and then
evaluated against the fixed actor `balanced_rule_residual_v19`.

## Evidence

- Result files:
  - `runtime/execution/generalization/execution_overlay_results_v19_release.json`
  - `runtime/execution/generalization/execution_overlay_results_v19_release_subset.json`
- SOXL key readout:
  - `base`: proxy `412.7 / 0.837 / 48.79`, `bear_2022 -23.2 / 31.32 / 0.262`,
    `recovery_2023_2024 41.7 / 32.81 / 0.472`
  - `regime_route_v1`: proxy `478.6 / 0.893 / 45.97`,
    `bear_2022 -22.6 / 31.00 / 0.255`,
    `recovery_2023_2024 43.0 / 34.55 / 0.464`
  - `stress_release_route_v1`: proxy `482.8 / 0.896 / 46.29`,
    `bear_2022 -22.6 / 31.00 / 0.255`,
    `recovery_2023_2024 43.0 / 34.55 / 0.464`
- Across `TQQQ`, `UPRO`, and `SPXL`, the release variant was identical or
  effectively identical to `regime_route_v1` on the key slices.

## Decision

- Drop `stress_release_route_v1` as a real next-step candidate.
- Keep release-regime support in code only as harness infrastructure for future
  diagnostics.
- Do not reopen overlay tuning unless:
  - release rows can be shown to be materially distinct in count and timing, or
  - new information is added beyond threshold drift on the existing row-level
    features.

## Next

- Prefer one of these directions before another retrain:
  - inspect release-state firing frequency and timing versus the failed windows
  - review checkpoint-selection around restart windows
  - test a new external-risk / cross-asset state rather than another overlay
    coefficient sweep
