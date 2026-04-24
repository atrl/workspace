---
date: '2026-04-21'
title: 2026-04-21 v22 VIX term-structure training results
kind: review
profiles:
- balanced_rule_residual_v22
- balanced_rule_residual_v22_vs_v19
- smoke
status: reference
tags:
- balanced_rule_residual_v22
- balanced_rule_residual_v22_vs_v19
- checkpoint-selection
- eval-2022
- reference
- review
- smoke
- stress-regime
- walk-forward
artifacts:
- artifacts/models/soxl_balanced_rule_residual_v19_summary.json
- evidence/post_train_review/balanced_rule_residual_v22_vs_v19.md
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: "First training run of balanced_rule_residual_v22 with SOXL_USE_VIX_TERM=1\
  \ (external VVIX / VIX9D / VIX3M / SKEW features added to the observation). Same\
  \ seed (12345) and profile otherwise identical to v19 baseline. Feature count 59\
  \ \u2192 64 and schema signature flipped as expected. All three fixed-split promotion\
  \ gates still pass. New features buy stress-window MaxDD improvement (test_2020\
  \ \u22121.9pts, test_2023_2024 \u22124.9pts) at the cost of a 5.8pt return drag\
  \ in test_2023_2024. Keep as candidate, do not flip defaults."
---

# 2026-04-21 v22 VIX term-structure training results

## Context

Continuation of the 2026-04-20 Tier 1-B scaffolding task. That task wired
external VVIX / VIX9D / VIX3M / SKEW series into `engineer_feature_frame` behind
`SOXL_USE_VIX_TERM=1` and registered `balanced_rule_residual_v22` as a
feature-only clone of v19. This note captures the first real training run of
that profile.

## Run setup

```bash
SOXL_USE_VIX_TERM=1 PYTHONPATH=. qlib_venv/bin/python -m src.soxl_quant.cli \
    --profiles balanced_rule_residual_v22 --smoke --skip-data-refresh --seed 12345
```

- seed = 12345 (matches v19 baseline for apples-to-apples)
- `--smoke` in this harness does not shorten training — it skips the warm-start
  branch and forces `stage_iters = profile.max_iters` (= full 80 per stage).
  Episodes stayed at 4, iters at 80, epi/iter at 10.
- Wall clock: ~20 minutes.

## Feature-schema evidence

- `feature_count`: 59 → 64 ✔
- `signature`: `99a98f3b024c` → `9ec7f9c42e73` ✔
- Five new columns computed in `engineer_feature_frame`:
  `vvix_zscore_252`, `vix_term_slope`, `vix_short_ratio`,
  `skew_zscore_252`, `vix_term_slope_velocity_5`.

## Slice results

| slice | metric | v19 | v22 | Δ |
| --- | --- | --- | --- | --- |
| valid | return | 56.59% | 62.93% | +6.34 |
| valid | sharpe | 1.56 | 1.63 | +0.08 |
| valid | max_dd | 27.01% | 30.56% | +3.55 |
| test_2020 | return | 109.41% | 115.60% | +6.19 |
| test_2020 | sharpe | 2.64 | 2.74 | +0.11 |
| test_2020 | max_dd | 18.14% | 16.22% | **−1.92** |
| test_2022 | return | −22.45% | −24.52% | −2.07 |
| test_2022 | sharpe | −0.84 | −0.91 | −0.07 |
| test_2022 | max_dd | 31.07% | 32.71% | +1.64 |
| test_2022 | avg_pos | 0.261 | 0.269 | +0.01 |
| test_2023_2024 | return | 43.60% | 37.85% | −5.75 |
| test_2023_2024 | max_dd | 37.34% | 32.43% | **−4.92** |
| test_2023_2024 | avg_pos | 0.475 | 0.447 | −0.03 |
| test_proxy | return | 735.10% | 695.45% | −39.65 |
| test_proxy | sharpe | 0.95 | 0.94 | −0.01 |
| test_proxy | max_dd | 54.83% | 54.52% | −0.31 |

Promotion gates (fixed-split): all three PASS —
`test_2022_max_dd` 32.7 < 35, `test_2022_avg_pos` 0.27 < 0.35,
`test_proxy_sharpe` 0.94 > 0.75. `eligible: False` is solely because
`walk_forward_evaluated: False`, not a gate failure.

## Interpretation

- New external VIX features **buy stress-window drawdown** without giving up the
  trend-capture quality in `test_2020`. That is the clearest signal.
- The trade-off is a modest return drag in the range regime (`test_2023_2024`
  return −5.8 pts, avg_pos −0.03) — the model has slightly backed off exposure
  when VIX term structure is steep, and in 2023-2024 that's net negative on
  return but net positive on DD.
- `test_2022` moves are inside seed-noise range (return −2pts, MaxDD +1.6pts).
  Not a regime change; the gate is still clear.
- v22 is NOT a drop-in replacement for v19 — it's a candidate. Flipping the
  default would trade return for DD across the proxy window without clear
  Sharpe improvement (test_proxy Sharpe essentially unchanged at 0.94 vs 0.95).

## Verdict

- Registry status: `candidate` (unchanged).
- Keep actor + summary.
- Do NOT change default feature schema. `SOXL_USE_VIX_TERM` stays opt-in.
- Legacy v16-v21 actors remain replay-compatible because `SOXL_USE_VIX_TERM` is
  off by default.

## Next actions

1. Run 2-3 seed confirm (12345 + 13579 + 24680) to check whether the
   `test_2023_2024` return dip is structural or seed noise.
2. Run walk-forward evaluation (`--walk-forward`) on v22 to clear the last
   promotion check.
3. If seeds confirm the DD wins and the return drop is within noise, optionally
   enable `use_pareto: true` on a v22-derived profile — the Pareto path is
   already scaffolded in `selection.py`.
4. Do not flip defaults or update canonical rows in the registry until
   walk-forward and seed confirm are both done.

## Related

- Scaffolding: `memory/2026-04-20-tier1-2-3-harness-scaffolding.md`
- Baseline comparator: `artifacts/models/soxl_balanced_rule_residual_v19_summary.json`
- Post-train review: `evidence/post_train_review/balanced_rule_residual_v22_vs_v19.md`
