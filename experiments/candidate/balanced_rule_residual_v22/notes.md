---
title: balanced_rule_residual_v22
direction: model
status: candidate
---

# balanced_rule_residual_v22

## Intent

Feature-only clone of v19 with external VIX term-structure, SKEW, and VVIX.

## New Features

- `vvix_zscore_252`
- `vix_term_slope` = VIX / VIX3M - 1
- `vix_short_ratio` = VIX9D / VIX - 1
- `skew_zscore_252`
- `vix_term_slope_velocity_5`

Gated by `SOXL_USE_VIX_TERM=1` for replay compatibility.

## Training Command

```bash
SOXL_USE_VIX_TERM=1 PYTHONPATH=. qlib_venv/bin/python -m src.soxl_quant.cli \
  --profiles balanced_rule_residual_v22 --smoke --skip-data-refresh --seed 12345
```

## Evidence Links

- Primary result: `runtime/model/2026-04-21-v22-vix-term-structure-results.md`
- Pairwise review: `runtime/model/reviews/balanced_rule_residual_v22_vs_v19.md`
- Landscape: `docs/experiment-landscape-review.md`
