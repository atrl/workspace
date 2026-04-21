# balanced_rule_residual_v22 notes

## Intent

Feature-only clone of v19 with external VIX term-structure, SKEW, and VVIX:
- `vvix_zscore_252`
- `vix_term_slope` = VIX / VIX3M - 1
- `vix_short_ratio` = VIX9D / VIX - 1
- `skew_zscore_252`
- `vix_term_slope_velocity_5`

Wired through `src/soxl_quant/data.py::engineer_feature_frame`, gated by
`SOXL_USE_VIX_TERM=1` so legacy 59-feature actors stay replay-compatible by
default.

## Training command

```bash
SOXL_USE_VIX_TERM=1 qlib_venv/bin/python -m soxl_quant.cli train \
    --profile balanced_rule_residual_v22 \
    --seed 12345
```

## Evidence Links

- Primary result note:
  - `memory/2026-04-21-v22-vix-term-structure-results.md`
- Pairwise review:
  - `runtime/post_train_review/balanced_rule_residual_v22_vs_v19.md`
- Frontier summary:
  - `docs/experiment-landscape-review.md`

## Current Role

- Active feature-only candidate derived from `v19`
- Tests whether external volatility-term-structure data improves drawdown shape
  without breaking the fixed-split gate
- Full result narrative now lives in memory instead of being duplicated here

## Training command (reproducible)

```bash
SOXL_USE_VIX_TERM=1 PYTHONPATH=. qlib_venv/bin/python -m src.soxl_quant.cli \
    --profiles balanced_rule_residual_v22 --smoke --skip-data-refresh --seed 12345
```

Note on `--smoke`: in this harness `smoke=True` forces `stage_iters =
profile.max_iters` (see `app/selection.py`) and skips the warm-start branch, so
the v22 run executed the full 80-iter × 4-stage staged training of the profile
(episodes=4, iters=80, epi/iter=10, ~20 min wall clock on this box). It did not
run behavioral-cloning warm-start because `smoke=True` gates it off.
