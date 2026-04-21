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

## Training result (2026-04-21, seed=12345, SOXL_USE_VIX_TERM=1)

Feature schema: 59 → 64, signature 99a98f3b → 9ec7f9c4 (distinct column set ✔).
Actor: `artifacts/models/soxl_balanced_rule_residual_v22_actor.pt`.

| slice | metric | v19 | v22 | Δ |
| --- | --- | --- | --- | --- |
| valid | return | 56.6% | 62.9% | +6.3 |
| valid | sharpe | 1.56 | 1.63 | +0.08 |
| valid | max_dd | 27.0% | 30.6% | +3.5 |
| test_2020 | return | 109.4% | 115.6% | +6.2 |
| test_2020 | sharpe | 2.64 | 2.74 | +0.11 |
| test_2020 | max_dd | 18.1% | 16.2% | **−1.9** |
| test_2022 | return | −22.5% | −24.5% | −2.1 |
| test_2022 | sharpe | −0.84 | −0.91 | −0.07 |
| test_2022 | max_dd | 31.1% | 32.7% | +1.6 |
| test_2022 | avg_pos | 0.261 | 0.269 | +0.01 |
| test_2023_2024 | return | 43.6% | 37.9% | −5.8 |
| test_2023_2024 | max_dd | 37.3% | 32.4% | **−4.9** |
| test_proxy | return | 735% | 695% | −39 |
| test_proxy | sharpe | 0.95 | 0.94 | −0.01 |
| test_proxy | max_dd | 54.8% | 54.5% | ~flat |

Promotion gates (fixed-split): all three PASS — `test_2022_max_dd` 32.7 < 35,
`test_2022_avg_pos` 0.27 < 0.35, `test_proxy_sharpe` 0.94 > 0.75. Walk-forward
has not been evaluated yet, so `eligible: False` is due to `walk_forward_missing`,
not a gate failure.

## Expectation scoreboard

- Must-haves: training completes ✔, schema signature differs ✔, `test_2022_max_dd`
  stays within gate (32.7 < 35) ✔, `test_2022_avg_pos` stable (0.27, same tier as
  v19) ✔.
- Nice-to-haves: `test_2020` return/sharpe/DD all improved ✔, `test_2023_2024`
  max_dd improved ✔ but return slipped by 5.8 pts, `test_proxy` essentially flat.

Verdict: **keep as candidate**. Feature wiring is healthy and the new features
clearly improve stress-window DD in `test_2020` and `test_2023_2024`; the cost
is a modest return drag in the range window. Not a drop-in replacement for v19
yet — before promoting we still need (a) a seed sweep to see whether the
`test_2023_2024` return dip survives variance, and (b) a walk-forward run to
clear the last promotion check.

## Next actions

- Run a 2-3 seed confirm (12345 / 13579 / 24680) keeping all other knobs fixed,
  to decide whether the return drop in `test_2023_2024` is structural or seed
  noise.
- If seeds confirm: try Pareto checkpoint selection (`use_pareto: true` already
  scaffolded) — the mild DD wins suggest the new features add information that a
  Sharpe-only selection may be mis-prioritising.
- Run walk-forward eval to clear promotion gate.
- Leave `SOXL_USE_VIX_TERM=1` as an opt-in; do not flip legacy replay defaults.

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
