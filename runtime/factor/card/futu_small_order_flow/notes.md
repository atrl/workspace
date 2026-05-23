# Futu Small Order Flow Feature Pack

Date: 2026-05-23

## Goal

Create a lagged feature pack from Futu SOXL/SOXX daily capital-flow data and
evaluate whether it is strong enough to justify model training.

## Feature Pack

Pack name: `futu_small_order_flow`

Features:

| Feature | Definition |
|---------|------------|
| `FUTU_SOXL_SMALL_FLOW_ZSCORE_21` | SOXL small-order net inflow 21d z-score |
| `FUTU_SOXX_SMALL_FLOW_ZSCORE_21` | SOXX small-order net inflow 21d z-score |
| `FUTU_SOXL_SOXX_SMALL_FLOW_SPREAD_21` | SOXL small-flow z-score minus SOXX small-flow z-score |
| `FUTU_SOXL_SMALL_FLOW_SHARE` | SOXL small-flow share of absolute order-flow buckets |
| `FUTU_SOXX_SMALL_FLOW_SHARE` | SOXX small-flow share of absolute order-flow buckets |
| `FUTU_SOXL_SMALL_VS_BLOCK_ZSCORE_21` | SOXL small-flow minus big/super-flow 21d z-score |
| `FUTU_STRESS_SMALL_FLOW_CAPITULATION` | negative SOXL small-flow z-score gated by prior stress |

Leakage guard:

- All Futu flow fields are shifted by one trading day before feature
  construction.
- Missing pre-coverage history is filled with zero after feature construction,
  so this pack does not silently cut the full feature matrix down to 2025-only
  rows.

## Data Coverage

Futu capital-flow data is short-history only in the current account/source:

- SOXL daily capital flow: `251` rows, `2025-05-23` to `2026-05-22`
- SOXX daily capital flow: `251` rows, `2025-05-23` to `2026-05-22`

This pack is not valid for full-horizon RL training unless another provider or
subscription supplies pre-2025 small-order-flow history.

## Evaluation

Full-horizon evaluator output is intentionally diluted by zero-valued
pre-coverage rows. It is kept for normal repo comparability:

- Best full-horizon IC row: `FUTU_SOXL_SMALL_FLOW_SHARE`, h20
  `ic_mean=+0.002084`, `ic_positive_ratio=0.029`, `ic_t=2.130`
- Max core correlation remains low: `<=0.095` in the full-horizon correlation
  budget.

Short-history active-window review (`2025-05-23` onward):

| Feature | Best h | IC mean | IC+ | PF | Acc | Max core rho |
|---------|--------|---------|-----|----|-----|--------------|
| `FUTU_SOXL_SMALL_FLOW_SHARE` | 20 | +0.034131 | 0.468 | 1.657 | 0.602 | 0.258 |
| `FUTU_SOXL_SMALL_VS_BLOCK_ZSCORE_21` | 20 | +0.028232 | 0.472 | 1.751 | 0.625 | 0.145 |
| `FUTU_SOXL_SMALL_FLOW_ZSCORE_21` | 20 | +0.028146 | 0.459 | 1.600 | 0.612 | 0.125 |
| `FUTU_SOXL_SOXX_SMALL_FLOW_SPREAD_21` | 20 | +0.020106 | 0.420 | 1.733 | 0.618 | 0.139 |

## Decision

Do not promote this pack into a training profile.

Rationale:

- The best active-window IC mean clears `0.03`, but IC positive ratio is below
  the `0.55` gate.
- Good PF/accuracy rows mostly come from broad long exposure in a short
  post-2025 window, not enough evidence for robust timing alpha.
- Historical coverage is only one year, so full-horizon training would create a
  coverage mismatch.

Keep as a short-history/live overlay candidate only. A valid next branch would
need either a longer order-flow provider or a stricter sparse-event validation
rule designed for live overlays, not core RL features.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_tushare_integration.py -q
# 5 passed

PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_factor_pack.py --pack futu_small_order_flow
# writes runtime/factor/futu_small_order_flow_ic.csv
# writes runtime/factor/card/futu_small_order_flow/
```
