# 2026-04-17: Latest `balanced_floor40_sym` training loses to canonical balanced

## Command

- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_floor40_sym --skip-data-refresh --seed 20260411`

## Latest result

- Fixed split backtest log:
  - `476.2% / Sharpe 0.79 / MaxDD 63.8% / 790 trades`
- Saved summary:
  - `artifacts/models/soxl_balanced_floor40_sym_summary.json`

## Comparison target

- Canonical qlib strategy:
  - `artifacts/models/soxl_balanced_summary.json`
  - `test_proxy = 468.9% / Sharpe 0.858 / MaxDD 50.18%`
  - `promotion.eligible = true`

## Key deltas

- Validation got stronger:
  - `best_score 61.42 -> 80.12`
- But out-of-sample quality got worse:
  - `test_proxy return 468.9% -> 443.7%`
  - `test_proxy sharpe 0.858 -> 0.794`
  - `test_proxy maxDD 50.2% -> 61.5%`
- `2022` degraded materially:
  - return `-25.8% -> -37.0%`
  - maxDD `33.0% -> 45.4%`
  - avg_pos `0.283 -> 0.429`
- `2020` and `2023-2024` improved, but only by carrying more exposure.

## Conclusion

- `balanced_floor40_sym` is **not** the best general qlib strategy in the current harness.
- It is a classic “validation improved, OOS degraded” branch.
- The profile buys upside by staying too invested during stress, which breaks the promotion gate and makes it worse than the canonical balanced model for general use.

## Follow-up

- Keep canonical `balanced` as the baseline for future comparisons.
- If this branch is revisited, the first target should be stress-window de-risking, not longer training.
