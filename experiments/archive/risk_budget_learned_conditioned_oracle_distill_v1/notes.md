# risk_budget_learned_conditioned_oracle_distill_v1

## Motivation

The previous learned-router oracle distill improved over the older fixed-replay
oracle, but the actor still had to infer the slow regime implicitly. This run
tests the next cheapest change: explicitly condition the fast actor on the slow
mode one-hot during both distillation and evaluation.

## External Reference

- Conditional policy / option-style hierarchy literature: a lower policy often
  benefits from explicit option or mode identity rather than inferring it only
  from raw observation state.
- DAgger / imitation-learning capacity probes remain the frame here; this is
  still no-promotion by construction.

## Hypothesis

If explicit mode conditioning materially improves the learned-router oracle
probe, then the next real training loop should make regime identity an explicit
input to the fast actor instead of relying on implicit recovery.

## Target Slices / Windows

- Diagnostic only on `test_2020`, `test_2022`, `test_2023_2024`, and
  `test_proxy`.
- Compare against `risk_budget_learned_oracle_distill_v1` and
  `hier_learned_slow5`.

## Acceptance Criteria

- The output is marked no-promotion.
- Replay source remains `soxl_hier_learned_slow5.csv`.
- Mode one-hot is appended during both BC dataset construction and runtime
  evaluation.

## Expected Trade-Off

This run still uses held-out replay labels. A good result only proves that
explicit regime conditioning is a useful modeling direction.

## Failure Mode

If explicit conditioning does not materially improve the oracle probe, the next
loop should change model capacity or loss structure rather than just add the
mode feature to PPO.

## Comparison-Table Row Key

`risk_budget_learned_conditioned_oracle_distill_v1`

## Result

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_conditioned_oracle_distill_v1 \
  --replay-daily runtime/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --condition-on-mode \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json runtime/post_train_review/risk_budget_learned_conditioned_oracle_distill_v1.json \
  --output-md runtime/post_train_review/risk_budget_learned_conditioned_oracle_distill_v1.md
```

Evidence:

- samples: `1508`
- label distribution: `{0: 238, 1: 1138, 2: 48, 3: 52, 4: 32}`
- BC loss / accuracy: `1.125 / 0.781`
- `test_2022`: `-11.7% / Sharpe -1.054 / MaxDD 13.2% / avg_pos 0.105`
- `test_2023_2024`: `77.7% / Sharpe 1.114 / MaxDD 24.9% / avg_pos 0.341`
- `test_proxy`: `387.1% / Sharpe 1.008 / MaxDD 30.1% / avg_pos 0.298`

## Decision

Drop as promotable by construction. Explicit slow-mode conditioning did not
improve the learned-router oracle probe versus the unconditioned version
(`387.1%` vs `412.9%` proxy return, slightly worse drawdown), so adding regime
identity alone is not the missing ingredient.
