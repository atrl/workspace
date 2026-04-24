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
