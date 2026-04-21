# Expectations

- Must be judged against canonical `balanced`, not against `smoke`.
- Main question:
  - can residual-over-rule improve trade quality without breaking fixed-split promotion quality?
- Minimum evidence required:
  - summary JSON
  - profile-specific detailed backtest export
  - profile-specific phase1/phase2 diagnosis
  - profile-specific trade-quality audit
- A good run should show:
  - no obvious regression in `test_2022` protection
  - improved or at least non-negative `mid_3_10d` trade quality
  - reduced dependence on weak micro rebalances
