# balanced_rule_residual_v18 — Seed Robustness Check

## Hypothesis

v16 (window=5, penalty=2.0, seed=20260420) achieved 632.4% return — the best GRU result.
Is this reproducible with a different seed, or was it seed-dependent luck?

## Changes from v16

- Same config as v16
- Different seed: 42 (was 20260420)

## Expected outcomes

- If v18 ≈ v16 (600%+): the config is robustly strong
- If v18 ≈ v12/v13 (492%): v16 was seed-lucky
- If v18 in between: moderate seed sensitivity
