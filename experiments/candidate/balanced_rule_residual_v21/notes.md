# balanced_rule_residual_v21 — GRU Capacity Increase (128)

## Hypothesis

v19 with gru_hidden_size=64 achieves 635.1% return.
The 2-layer GRU may be capacity-constrained with window=5 and 59 features.
Increasing hidden size to 128 may allow richer temporal representations,
potentially improving return without hurting drawdown.

## Changes from v19

- `gru_hidden_size`: 64 → 128
- Same seed: 12345 (proven good seed)
- `episode_drawdown_penalty`: 2.0 (reverted from v20's failed 3.0)
- All else identical to v19

## Results

(TBD after training)
