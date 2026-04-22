# Multi-Asset Generalization Plan for refined_v1

## Context

Previous multi-asset joint training (2026-04-14) used the P0 baseline reward.
Results showed:
- SOXL in-domain improved: 371% → 514% (+143pp)
- But SOXL MaxDD increased: 49.9% → 58.27%
- Cross-asset DDs were ~half of Buy&Hold levels

The key issue: the previous reward lacked sufficient drawdown control.
`refined_v1` strengthens risk guardrails specifically to address this.

## Execution Plan

### Step 1: Validate single-asset refined_v1
Wait for current training to complete. Check:
- proxy MaxDD < 55% (vs v19's 54.8%)
- test_2022 MaxDD < 33% (vs v19's 31.1%)
- promotion gate passes

### Step 2: Multi-asset joint training

Command:
```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles balanced_rule_residual_refined_v1 \
  --skip-data-refresh --seed 12345 --multi-asset
```

This pools episodes from:
- SOXL (semiconductor, 3x)
- TQQQ (NASDAQ, 3x)
- UPRO (S&P 500, 3x)
- SPXL (S&P 500, 3x)
- TECL (technology, 3x)

Asset one-hot embedding adds 5 dimensions to observation (48 → 53).
Validation and test remain SOXL-only for comparability.

### Step 3: Cross-asset evaluation

After joint training, evaluate the same actor on all 5 ETFs:
- Expect SOXL in-domain to improve (richer regime patterns from pooled data)
- Expect cross-asset MaxDD to remain low (previous result: ~half of Buy&Hold)
- Expect TQQQ/TECL gap vs Buy&Hold to narrow further

### Step 4: Decision tree

| single-asset result | multi-asset result | decision |
|--------------------|--------------------|----------|
| Pass gate, DD < 55% | SOXL DD < 55% | Keep multi-asset as candidate |
| Pass gate, DD < 55% | SOXL DD > 55% | Multi-asset needs stronger per-asset risk caps |
| Fail gate | — | Fix single-asset first |

## Risk considerations

1. Multi-asset training takes ~5x longer (more initial states)
2. Asset embedding may dilute SOXL-specific signal
3. Different ETFs have different volatility profiles — need to verify position_cap works across assets

## Expected timeline

- Single-asset training: ~20 min
- Multi-asset training: ~60-90 min
- Evaluation: ~10 min
