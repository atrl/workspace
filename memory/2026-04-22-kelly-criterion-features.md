---
date: '2026-04-22'
title: "Kelly Criterion Dynamic Position Control \u2014 Implementation"
kind: failure
profiles:
- balanced_kelly_dynamic_v5
- soxl
status: dropped
tags:
- balanced_kelly_dynamic_v5
- dropped
- eval-2022
- failure
- soxl
- walk-forward
artifacts: []
workpad: memory/index.json
summary: Legacy memory note normalized into the repo-wide schema. Implemented rolling
  Kelly criterion features and Kelly tracking reward penalty to address the fixed
  low-position collapse observed in v2/v4 models. 1. **`src/soxl_quant/data.py`**
  Add
---

# Kelly Criterion Dynamic Position Control — Implementation

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What Was Done

Implemented rolling Kelly criterion features and Kelly tracking reward penalty
to address the fixed low-position collapse observed in v2/v4 models.

### Files Modified

1. **`src/soxl_quant/data.py`**
   - Added rolling Kelly calculation in `engineer_feature_frame()`:
     - `kelly_position_21`: Kelly-optimal position based on 21-day win-rate and odds
     - `win_rate_21`: 21-day proportion of up days
     - `odds_21`: avg_win / avg_loss over 21 days
     - `sharpe_21`: 21-day annualized Sharpe ratio
   - Registered `"kelly"` feature pack in `BUILTIN_FEATURE_PACKS`
   - Added Kelly context variables to `context` dict for pack builder

2. **`src/soxl_quant/policy/environment.py`**
   - Added Kelly feature extraction in `get_state()`:
     - `kelly_position_21`, `win_rate_21`, `odds_21`, `sharpe_21`
   - Safe access with fallbacks (0.0, 0.5, 1.0, 0.0) for backward compatibility

3. **`src/soxl_quant/config.py`**
   - Added to `RewardConfig`:
     - `kelly_tracking_penalty: float = 0.0`
     - `kelly_vol_attenuation: bool = True`
     - `kelly_min_signal: float = 0.1`

4. **`src/soxl_quant/policy/reward.py`**
   - Added Kelly tracking penalty in `_moderate_reward()` (section 5b):
     - Penalizes deviation from Kelly-optimal position when |kelly_pos| > min_signal
     - Position target is clipped to [0, 1] (long-only)
     - Volatility attenuation: reduces tracking weight when realized_vol > 1.0

5. **`configs/soxl/profile_registry.yaml`**
   - Registered `balanced_kelly_dynamic_v5` as candidate

6. **New experiment package: `experiments/candidate/balanced_kelly_dynamic_v5/`**
   - `profile.yaml`: Based on v4 (best balance), adds `feature.packs: [kelly]` and
     `reward.kelly_tracking_penalty: 0.30`
   - `notes.md`: Full experiment card with motivation, hypothesis, baseline,
     target improvements, failure modes
   - `expectations.md`: Go/no-go criteria

## Validation

- `scripts/check_harness.py`: 39/39 passed
- Feature frame generation: 757 rows × 63 columns with Kelly features
- Kelly position range on SOXL 2019-2022: [-1.000, 0.669], mean 0.025
- Reward integration test: Kelly tracking penalizes deviation correctly
- Vol attenuation: reduces penalty in extreme vol (realized_vol > 1.0)

## Next Steps

Run `balanced_kelly_dynamic_v5` training:
```bash
qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles balanced_kelly_dynamic_v5 \
  --seed 42
```

Watch for:
- Dynamic position in 2022 (pos_std > 0.10 vs v4's ~0.05)
- Walk-forward 2023-2024 return improvement (target > 80%)
- test_2022 MaxDD staying < 33%

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
