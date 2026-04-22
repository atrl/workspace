# balanced_kelly_warmstart_v9 — Warm-start v4 + Kelly Reward Fine-tuning

## Problem Statement

v5-v8 all failed because training from scratch (20 iters) cannot teach the model
to use Kelly features in a 63-dimensional observation space.  The episodic
drawdown penalty dominates and the model collapses to fixed low position.

## New Approach

1. **Warm-start from v4 checkpoint** — start from an already-good policy
   instead of random initialization.
2. **No feature pack needed** — Kelly values are computed on-the-fly in
   `SoxlSimulator.get_state()` from rolling 21-day returns.  The observation
   space stays at v4's 59 dimensions, so the checkpoint loads cleanly.
3. **Very gentle fine-tuning** — `lr=0.00005`, `max_iters=10`.  Goal is to
   nudge the policy slightly toward higher exposure during Kelly-positive
   windows, not to relearn everything.
4. **Disable staged_training iter-0 fallback** — `staged_training: false`
   prevents the harness from reverting to the initial checkpoint when the
   first training iter degrades performance.

## Kelly Reward Design

```python
if kelly_pos > 0.15 and win_rate > 0.45:
    alignment = max(0.0, 1.0 - abs(pos - min(kelly_pos, 1.0)))
    reward += alignment * kelly_tracking_bonus * signal_quality * vol_attenuation
```

- Threshold lowered to 0.15 (was 0.20 in v8) to capture more rebound signals.
- Bonus set to 0.40 — enough to create positive gradient without overwhelming
  the existing reward structure.

## Baseline

- v4: test_2022 MaxDD 31.3%, WF 51.8%, test_2020 111.6%
- v9 starts from v4 weights, so any regression is immediately visible.

## Experiment Role

`replace` — if successful, v9 replaces v4 as the canonical candidate.

## Changed Layer

- `reward`: kelly_tracking_bonus 0.40 (computed from on-the-fly returns)
- `train`: checkpoint_path loading + lr reduction
- `environment`: `_compute_kelly_from_returns()` in SoxlSimulator
