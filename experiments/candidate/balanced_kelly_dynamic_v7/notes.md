# balanced_kelly_dynamic_v7 — Bonus-Based Kelly Reward

## v5/v6 Failure Analysis

Both v5 (penalty=0.30) and v6 (penalty=0.80, episodic=2.5) produced **identical**
fixed 0.15 position collapse:

- test_2022: MaxDD 33.09%, pos_std ≈ 0.00
- test_2023_2024: Return 45.61%

Root cause identified: **Penalty-based tracking cannot compete with episodic fear**.
When the model keeps low position, it incurs a small Kelly penalty (~0.1-0.3)
but avoids massive episodic drawdown penalty (1.5-3.5). The rational choice is
always "accept the small penalty, avoid the big one."

## New Approach: Positive Bonus

v7 switches from **penalty for deviation** to **bonus for alignment**:

```
reward += alignment * kelly_tracking_bonus * signal_quality * attenuation
```

- Only activates when Kelly > 0.1 (positive edge)
- Signal quality filters: win_rate > 0.45 AND sharpe > 0
- Creates a **positive incentive** to size up during rebound windows

Additionally, `episode_drawdown_penalty` is dropped to **1.5** to break the
fear loop.

## Hypothesis

Positive Kelly bonus will create an "upside pull" that episodic fear cannot
drown. When 2022 Jul/Aug or Nov rebound produces Kelly > 0.3, the model will
receive meaningful bonus (up to ~0.6) for increasing position, making it
rational to depart from minimum exposure.

## Baseline

- v6: identical fixed-position behavior
- v4: best balance baseline (test_2022 MaxDD 31.3%, WF 51.8%)

## Experiment Role

`diagnose` — testing whether positive reward shaping can break the fear loop.

## Changed Layer

- `reward`: kelly_tracking_penalty → kelly_tracking_bonus (0.60), 
  episode_drawdown_penalty 3.5 → 1.5
