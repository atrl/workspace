# balanced_kelly_dynamic_v6 — Stronger Kelly Pull, Weaker Episodic Fear

## v5 Failure Analysis

v5 (Kelly tracking=0.30, episode_drawdown=3.5) produced **identical fixed low-position
collapse** as v4:

- test_2022: MaxDD 33.09%, avg position fixed at 0.15 (p10=p90)
- test_2023_2024: Return 45.61% (**worse** than v4's 51.8%)
- Model completely ignored Kelly features

Root cause: `episode_drawdown_penalty=3.5` creates overwhelming risk aversion.
Kelly tracking penalty of 0.30 is ~10x too weak to compete. The model learns
"minimum exposure = survival" and treats Kelly signals as noise.

Evidence: 2022 had **55 days with Kelly > 0.1** (Jul/Aug rebound: Kelly up to 0.64,
Nov rebound: Kelly up to 0.54), yet model never budged from 0.15.

## Hypothesis

If we:
1. **Increase Kelly tracking penalty** from 0.30 → 0.80
2. **Decrease episode_drawdown_penalty** from 3.5 → 2.5

Then the model will have enough "positive pull" from Kelly signals to overcome
episodic fear and dynamically size positions during rebound windows.

## Baseline & Comparison

- **Direct comparison**: v5 (balanced_kelly_dynamic_v5)
  - test_2022 MaxDD: 33.09%, avg_pos: 0.15 (fixed)
  - test_2023_2024 Return: 45.61%
- **Ultimate target**: v4 + dynamic sizing
  - test_2022 MaxDD: < 33%, test_2023_2024 Return: > 80%

## Experiment Role

`diagnose` — testing whether Kelly signals are ignored due to penalty imbalance.

## Changed Layer

- `reward`: kelly_tracking_penalty 0.30 → 0.80, episode_drawdown_penalty 3.5 → 2.5

## Target Improvement Window

- test_2022: Position should become dynamic (std > 0.10). Must capture at least
  some of the Jul/Aug and Nov rebound Kelly signals.
- test_2023_2024: Must exceed v5's 45.61%. Target > 70%.

## Failure Mode

- Even with 0.80 Kelly penalty, model still collapses to fixed position →
  indicates fundamental architectural limitation (GRU can't learn new features
  in 20 iters, or observation space too noisy).
- MaxDD explodes beyond 35% due to weaker episodic constraint.
