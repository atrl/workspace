# balanced_kelly_execution_v8 — Kelly-Aware Dynamic Hard Floor

## v5/v6/v7 Failure Analysis

All three reward-shaping approaches failed:
- v5: penalty-based Kelly tracking (0.30)
- v6: aggressive penalty (0.80) + reduced episodic (2.5)
- v7: bonus-based reward (0.60) + episodic (1.5)

Common failure mode: model collapses to fixed ~0.15 position in 2022.
Training valid score degrades from iter-0 (84.70) to iter-20 (73-77).

Root cause: PPO cannot learn to use Kelly features in 20 iters when the
observation space is 63-dimensional and episodic drawdown fear dominates.

## New Approach: Execution-Layer Kelly Control

Instead of asking the RL model to "learn" Kelly, we use Kelly as a
**mechanical control signal** in the execution layer:

```python
if kelly_position_21 > 0.20 and win_rate_21 > 0.48:
    hard_floor = max(hard_floor, min(kelly_pos * 0.7, 0.45))
```

This directly raises the minimum allowed exposure when recent 21-day
statistics show positive edge. The model still chooses actions, but the
action interpreter enforces a higher floor during rebound windows.

## Implementation

- `src/soxl_quant/policy/environment.py`: PositionAction.interpret()
- `src/soxl_quant/policy/evaluation.py`: evaluate_policy()

Both compute `effective_hard_floor` from current row/state Kelly values
before calling target_position_for_profile().

## Hypothesis

By mechanically raising the floor during 2022 Jul/Aug and Nov rebound
windows (Kelly > 0.3), the model will be forced to maintain higher
exposure without needing to "learn" this behavior through reward shaping.

## Baseline

- v4: best balance (test_2022 MaxDD 31.3%, WF 51.8%)
- v7: best valid@20 among Kelly experiments (77.36), but identical test results

## Experiment Role

`ablate` — testing mechanical execution control vs learned reward shaping.

## Changed Layer

- `execution`: Dynamic hard_floor based on Kelly in PositionAction + evaluation
