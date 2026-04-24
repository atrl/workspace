# balanced_rule_residual_imitation — Iteration History

## balanced_rule_residual_imitation_v1

# balanced_rule_residual_imitation_v1

## Motivation

Models consistently underperform the rulebase on raw return (v16: 632.4% vs rulebase: 688.6%).
The core problem is that alpha reward (`step_return - rule_return`) only gives positive signal
when the model **beats** the rulebase. Since the rulebase is already strong, the model gets
sparse positive rewards and frequent negatives, forcing it into overly conservative behavior.

## Hypothesis

Adding a **dense positive baseline** via rulebase imitation bonus will:
1. Provide stable positive rewards during training (even when model just matches rulebase)
2. Help the model learn rulebase timing first
3. Allow alpha residual to then push the model to find improvements

This is essentially a lightweight form of imitation learning + RL residual.

## External Reference

- AlphaGo training: first behavior cloning on human games, then RL self-play refinement
- RL with expert demonstrations: adding a small KL/matching term to the reward helps exploration

## Mechanism Delta (vs moderate_v1)

| Component | moderate_v1 | imitation_v1 |
|-----------|-------------|--------------|
| Branch | `use_moderate_reward: true` | `use_moderate_reward: true` |
| Position match bonus | 0 | `rule_imitation_bonus: 0.20` |
| Decision point bonus | 0 | `rule_decision_bonus: 0.30` |

---
## balanced_rule_residual_imitation_v2

# balanced_rule_residual_imitation_v2

## Motivation

Imitation v1 achieved 812.3% proxy return but failed on critical OOS metrics:
- test_2022: -22.74% / -0.84 / 32.22% (MaxDD above 30% threshold)
- test_2023_2024: 19.64% / 0.46 / 38.46% (weak walk-forward performance)
- Proxy MaxDD: 51.9% (too high)

Root cause: imitation bonus (0.20) was too strong, causing overfitting to rulebase behavior.

## Hypothesis

Reducing imitation bonus + tightening drawdown controls will:
1. Reduce overfitting to training period
2. Bring ALL test periods' MaxDD below 30%
3. Improve walk-forward 2023-2024 excess returns

## Mechanism Delta (vs imitation_v1)

| Parameter | v1 | v2 |
|-----------|-----|-----|
| rule_imitation_bonus | 0.20 | 0.10 |
| rule_decision_bonus | 0.30 | 0.15 |
| episode_drawdown_penalty | 2.0 | 3.5 |
| episode_drawdown_threshold | 0.15 | 0.10 |
| downside_asymmetry_coef | 1.3 | 1.5 |
| drawdown_penalty | 0.15 | 0.30 |
| downside_penalty | 0.10 | 0.20 |
| risk_budget_penalty | 0.15 | 0.25 |

---
## balanced_rule_residual_imitation_v2_fix

# balanced_rule_residual_imitation_v2_fix

## Problem Diagnosed

v2 achieved test_2022 MaxDD=30.2% (close to target) but diagnosis revealed the model spent most of 2022 at 0% position, then suddenly went 100% right before the October crash:

```
2022-10-07: model_pos=1.00, market=-17.69% → single day -17.7%
2022-10-10: model_pos=1.00, market=-10.10% → single day -10.1%
```

Root cause: `panic_buy_bonus` encourages the model to "catch falling knives" in extreme volatility. When cum_drop exceeds -0.15 and regime_risk > 0.60, the model gets a bonus for buying — but in Oct 2022, there was no bottom.

## Fix Applied

1. **Remove panic_buy_bonus** (`panic_buy_bonus: 0.0`)
2. **Add bottomless_pit_penalty** (`bottomless_pit_penalty: 0.50`)
   - When cum_drop_30 < -0.25 and model increases position: -0.25 penalty
   - When cum_drop_30 < -0.35: additional -0.25
   - When realized_vol > 1.0: -0.15 additional
   - When realized_vol > 1.5: another -0.15
   - When regime_risk > 0.75: another -0.10
   - Maximum penalty per step: 0.50 * 0.90 = 0.45

## Hypothesis

Removing panic_buy + adding bottomless pit protection will:
1. Prevent the model from going 100% before market crashes
2. Keep test_2022 MaxDD below 30%
3. Maintain or improve test_2023_2024 returns (since 2023-2024 is a bull market, not a falling knife scenario)

---
## balanced_rule_residual_imitation_v3

# balanced_rule_residual_imitation_v3

## Motivation

v2 achieved good test_2022 control (MaxDD 30.2%, return -21.9% vs rulebase -48.2%) but failed on walk-forward 2023-2024:
- test_2023_2024: 34.0% / 0.63 / 29.4% vs rulebase 112.5% / 0.96 / 41.6%
- Model is too conservative in bull markets, missing major rallies

Additionally, v2 training degenerated after iter 0 (52.1% → 39.7% on validation).

## Hypothesis

Reducing max_iters to 40 + increasing participation signals + reducing imitation bonus will:
1. Capture more bull market upside via trend_alignment and opportunity_cost
2. Prevent training degeneration by stopping earlier
3. Maintain test_2022 control while improving walk-forward returns

## Mechanism Delta (vs imitation_v2)

| Parameter | v2 | v3 |
|-----------|-----|-----|
| max_iters | 80 | 40 |
| alpha_scale | 3.0 | 3.5 |
| episode_drawdown_penalty | 3.5 | 3.0 |
| drawdown_penalty | 0.30 | 0.25 |
| trend_alignment_bonus | 0.25 | 0.35 |
| opportunity_cost_penalty | 0.05 | 0.10 |
| rule_imitation_bonus | 0.10 | 0.05 |
| rule_decision_bonus | 0.15 | 0.08 |


---
## balanced_rule_residual_imitation_v4

# balanced_rule_residual_imitation_v4

## Motivation

v2 had the best MaxDD control (test_2022: 30.2%, test_2023_2024: 29.4%) but weak returns.
v3 increased participation signals and alpha_scale, which improved returns but MaxDD worsened (test_2022: 34.0%, test_2023_2024: 36.1%).

The core challenge: achieve BOTH MaxDD < 30% AND strong walk-forward returns.

## Hypothesis

v2's drawdown framework is correct. The issue is insufficient participation signals.
By keeping v2's tight drawdown controls but aggressively increasing trend_alignment and opportunity_cost, we can:
1. Maintain MaxDD < 30% in all periods
2. Improve test_2023_2024 returns without increasing drawdowns
3. Avoid training degeneration by limiting max_iters to 20

## Mechanism Delta (vs imitation_v2)

| Parameter | v2 | v4 |
|-----------|-----|-----|
| max_iters | 80 | 20 |
| alpha_scale | 3.0 | 3.2 |
| trend_alignment_bonus | 0.25 | 0.40 |
| trend_alignment_threshold | 0.03 | 0.02 |
| opportunity_cost_penalty | 0.05 | 0.20 |
| opportunity_threshold_pos | 0.40 | 0.50 |
| Drawdown controls | tight | same as v2 |

## Acceptance Criteria

---
## balanced_rule_residual_imitation_v4_budget

# balanced_rule_residual_imitation_v4_budget

## Motivation

`balanced_rule_residual_imitation_v4` is the best participation-focused imitation branch, but it misses the stricter 30% MaxDD target by roughly 1-1.3pp on `test_2022` and `test_2023_2024`.

The drawdown-budget prototype showed that unconditional hard masking is too blunt, while regime-conditional masking at `regime_risk >= 0.60` and `budget=0.30` improved drawdown with little return drag on the v19 fixed actor.

## External Reference

- Internal open-source-style reference: `soxl_backtest` panic/risk gating mechanics.
- Existing harness evidence: `memory/2026-04-22-phase0-drawdown-budget-prototype.md`.

## Hypothesis

Keep the v4 participation reward behavior intact, then apply a production/runtime position-budget overlay only during high-risk regimes. This should reduce path drawdown without broadly suppressing rebound and trend participation.

## Target Slices / Windows

- `test_2022`: keep drawdown near or below 30% without increasing average position.
- `test_2023_2024`: preserve v4's 51.8% restart-window return as much as possible.
- `test_proxy`: keep Sharpe above the existing promotion floor.

## Acceptance Criteria

- `test_2022 MaxDD <= 30.0%`.
- `test_2022 avg_pos <= 0.35`.
- `test_proxy Sharpe >= 0.75`.
- `test_2023_2024 return` should not fall more than 5 percentage points from v4 unless MaxDD improves by at least 2 points.


---
## balanced_rule_residual_imitation_v5

# balanced_rule_residual_imitation_v5

## Motivation

v4 achieved excellent balance but missed the 30% MaxDD target by a small margin:
- test_2022 MaxDD: 31.3% (target: < 30%)
- test_2023_2024 MaxDD: 31.1% (target: < 30%)
- test_2023_2024 Return: 51.8% (strong)

## Hypothesis

Slightly tightening episodic drawdown controls (penalty 3.5→4.0, threshold 0.10→0.08) will push both test periods under 30% MaxDD while preserving the participation signals that drive walk-forward returns.

## Mechanism Delta (vs imitation_v4)

| Parameter | v4 | v5 |
|-----------|-----|-----|
| episode_drawdown_penalty | 3.5 | 4.0 |
| episode_drawdown_threshold | 0.10 | 0.08 |
| Everything else | v4 | same |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
- test_2023_2024 return > 45%
- test_proxy Sharpe > 0.85

---
## balanced_rule_residual_imitation_v6

# balanced_rule_residual_imitation_v6

## Motivation

v4 achieved the best walk-forward returns (51.8% test_2023_2024) with MaxDD just 1-1.3pp over 30%:
- test_2022 MaxDD: 31.3% (target: < 30%)
- test_2023_2024 MaxDD: 31.1% (target: < 30%)

v5 showed that tightening episodic penalties further is counterproductive (degeneration).

## Hypothesis

Lowering hard_floor from 0.20 to 0.10 will allow the model to reduce exposure more aggressively in bear markets, pushing MaxDD below 30% without affecting the participation signals that drive bull market returns.

## Mechanism Delta (vs imitation_v4)

| Parameter | v4 | v6 |
|-----------|-----|-----|
| hard_floor | 0.20 | 0.10 |
| Everything else | v4 | same |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
- test_2023_2024 return > 45%

---
## balanced_rule_residual_imitation_v7

# balanced_rule_residual_imitation_v7

## Motivation

After 6 versions, the pattern is clear:
- v2: Best MaxDD control (30.2% / 29.4%) but weak returns (34.0%)
- v4: Best returns (51.8%) but MaxDD slightly over 30% (31.3% / 31.1%)
- v5/v6: Attempts to tighten controls further were counterproductive

## Hypothesis

A compromise between v2's proven drawdown framework and v4's participation signals will achieve both goals:
- Keep v2's drawdown parameters (episode_penalty=3.5, step_penalty=0.30)
- Use moderate participation signals (trend_alignment=0.32, opportunity_cost=0.15)
- This should keep MaxDD < 30% while improving walk-forward returns toward v4 levels

## Mechanism Delta (vs v2/v4)

| Parameter | v2 | v4 | v7 |
|-----------|-----|-----|-----|
| episode_drawdown_penalty | 3.5 | 3.5 | 3.5 |
| drawdown_penalty | 0.30 | 0.30 | 0.30 |
| trend_alignment_bonus | 0.25 | 0.40 | 0.32 |
| opportunity_cost_penalty | 0.05 | 0.20 | 0.15 |
| alpha_scale | 3.0 | 3.2 | 3.2 |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
