---
date: '2026-04-20'
title: "v10: Episode-Level Drawdown Penalty \u2014 Code-Level Reward Redesign"
kind: failure
profiles:
- balanced_rule_residual_v10
- balanced_rule_residual_v2
status: dropped
tags:
- balanced_rule_residual_v10
- balanced_rule_residual_v2
- checkpoint-selection
- dropped
- failure
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20
  Profile: `balanced_rule_residual_v10` Baseline: `balanced_rule_residual_v2` (best
  valid score), `v8/v9` (failed hyperparameter tuning) v8 (strong penalties) and v9
  (m'
---

# v10: Episode-Level Drawdown Penalty — Code-Level Reward Redesign

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

Date: 2026-04-20
Profile: `balanced_rule_residual_v10`
Baseline: `balanced_rule_residual_v2` (best valid score), `v8/v9` (failed hyperparameter tuning)

## Why v8/v9 Failed

v8 (strong penalties) and v9 (moderate penalties) produced **identical results**:
- Test return: 539.3%, Sharpe: 0.85, MaxDD: 53.4%
- Best valid score: 89.54 at iter 0 (untrained model)
- Training process degraded performance from initial random policy

Root cause: **step-level drawdown penalties are fundamentally insufficient** for controlling episode-level max drawdown. The agent cannot learn to optimize a metric that is computed over the entire episode when the penalty is distributed per-step.

## Paper Evidence

From "Stock Trading Strategy Developing Based on ...":
> "A straightforward idea is to add the maximum drawdown over a period of time as a penalty to the reward function, but risk indicators are usually calculated based on a period of time data and cannot be assigned to each time step, which makes it difficult for the agent to optimize its strategy due to reward distribution, resulting in poor actual performance."

## v10 Solution

Two code-level changes to `src/soxl_quant/policy/reward.py` and `environment.py`:

### 1. Episode-Level Max Drawdown Terminal Penalty

When `done=True`, penalize the agent based on the episode's maximum drawdown:

```python
if state.get("done", False) and cfg.episode_drawdown_penalty > 0:
    episode_max_dd = float(state.get("episode_max_dd", 0.0))
    excess_episode_dd = max(episode_max_dd - cfg.episode_drawdown_threshold, 0.0)
    reward -= (excess_episode_dd ** 2) * cfg.episode_drawdown_penalty * 100.0
```

- `episode_drawdown_penalty: 2.0`
- `episode_drawdown_threshold: 0.20` (allow up to 20% without penalty)
- Quadratic penalty: small drawdowns → small penalty, large drawdowns → severe penalty

### 2. Downside Asymmetry Penalty

When step return is negative, amplify the penalty:

```python
if cfg.downside_asymmetry_coef > 1.0 and pnl_ret < 0:
    asymmetry_penalty = abs(pnl_ret) * (cfg.downside_asymmetry_coef - 1.0) * 100.0
    reward -= asymmetry_penalty
```

- `downside_asymmetry_coef: 1.30` (30% extra penalty on losses)
- Based on paper finding: multiplying negative reward by 1.05x improves risk aversion

### 3. Revert to v2 Proven Baseline

v8/v9 degraded performance by changing execution settings (base_threshold=0.06). v10 reverts:
- `execution.base_threshold: 0.04` (v2 default)
- `drawdown_penalty: 0.28` (v2 level)
- `hard_floor: 0.20` (v2 level)

## Expected Outcome

- Valid score should recover to v2's ~100+ level
- Episode-level penalty should directly reduce test MaxDD
- Downside asymmetry should make agent more loss-sensitive

## If v10 Fails

Next steps:
1. **Network architecture**: Add LSTM/GRU layers to MLP
2. **Feature engineering**: Expand feature set beyond 59 dimensions
3. **Alternative algorithms**: Try SAC or DQN instead of PPO

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
