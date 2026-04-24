# balanced_rule_residual_refined_v1

## Motivation

`simplified_v1` failed because it stripped too many guardrails (stale_position,
trapped_loser, downside_signal, explicit risk_budget). The result: proxy MaxDD
worsened from 54.8% → 58.5%, test_2022 return fell from -22.5% → -25.35%.

User feedback: "不是简化reward，而是修改合适的reward" + "cash should be
respected as an asset" + "too many hard-coded reward parameters create
conflicting gradients".

## Hypothesis

The problem is not reward complexity — it is **conflicting incentives**:
- `activity_bonus` (1.25) encourages trading
- `inactivity_penalty` (0.80) punishes holding cash
- `opportunity_cost_penalty` (0.05) punishes low position

These three create a "must trade" pressure that disrespects cash. The model is
forced to always have position, which prevents true risk-off behavior.

Meanwhile, the **risk guardrails** (drawdown_penalty, downside_penalty,
risk_budget_penalty, stale_position, trapped_loser) are actually protective and
should be strengthened, not removed.

## Changes from v19

### Reward: respect cash, strengthen risk

| parameter | v19 | refined_v1 | rationale |
|-----------|-----|-----------|-----------|
| activity_bonus | 1.25 | **0.0** | Remove forced-trading incentive |
| inactivity_penalty | 0.80 | **0.30** | Allow organic cash holding |
| opportunity_cost_penalty | 0.05 | **0.02** | Less FOMO pressure |
| drawdown_penalty | 0.28 | **0.35** | Stronger DD control |
| downside_penalty | 0.20 | **0.25** | Stronger downside protection |
| risk_budget_penalty | 0.24 | **0.30** | Tighter risk budget |
| high_exposure_penalty | 0.08 | **0.12** | Penalize over-exposure |
| episode_drawdown_penalty | 2.0 | **2.5** | Stronger episodic control |
| episode_drawdown_threshold | 0.15 | **0.12** | Earlier trigger |

All other reward components identical to v19 (proven effective):
- alpha reward (Kelly-log, scale=3.0)
- PBRS (coef=0.15)
- turnover penalties
- profit_taking_bonus, panic_buy_bonus
- stale_position, trapped_loser
- downside_asymmetry_coef=1.3

### Network: return to proven scale

| parameter | simplified_v1 | refined_v1 |
|-----------|---------------|-----------|
| hidden_sizes | [256,128,64] | **[128,64]** |
| gru_hidden_size | 128 | **64** |
| episode_len | 500 | **300** |

Larger network + longer episode with same data budget = overfitting.

## Target Improvement Window

- proxy MaxDD: < 54.8% (v19 baseline)
- test_2022 MaxDD: < 31.1% (v19 baseline)
- test_2022 return: better than -22.5%
- Keep proxy return > 600%

## Accepted Trade-off

- May lose some upside capture if cash holding is more common
- Acceptable if Sharpe improves and DD improves

## Failure Mode

- If cash holding becomes excessive (flat_steps too high), may need to
tweak opportunity_cost_penalty or add a minimal activity signal
- If DD still does not improve, the issue is structural (state representation
or execution), not reward

## Comparison Baseline

- Canonical: `balanced_rule_residual_v19`
- Direct compare: `balanced_rule_residual_simplified_v1`
