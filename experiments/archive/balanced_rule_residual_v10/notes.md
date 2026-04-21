# v10 Design Notes

## Hypothesis

v2-v9 的超参数探索已证明走到了死胡同：step-level drawdown penalty 无论力度多大都无法有效降低 test MaxDD。论文明确指出：

> "risk indicators are usually calculated based on a period of time data and cannot be assigned to each time step, which makes it difficult for the agent to optimize its strategy due to reward distribution, resulting in poor actual performance."

v10 彻底跳出超参数调优框架，引入两项经过论文验证的机制：

1. **Episode-level max drawdown terminal penalty** — 在 episode 结束时根据整个 episode 的最大回撤给予一次性惩罚
2. **Downside asymmetry penalty** — 当 step return 为负时，额外放大惩罚系数

## Key Changes from v9

### Code-level reward redesign (NEW)
- **`episode_drawdown_penalty: 2.0`** — episode 结束时，若 max_dd > 20%，按 quadratic penalty 惩罚
- **`episode_drawdown_threshold: 0.20`** — 允许最多 20% 回撤不惩罚
- **`downside_asymmetry_coef: 1.30`** — 负收益时额外放大 30% 惩罚

### Revert to v2 proven baseline
- **`drawdown_penalty: 0.28`** — 回到 v2 的 step-level 惩罚力度
- **`downside_penalty: 0.20`** — v2 水平
- **`risk_budget_penalty: 0.24`** — v2 水平
- **`high_exposure_penalty: 0.08`** — v2 水平
- **`execution.base_threshold: 0.04`** — 回到 v2 默认（v8/v9 的 0.06 降低了初始模型质量）
- **`hard_floor: 0.20`** — v2 水平

## Expected Outcome

- Valid score 应该恢复至 v2 的 ~100+ 水平（execution/base_threshold 回归 v2）
- Episode-level penalty 应该直接降低 test MaxDD（论文验证有效）
- Downside asymmetry 应该让 agent 对亏损更敏感，更快止损
- 2022 bear-market de-risking 应该改善

## Risk

- Episode-level penalty 如果太大，可能导致 agent 过于保守（所有论文都提到这个风险）
- Terminal reward 与 PPO 的 credit assignment 机制可能有冲突
- 如果仍然无效，下一步需要尝试：① LSTM/GRU 网络架构 ② 特征工程扩展
