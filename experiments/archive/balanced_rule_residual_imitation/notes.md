---
title: balanced_rule_residual_imitation
direction: reward
status: candidate
---

# balanced_rule_residual_imitation

## 背景

Rulebase 策略在 SOXL 上达到 688.6% return，而 RL 模型（v16: 632.4%）持续跑输。
核心问题：alpha reward 只在模型**打败** rulebase 时才给正信号，导致模型过度保守。

## 核心假设

在 rulebase 的"埋点"（关键决策点）上给超额奖励，让模型先学会 rulebase 的择时能力，再用 alpha residual 寻找超额收益。
这是一种轻量级的 Imitation Learning + RL Residual 混合策略。

## 迭代摘要

| 版本 | 核心改动 | 关键结果 | 结论 |
|------|---------|---------|------|
| v1 | rule_imitation_bonus=0.20, rule_decision_bonus=0.30 | proxy 812.3% return | 学习有效但 OOS MaxDD 32.22% |
| v2 | + panic_buy 移除 + bottomless pit protection | test_2022 MaxDD 30.2% | MaxDD 控制最佳，但牛市回报弱 |
| v3 | max_iters=40, alpha_scale=3.5, 增加参与信号 | test_2023_2024 回报改善 | MaxDD 恶化到 34.0% |
| v4 | 保持 v2 紧 drawdown 控制 + 大幅增加 trend_alignment/opportunity_cost | 平衡最佳候选 | MaxDD 接近但未达 30% 门槛 |
| v4_budget | v4 + regime-conditional position budget overlay | 待验证 | 在 v19 fixed actor 上验证有效 |
| v5-v7 | 微调 penalty/bonus 系数 | 边际改善 | 无突破 |

## 核心结论

1. **Imitation bonus 确实有效**：v1 proxy 812.3% 证明模型能学习 rulebase  timing。
2. **MaxDD < 30% 和强回报难以兼得**：v2 MaxDD 控制好但回报弱，v3/v4 回报改善但 MaxDD 上升。
3. **v4 是最佳平衡点**：参与信号 + 紧 drawdown 控制的组合最接近目标。
4. **Regime-conditional budget overlay 是下一步**：在 v4 基础上叠加 runtime 仓位预算，而非继续调 reward 系数。

## 外部参考

- AlphaGo：先 behavior cloning 人类棋谱，再 RL self-play 精修
- RL with expert demonstrations：reward 中加 small KL/matching term 帮助探索

## 下一步

- [ ] 验证 v4_budget 在完整训练流程中的表现
- [ ] 若仍 fail 30% MaxDD gate，转向 external hedge 或更强制的 regime routing
