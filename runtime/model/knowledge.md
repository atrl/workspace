# MODEL Knowledge

> 最后更新: 2026-04-24  
> 该方向负责 checkpoint 选择、promotion gate、walk-forward 验证和模型保留策略。

## 核心结论

1. **当前最佳系列**: `balanced_rule_residual_v19`（canonical 基线）和 `v22`（最新迭代）。
2. **Checkpoint 选择**: validation 最佳 checkpoint 常在 iter 40 而非 iter 80，但选出的 actor 仍可能 fail 2022 gate。
3. **Promotion Gate 硬约束**: test_2022 MaxDD < 35% 且 avg_pos < 0.35，当前无模型能同时满足。
4. **Walk-forward 有效**: 2020/2022/2023-2024 三个测试窗口的 walk-forward 验证能暴露过拟合。
5. **固定切分 vs  walk-forward**: 固定切分通过的模型 walk-forward 可能 fail，后者是更严格的 gold standard。

## 当前最佳模型指标

| 模型 | Return | Sharpe | MaxDD | test_2022 MaxDD | avg_pos | 状态 |
|------|--------|--------|-------|-----------------|---------|------|
| balanced (canonical) | ~480% | ~0.85 | ~63% | ~46% | ~0.38 | canonical |
| balanced_rule_residual_v19 | ~635% | ~0.96 | ~51% | ~39% | ~0.34 | 最佳候选 |
| balanced_rule_residual_v22 | 待补充 | 待补充 | 待补充 | ~33.9% | 待补充 | 最新迭代 |

## 有效方法 ✅

| 方法 | 验证结果 |
|------|---------|
| Checkpoint @60 (v1) | 优于 @80，trade quality 改善 |
| Walk-forward 三窗口 | 2020/2022/2023-2024 验证暴露过拟合 |
| Seed robustness check (v18) | v16 配置 seed 敏感，确认后固定 seed=20260411 |
| Staged checkpoint review | iter 40 常优于 iter 80，但仍 fail 2022 gate |

## 无效方法 ❌

| 方法 | 失败原因 |
|------|---------|
| Phase-balanced sampling | 扩大 bull/restart 采样未修复 2022 excess |
| Structural model loop (v1) | 更长 context + hard regime routing，MaxDD 仍 39.86% |
| Reward-only retrain (reward_guard) | test_2022 MaxDD 恶化到 47.36% |
| Checkpoint review + overlay | position-cap sweep 仅边际改善，未通过 full gate |

## 待验证方向

- [ ] 改变 stress set 或 episode 采样策略（而非调惩罚系数）
- [ ] 引入外部对冲机制（如 VIX call）降低 2022 MaxDD
- [ ] v22 的 walk-forward 完整验证结果

---

## 实验记录（参考）

见 memory archive 原始记录，按日期排序。
