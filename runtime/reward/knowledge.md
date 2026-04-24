# REWARD Knowledge

> 最后更新: 2026-04-24
> 该方向负责奖励函数设计、shaping 策略和惩罚系数调优。

## 奖励设计索引

| 奖励类型 | 类别 | 状态 | 实验数 | 最佳 Sharpe | 卡片 |
|----------|------|------|--------|-------------|------|
| Base Log | base_reward | ✅ Baseline | 50+ | 2.24 (v19) | [card](cards/base_log/card.md) |
| Kelly-Alpha | kelly_reward | 🟡 Incremental | 7 | 0.91 (warmstart_v9) | [card](cards/kelly_alpha/card.md) |
| Imitation | imitation_reward | 🟡 Incremental | 7 | 0.93 (v6) | [card](cards/imitation/card.md) |

## 核心结论

1. **纯对数奖励是基线**，但单独使用不足以驱动复杂交易。
2. **Kelly-alpha 是有效增量**，但不同变体结果高度重复，需澄清。
3. **Imitation v1 曾达 812% proxy return**，但边际收益递减。
4. **惩罚系数甜蜜点 = 2.0**，4x 惩罚未改善 trade compression。
5. **activity_bonus 是陷阱**：forced trading 导致摩擦，已废弃。

## 待验证方向

- [ ] Opportunity-cost penalty（踏空惩罚）
- [ ] Multi-asset generic reward
- [ ] 动态 imitation weight 衰减
