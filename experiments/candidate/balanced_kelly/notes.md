---
title: balanced_kelly
direction: reward
status: candidate
---

# balanced_kelly

## 背景

v5-v8 全部失败：从零开始训练（20 iters）无法在 63 维观察空间中教会模型使用 Kelly 特征。
Episodic drawdown penalty 主导，模型 collapse 到固定低仓位（~15-20%）。

## 核心假设

从已验证有效的 v4 checkpoint 做 warm-start，用极低学习率（lr=0.00005）做 gentle fine-tuning，
目标是轻微推高 Kelly-positive 窗口的仓位，而非重新学习一切。

## 迭代摘要

| 版本 | 核心改动 | 关键结果 | 结论 |
|------|---------|---------|------|
| v9 | Warm-start from v4 + Kelly fine-tuning (max_iters=10, lr=5e-5) | 待补充完整指标 | 方向正确，需更多验证 |
| v9b | v9 参数微调 | 待补充 | 边际调整 |

## Kelly Reward 设计

```python
if kelly_pos > 0.15 and win_rate > 0.45:
    alignment = max(0.0, 1.0 - abs(pos - min(kelly_pos, 1.0)))
    reward += alignment * kelly_tracking_bonus * signal_quality * vol_attenuation
```

## 核心结论

1. **Warm-start 是必要的**：从零训练无法学会使用 Kelly 特征。
2. **Gentle fine-tuning 策略正确**：lr=5e-5, max_iters=10 避免破坏已有策略。
3. **Kelly 特征 on-the-fly 计算**：不扩展观察维度，从 rolling 21-day returns 实时计算。

## 下一步

- [ ] 补充完整 walk-forward 验证结果
- [ ] 若有效，推广为 canonical reward 组件
