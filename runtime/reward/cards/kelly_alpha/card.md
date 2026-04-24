# Kelly-Alpha Reward

**方向**: reward
**类别**: kelly_reward
**状态**: 🟡 Incremental

## 假设

在 log 奖励基础上叠加 Kelly Criterion 的信号质量项，利用 rolling win rate 和 profit factor 的置信度来指导仓位决策。

## 奖励公式

```
R = R_log + alpha * signal_quality
# signal_quality: 基于 rolling win rate 和 profit factor 的置信度
```

## 参数与结果对比

| Profile | alpha | 变体 | Return | Sharpe | MaxDD | AvgPos | 状态 |
|---------|-------|------|--------|--------|-------|--------|------|
| balanced_kelly_dynamic_v5 | dynamic | Kelly-alpha dynamic | 682 | 0.90 | 60.3% | 0.56 | 🟡 Good |
| balanced_kelly_dynamic_v6 | dynamic | Kelly-alpha dynamic | 682 | 0.90 | 60.3% | 0.56 | 🟡 Good |
| balanced_kelly_dynamic_v7 | dynamic | Kelly-alpha dynamic | 682 | 0.90 | 60.3% | 0.56 | 🟡 Good |
| balanced_kelly_execution_v8 | dynamic | Kelly-alpha + execution | 682 | 0.90 | 60.3% | 0.56 | 🟡 Good |
| balanced_kelly_warmstart_v9 | warmstart | Kelly warm-start | 584 | 0.91 | 53.4% | 0.49 | 🟡 Good |
| balanced_kelly_warmstart_v9b | warmstart | Kelly warm-start | 584 | 0.91 | 53.4% | 0.49 | 🟡 Good |
| balanced_kelly_attention_v10 | attention | Kelly + Attention 网络 | 386 | 0.76 | 59.1% | 0.55 | 🟡 Moderate |

## 结论

Kelly-alpha 是有效增量，但不同变体（dynamic v5-v7, execution v8）的结果**完全相同**（Sharpe=0.898），说明这些标签可能只是同一实验的重复记录。warmstart 的 Sharpe 略高（0.91）但 MaxDD 更低（53.4%），值得进一步分析。Attention + Kelly 的组合未显示优势。

## 与 Base Log 的差异

- 在 log 基础上增加了信号质量加权
- 理论上应改善方向正确性，但当前结果与 base log 的 canonical（Sharpe=2.16）差距较大——注意这些实验的基线配置不同（非 canonical），不能直接对比

## 下一步

- [ ] 澄清 v5/v6/v7/v8 结果完全相同的原因（是否同一实验）
- [ ] 在 canonical balanced 基线上测试 Kelly reward
- [ ] 精确量化 alpha 的甜蜜点
