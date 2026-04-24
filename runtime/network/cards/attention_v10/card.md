# balanced_kelly_attention_v10

**方向**: network
**类别**: architecture
**状态**: 🟡 Moderate

## 假设

用注意力机制替代 GRU，提升时序特征提取能力。

## 方法/配置

网络架构: Attention
其余配置与对应基线一致。

## 过程

训练 → validation → test 回测 → 与同基线对比。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 386 |
| Sharpe | 0.759 |
| MaxDD | 59.115 |
| AvgPos | 0.546 |

## 结论

Attention 架构表现 Moderate。Sharpe=0.759，MaxDD=59.115%。

## 与基线/相关实验的差异

与 GRU 基线对比：balanced 基线 Sharpe=0.816。

## 下一步

继续观察更大规模数据。
