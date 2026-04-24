# smoke

**方向**: network
**类别**: architecture
**状态**: 🟡 Moderate

## 假设

测试 GRU 网络架构在 SOXL 上的表现。

## 方法/配置

网络架构: GRU
其余配置与对应基线一致。

## 过程

训练 → validation → test 回测 → 与同基线对比。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 298 |
| Sharpe | 0.683 |
| MaxDD | 76.832 |
| AvgPos | 0.727 |

## 结论

GRU 架构表现 Moderate。Sharpe=0.683，MaxDD=76.832%。

## 与基线/相关实验的差异

与 GRU 基线对比：balanced 基线 Sharpe=0.816。

## 下一步

继续观察更大规模数据。
