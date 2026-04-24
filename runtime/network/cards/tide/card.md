# smoke_tide

**方向**: network
**类别**: architecture
**状态**: 🟡 Moderate

## 假设

使用时序分解编码器（TiDE）替代 GRU。

## 方法/配置

网络架构: TiDE
其余配置与对应基线一致。

## 过程

训练 → validation → test 回测 → 与同基线对比。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 307 |
| Sharpe | 0.693 |
| MaxDD | 70.600 |
| AvgPos | 0.615 |

## 结论

TiDE 架构表现 Moderate。Sharpe=0.693，MaxDD=70.600%。

## 与基线/相关实验的差异

与 GRU 基线对比：balanced 基线 Sharpe=0.816。

## 下一步

继续观察更大规模数据。
