# TiDE 架构

**方向**: network
**类别**: temporal_decomposition
**状态**: 🟡 No improvement

## 假设

使用时序分解编码器（Temporal Dense Encoder），将时序特征通过投影-编码-瓶颈-解码的流程提取压缩表示。

## 网络结构

```python
TiDENet(
    hidden_size       = 64,
    feature_dim       = ~50,
    observation_window= 5,
    dropout           = 0.1,
)
# FeatureProjection → TemporalEncoder → DenseEncoder(bottleneck) → DenseDecoder → +static → Output
```

**参数规模**: ~40K

## 超参对比

| Profile | hidden | proj_dim | bottleneck | dropout | Return | Sharpe | MaxDD | 状态 |
|---------|--------|----------|------------|---------|--------|--------|-------|------|
| smoke_tide | 64 | max(25,8) | 32 | 0.1 | 307 | 0.69 | 70.6% | 🟡 Moderate |

## 结论

TiDE 在 smoke 配置上未超越 GRU。时序分解的额外复杂度没有转化为更好的 test 表现。可能与 SOXL 的短窗口（5步）特性有关——分解收益需要更长序列。

## 与基线/相关实验的差异

- 与 GRU：结构更复杂（projection + encoder/decoder bottleneck），但效果未提升
- 仅 smoke 配置测试，未在 balanced 完整训练上验证

## 下一步

- [ ] 若未来测试 window=20+，可重新评估 TiDE
- [ ] 当前优先级低
