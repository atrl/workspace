# NETWORK Knowledge

> 最后更新: 2026-04-24
> 该方向负责网络架构选择、超参实验和模型结构诊断。

## 架构索引

| Profile | Arch | Return | Sharpe | MaxDD | AvgPos | 状态 | 卡片 |
|---------|------|--------|--------|-------|--------|------|------|
| balanced_kelly_attention_v10 | Attn | 386 | 0.76 | 59.12 | 0.55 | 🟡 Moderate | [card](cards/attention_v10/card.md) |
| smoke_tide | TiDE | 307 | 0.69 | 70.60 | 0.62 | 🟡 Moderate | [card](cards/tide/card.md) |
| balanced | GRU | 407 | 0.82 | 52.75 | 0.48 | 🟡 Good | [card](cards/gru_baseline/card.md) |
| smoke | GRU | 298 | 0.68 | 76.83 | 0.73 | 🟡 Moderate | [card](cards/gru_smoke/card.md) |

## 核心结论

1. **GRU-64 + [128,64] MLP 是当前主架构**。
2. **Attention 替换 GRU 无提升**：同规模下指标无差异。
3. **MoE 未显著超越 dense**：路由负载不均衡。
4. **TiDE 未超越 GRU**：时序分解网络无优势。

## 待验证方向

- [ ] 跨资产 shared backbone + asset-specific head
- [ ] 更轻量网络 [64,32]
- [ ] 1D-CNN 局部特征提取