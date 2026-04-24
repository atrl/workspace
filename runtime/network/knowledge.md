# NETWORK Knowledge

> 最后更新: 2026-04-24
> 该方向负责网络架构选择、超参实验和模型结构诊断。

## 架构索引

| 架构 | 类别 | 状态 | 实验数 | 关键指标 | 卡片 |
|------|------|------|--------|----------|------|
| GRU | recurrent | ✅ Canonical | 60+ | Sharpe 0.69-2.24 | [card](cards/gru/card.md) |
| Attention | attention | 🟡 No improvement | 1 | Sharpe 0.76 | [card](cards/attention/card.md) |
| TiDE | temporal_decomposition | 🟡 No improvement | 1 | Sharpe 0.69 | [card](cards/tide/card.md) |
| MLP Fallback | dense | ⚪ Unused | 0 | — | [card](cards/mlp_fallback/card.md) |

## 核心结论

1. **GRU-64 + [128,64] MLP 是当前主架构**，observation_window=5。
2. **Attention 替换 GRU 无提升**：同规模下指标无差异，计算成本更高。
3. **TiDE 未超越 GRU**：时序分解网络无优势，可能与短窗口（5步）有关。
4. **MoE 未实际实验**：代码存在但未在 SOXL 上跑过完整训练。

## 待验证方向

- [ ] 跨资产 shared backbone + asset-specific head
- [ ] 更轻量网络 [64,32]
- [ ] 1D-CNN 局部特征提取
