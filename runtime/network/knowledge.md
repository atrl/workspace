# NETWORK Knowledge

> 最后更新: 2026-04-24
> 该方向负责网络架构选择、超参实验和模型结构诊断。

## Architecture Comparison Table

| Profile | Arch | Return | Sharpe | MaxDD | AvgPos | FS | WF | Verdict |
|---------|------|--------|--------|-------|--------|----|----|---------|
| balanced_kelly_attention_v10 | Attn | 386 | 0.759 | 59.1 | 0.546 | ❌ | ❌ | 🟡 Moderate |
| smoke_tide | GRU | 307 | 0.693 | 70.6 | 0.615 | ❌ | ❌ | 🟡 Moderate |

### Verdict Legend

- ✅ **Strong** — Sharpe > 1.5 and MaxDD < 30%
- 🟡 **Good** — Sharpe > 0.8
- 🟡 **Gate-pass** — Passed fixed-split or walk-forward gate
- 🟡 **Moderate** — Sharpe > 0.5
- ❌ **Weak** — Sharpe <= 0.5
- ⚪ **Unverified** — No metrics

## Core Conclusions

1. **GRU-64 + [128,64] MLP 是当前主架构**，observation_window=5 显著优于 window=3。
2. **注意力替换 GRU 无提升**：WindowSelfAttentionNet 与 GRUNet 在同等规模下指标无差异。
3. **MoE 未显著超越 dense**：RegimeMoENet 的 noisy Top-1 路由在 SOXL 上未带来可感知的提升。
4. **网络规模边际递减**：从 [128,64] 扩大到 [256,128] 未改善 drawdown，反而增加训练不稳定性。
5. **TiDE 实验中**，时序分解网络未显示出相对于 GRU 的优势。

## 有效方法 ✅

| 方法 | 关键参数 | 验证结果 |
|------|---------|---------|
| GRU-64, window=5 | gru_hidden=64, obs_window=5 | v16: 632% return, Sharpe 0.93, MaxDD 54.8% |
| MLP head [128,64] | hidden_sizes=[128,64] | 平衡了表达力和训练稳定性 |

## 无效方法 ❌

| 方法 | 失败原因 |
|------|---------|
| Attention 替换 GRU | 同规模下指标无差异，计算成本更高 |
| MoE (RegimeMoENet) | 路由负载不均衡，未提升 test 表现 |
| [256,128] MLP | 更大网络未改善 drawdown，训练更不稳定 |

## 待验证方向

- [ ] 跨资产联合训练的网络结构：shared backbone + asset-specific head
- [ ] 更轻量的网络（如 [64,32]）是否足够
- [ ] 1D-CNN 局部特征提取层替代 GRU 的可行性