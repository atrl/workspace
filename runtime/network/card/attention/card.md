# Attention 架构

**方向**: network
**类别**: attention
**状态**: 🟡 No improvement

## 假设

用 multi-head self-attention 替代 GRU，期望通过全局注意力机制更好地捕捉时序特征间的长程依赖。

## 网络结构

```python
WindowSelfAttentionNet(
    feature_dim       = ~50,
    observation_window= 5,
    attention_dim     = 64,
    attention_heads   = 4,
    mlp_hidden_sizes  = [128, 64],
)
# InputProj → +PosEmbed → MultiheadAttention → LayerNorm → mean_pool → MLP
```

**参数规模**: ~55K (含 pos_embed + MHA + MLP)

## 超参对比

| Profile | attention_dim | heads | obs_window | MLP head | Return | Sharpe | MaxDD | 状态 |
|---------|--------------|-------|------------|----------|--------|--------|-------|------|
| balanced_kelly_attention_v10 | 64 | 4 | 5 | [128,64] | 386 | 0.76 | 59.1% | 🟡 Moderate |

## 结论

Attention 替换 GRU 未带来可感知的提升。Sharpe=0.76 甚至略低于同配置的 GRU 基线（0.82）。计算成本更高，性价比不足。

## 与基线/相关实验的差异

- 与 GRU 基线：同规模下指标无差异，但参数量略大（pos_embed + MHA）
- 没有多头或更大 attention_dim 的实验

## 下一步

- [ ] 考虑放弃 Attention 路径，除非配合更大窗口（window=20+）
