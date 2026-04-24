# GRU 架构

**方向**: network
**类别**: recurrent
**状态**: ✅ Canonical

## 假设

GRU 在处理时序观测窗口（observation_window=5）时，能有效捕捉 regime 转换的短期动态，且训练稳定性优于 LSTM。

## 网络结构

```python
GRUNet(
    input_size  = feature_dim,      # ~50
    hidden_size = gru_hidden_size,  # 64 (default) / 128 (simplified_v1)
    batch_first = True,
)
# GRU 输出 → concat(extra_features) → MLP head
MLP = [gru_hidden + extra] → Linear(hidden[0]) → ReLU → Linear(hidden[1]) → ReLU → ...
```

**输入维度**: `obs_dim = feature_dim * observation_window + extra_dim + n_assets ≈ 50*5 + 10 + 1 = ~261`

**参数规模**: ~50K (gru_hidden=64, mlp=[128,64])

## 超参对比

| Profile | gru_hidden | obs_window | MLP head | Return | Sharpe | MaxDD | 状态 |
|---------|-----------|------------|----------|--------|--------|-------|------|
| balanced (canonical) | 64 | 5 | [128,64] | 97 | 2.16 | 21.1% | ✅ Strong |
| smoke | 64 | 5 | [128,64] | 307 | 0.69 | 70.6% | 🟡 Moderate |
| balanced_rule_residual_simplified_v1 | 128 | 5 | [128,64] | 570 | 0.90 | 52.5% | 🟡 Good |
| ... (其余 60+ profiles 均使用相同配置) | 64 | 5 | [128,64] | — | — | — | — |

## 结论

GRU-64 + [128,64] MLP 是当前主架构，observation_window=5 显著优于 window=3。gru_hidden 从 64 扩大到 128 未带来突破（simplified_v1 的 Sharpe 0.90 vs canonical 的 2.16，但 smoke 基线不同，不可直接对比）。

## 与基线/相关实验的差异

- 与 Attention 对比：同规模下指标无差异，GRU 计算成本更低
- 与 TiDE 对比：GRU 更简单，TiDE 未显示优势
- 与 MoE 对比：MoE 未实际实验

## 下一步

- [ ] 测试 observation_window=10 的边际收益
- [ ] 测试更轻量 GRU-32 + [64,32]
