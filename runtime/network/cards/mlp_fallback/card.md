# MLP Fallback 架构

**方向**: network
**类别**: dense
**状态**: ⚪ Unused

## 假设

当不使用 GRU/Attention/MoE/TiDE 时，回退到纯 MLP（tianshou Net）。用于快速 baseline 或特征重要性分析。

## 网络结构

```python
Net(
    (obs_dim,),                    # ~261
    hidden_sizes=[128, 64],        # 可配置
    device="cpu",
)
# 纯全连接，无时序建模能力
```

**参数规模**: ~40K

## 超参对比

| Profile | hidden_sizes | obs_window | Return | Sharpe | MaxDD | 状态 |
|---------|-------------|------------|--------|--------|-------|------|
| — | — | — | — | — | — | ⚪ 未实验 |

## 结论

当前没有 profile 使用纯 MLP fallback。所有实验都使用时序模型（GRU/Attention/TiDE）。MLP 仅作为代码回退路径保留。

## 下一步

- [ ] 可作为消融实验：纯 MLP vs GRU 的差距量化
