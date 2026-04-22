---
name: rl-kelly-integration
description: >
  结合RL（PPO）与Kelly Criterion进行动态仓位控制的实验方法论。
  涵盖reward shaping、执行层干预、warm-start、episode采样策略等
  多种结合方式的失败教训与可行路径。
---

# RL + Kelly Criterion 整合指南

## 背景

SOXL harness 中的 PPO 模型倾向于 collapse 到固定低仓位（~15-20%）。
Kelly Criterion 提供了基于近期统计优势的动态仓位信号，但直接将 Kelly
特征加入 observation space 并让模型在 20 iters 内学会利用它，已被证明
极其困难。

## 已验证的失败路径

### 1. Reward Shaping（Penalty/Bonus）— v5/v6/v7

**方法**：
- v5: `reward -= deviation * 0.30`（惩罚偏离Kelly建议仓位）
- v6: `reward -= deviation * 0.80` + `episode_drawdown_penalty` 降至 2.5
- v7: `reward += alignment * 0.60`（正向奖励跟随Kelly）

**结果**：所有配置均导致 valid score 从 iter-0 (84.70) 下降到 iter-20 (63-77)。
模型学会固定低仓位作为最优策略，完全忽略 Kelly 信号。

**根因**：
- `episode_drawdown_penalty`（即使降至 1.5）创造了深度风险厌恶
- Kelly bonus/penalty（~0.1-0.6）无法对抗 episodic fear（~1.5-3.5）
- 20 iters 内，63维 observation 空间中 4 个新特征的信号被淹没

### 2. 执行层动态 Hard Floor — v8

**方法**：在 `PositionAction.interpret()` 和 `evaluation.py` 中，当
`kelly_position_21 > 0.20` 时机械式提高 `hard_floor`（0.20 → 0.35）。

**结果**：模型学会了**对抗**这个约束——通过选择更激进的减仓 action
（delta=-0.10）来抵消 floor 提升。valid score 从 84.70 降至 63.29。

**根因**：
- hybrid overlay 架构（base ± delta）允许模型用 action 选择抵消外部约束
- 只调整 floor 不调 cap，给了模型足够的操作空间来规避

## 可行路径（待验证）

### 路径 A：Warm-start from v4 + 少量微调

**前提**：v4 是当前最佳平衡点（test_2022 MaxDD 31.3%，WF 51.8%）。

**步骤**：
1. 创建和 v4 **完全相同的配置**（不加 kelly feature pack，保持 59 维 obs）
2. 从 v4 checkpoint 加载 actor 权重
3. 在 reward.py 中通过 `state.get("kelly_position_21")` 获取 Kelly（直接从 df
   读取，不经过 observation 空间）
4. 使用 **极低学习率**（0.00005）和 **极少 iter**（5-10）微调
5. 关闭 staged_training 的 iter-0 回退（避免 baseline_score > final_score 时回退）

**优势**：
- 不破坏 v4 已学到的能力
- Kelly 只影响 reward，不改变 observation 维度

**风险**：
- 即使微调，也可能破坏 v4 的平衡

### 路径 B：纯规则化 Kelly 仓位（不做 RL）

**方法**：直接用 Kelly 值作为目标仓位的基础：
```python
target_pos = clip(kelly_position_21 * 0.5, 0.15, 0.80)
```

**适用场景**：
- 当 RL 反复无法收敛时，规则化策略是可靠的 fallback
- 可作为 RL 的 baseline 来比较

### 路径 C：修改 Episode 采样策略

**问题**：当前 `episode_len=300`，训练集中约 14 个 episode。
随机采样很少覆盖完整的"暴跌→反弹"周期。

**改进**：
- `episode_len` 增至 500-600（覆盖完整牛熊周期）
- `episode_per_iter` 增至 20-30
- 引入 stratified sampling：确保每个 iter 至少包含 2-3 个反弹窗口

### 路径 D：Feature Adapter（维度对齐加载）

**方法**：
1. 创建 v8 的 actor（63 维 input）
2. 加载 v4 的权重到共享层（除第一层外所有层维度匹配）
3. 第一层新添加的 4 个 Kelly 维度初始化为 0
4. 冻结底层，只训练第一层和顶层

**技术实现**：
```python
v4_state = torch.load("v4_actor.pt")
v8_actor, _ = build_policy(v8_obs_dim, v8_profile)
# 手动映射第一层权重
v8_actor.net[0].weight.data[:, :v4_obs_dim] = v4_state["net.0.weight"][:, :v4_obs_dim]
```

## 关键决策树

```
是否愿意放弃 20-iter 快速迭代？
├── 是 → 路径 A（warm-start + 低 lr 微调）或 路径 D（adapter 加载）
└── 否 → 路径 B（纯规则化）或 路径 C（增加 episode 采样）

observation 维度是否可以改变？
├── 可以 → 路径 D（adapter 加载 v4 + Kelly 特征）
└── 不可以 → 路径 A（v4 配置不变，Kelly 只在 reward 中使用）
```

## 实施检查清单

- [ ] 明确选择路径 A/B/C/D
- [ ] 记录 canonical baseline（v4 的切片指标）
- [ ] 修改代码后先运行 `scripts/check_harness.py`
- [ ] 训练结束后运行诊断脚本验证仓位动态性
- [ ] 更新 `docs/experiment-landscape-review.md` 的比较表

## 反模式

1. **不要**在 20 iters 内期望模型学会新特征（无论正向/负向 reward shaping）
2. **不要**只调整 floor 不调 cap（模型会对抗）
3. **不要**在 staged_training 中允许 iter-0 回退，如果训练目标是超越初始模型
4. **不要**忽略 valid score 下降的信号——如果 iter-20 < iter-0，说明训练在破坏模型
