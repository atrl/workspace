# RL 优化方案

## 当前问题

### 结果对比

| 指标 | 硬参数 | RL分层 | 差距 |
|------|--------|--------|------|
| 总收益 | **663.64%** | 238.05% | -425% ❌ |
| Sharpe | **0.95** | 0.90 | -0.05 |
| MaxDD | 45.69% | **19.88%** | -25.81% ✅ |
| 平均仓位 | 0.60 | **0.01** | 过度保守 ❌ |
| 做空比例 | 0% | **51.9%** | 太高 ❌ |

### 核心问题

1. **RL过度保守**：平均仓位只有0.01（应该是0.3-0.6）
2. **做空太多**：51.9%时间在做空（SOXL是3x杠杆，做空风险大）
3. **奖励函数偏差**：回撤惩罚过重，导致RL不敢持仓

---

## 优化方向

### 方案1：调整奖励函数（推荐）

#### 问题诊断

当前奖励：
```python
reward = return_reward + sharpe_reward + dd_penalty + vol_penalty

return_reward = daily_return * 100          # 收益奖励
sharpe_reward = mean / std * 10             # Sharpe奖励
dd_penalty = -drawdown * 50                 # 回撤惩罚（太重！）
vol_penalty = -volatility * 20              # 波动惩罚
```

**问题**：
- 回撤惩罚系数50，远高于收益奖励系数100
- 导致RL选择保守策略（空仓或少仓）

#### 优化方案

```python
reward = (
    return_reward * 1.0 +        # 收益奖励（权重1.0）
    sharpe_reward * 0.5 +        # Sharpe奖励（降低权重）
    dd_penalty * 0.3 +           # 回撤惩罚（从1.0降到0.3）
    vol_penalty * 0.1            # 波动惩罚（从1.0降到0.1）
)

# 新的惩罚计算
dd_penalty = -drawdown * 15      # 从50降到15
vol_penalty = -volatility * 5     # 从20降到5
```

**预期效果**：
- RL会更激进，提高仓位
- 收益提升，回撤略微增加
- 目标：收益 400-500%，回撤 < 30%

---

### 方案2：分层架构调整

#### 当前架构

```python
final_position = rule_signal * rl_position

# 例如：
rule_signal = 1.0 (买入)
rl_position = 0.1 (保守)
final_position = 0.1 (太低)
```

#### 优化方案A：加权混合

```python
# 给RL更多自主权
final_position = (
    0.3 * rule_signal +      # 规则策略权重30%
    0.7 * rl_position        # RL策略权重70%
)

# 例如：
rule_signal = 1.0
rl_position = 0.5
final_position = 0.3*1.0 + 0.7*0.5 = 0.65
```

#### 优化方案B：置信度加权

```python
# 根据RL的置信度调整
confidence = abs(rl_position)  # RL仓位绝对值
final_position = (
    (1 - confidence) * rule_signal +   # RL不确定时，用规则
    confidence * rl_position            # RL确定时，用RL
)
```

---

### 方案3：限制做空

#### 问题

SOXL是3x杠杆ETF，做空风险极大：
- 上涨10% → 做空亏损30%
- 连续上涨 → 爆仓风险

#### 优化方案

```python
# 限制做空范围
rl_position = long_ratio - short_ratio
rl_position = np.clip(rl_position, -0.3, 1.0)  # 限制做空最多30%

# 或完全禁止做空
rl_position = np.clip(rl_position, 0.0, 1.0)  # 只做多
```

**预期效果**：
- 减少做空亏损
- 提高整体收益
- 回撤略微增加

---

### 方案4：训练更长（可选）

#### 当前

- 200k步训练（约80秒）
- 可能未收敛

#### 优化

```python
# 训练500k步
model.learn(total_timesteps=500000)

# 或使用课程学习
for stage in [100k, 200k, 300k, 500k]:
    model.learn(total_timesteps=stage)
    evaluate(model)
```

---

## 推荐优化组合

### 阶段1：快速优化（5分钟）

1. **调整奖励函数**
   - 降低回撤惩罚：50 → 15
   - 降低波动惩罚：20 → 5

2. **限制做空**
   - 最大做空：-0.3

**预期**：
- 收益：400-500%
- MaxDD：25-30%
- 训练时间：5分钟

---

### 阶段2：深度优化（30分钟）

1. 调整分层架构（加权混合）
2. 训练500k步
3. 多版本对比

**预期**：
- 收益：500-600%
- MaxDD：30-35%
- 训练时间：30分钟

---

## 实现优先级

| 优先级 | 优化项 | 预期收益提升 | 实现难度 |
|--------|--------|------------|---------|
| **P0** | 调整奖励函数 | +200-300% | ⭐ 简单 |
| **P0** | 限制做空 | +50-100% | ⭐ 简单 |
| P1 | 加权混合架构 | +100-200% | ⭐⭐ 中等 |
| P2 | 训练更长 | +50-100% | ⭐ 简单 |
| P3 | 多版本对比 | +? | ⭐⭐⭐ 复杂 |

---

## 立即行动

我可以立即实现：

1. **修改奖励函数**（research/rl_hierarchical_env.py）
2. **限制做空范围**（research/rl_hierarchical_env.py）
3. **重新训练**（约5分钟）

需要我开始吗？
