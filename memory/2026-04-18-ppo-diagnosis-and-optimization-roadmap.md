---
date: '2026-04-18'
title: "PPO \u8BCA\u65AD\u62A5\u544A\u4E0E\u4F18\u5316\u8DEF\u7EBF\u56FE"
kind: review
profiles:
- smoke
status: reference
tags:
- reference
- review
- rule-based
- smoke
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: "Legacy memory note normalized into the repo-wide schema. > \u65E5\u671F\
  : 2026-04-18 > \u80CC\u666F: RL \u65E0\u6CD5\u8D85\u8D8A rule-based  baseline (688.6%)\uFF0C\
  \u6000\u7591 PPO \u672C\u8EAB\u5B58\u5728\u7ED3\u6784\u6027\u95EE\u9898 | \u7814\
  \u7A76 | \u7B97\u6CD5\u6BD4\u8F83 | \u5173\u952E\u7ED3\u8BBA | |------|---------|---------|\
  \ | **FinRL Contest 2023-2024 Review** (Holzer et al., 20"
---

# PPO 诊断报告与优化路线图

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

> 日期: 2026-04-18
> 背景: RL 无法超越 rule-based  baseline (688.6%)，怀疑 PPO 本身存在结构性问题

---

## 一、核心发现：PPO 在金融 Trading 中确实表现较差

### 1.1 多篇独立研究确认 PPO 的劣势

| 研究 | 算法比较 | 关键结论 |
|------|---------|---------|
| **FinRL Contest 2023-2024 Review** (Holzer et al., 2025) | PPO vs SAC vs DDPG | PPO 存在 **policy instability** 和 **sampling bottleneck** 两大问题 |
| **SET50 Portfolio Optimization** (MDPI 2025) | PPO vs A2C vs DDPG vs SAC vs TD3 | **PPO 表现最弱**，cumulative return 和 recovery rate 均垫底 |
| **Stanford CS230 Project** (2021) | PPO vs A2C vs DDPG vs TD3 vs SAC | PPO return 仅 63.7%，远低于 DDPG (142.3%) 和 TD3 (101.9%) |
| **DRL Trading Behavior Analysis** (arXiv 2024) | PPO vs SAC vs TD3 vs DDPG vs A2C | **A2C 是 top performer**；PPO 和 SAC 倾向于频繁交易、持仓时间短 |
| **RL in Financial Decision Making Survey** (arXiv 2025) | 综合对比 | PPO: "Sample efficiency is a major limitation"; "PPO struggles in sparse-reward settings" |

**核心模式**：PPO 在所有金融 trading 基准测试中都排在 **后 50%**。

### 1.2 PPO  vs 其他算法的特性对比

| 特性 | PPO (当前) | SAC-Discrete | A2C | TD3 |
|------|-----------|-------------|-----|-----|
| **Policy Type** | On-policy | Off-policy | On-policy | Off-policy |
| **Sample Efficiency** | **低** ⭐ | **很高** ⭐⭐⭐ | 中等 | 高 |
| **稳定性** | 高 | 高 | 中等 | 高 |
| **金融 Trading 表现** | **弱** | **强** | **强** | 中等 |
| **探索机制** | 随机策略 | **Entropy 最大化** (自动调节) | 随机策略 | 动作噪声 |
| **适合场景** | 模拟环境、游戏 | **数据稀缺、高方差** | 并行训练 | 连续控制 |

**结论**：PPO 的 on-policy 特性和低样本效率，与金融 trading "历史数据有限、不能随意探索" 的本质需求 **存在结构性冲突**。

---

## 二、当前 PPO 实现的 5 个具体问题

### 问题 1：Actor 和 Critic 共享网络（最严重）

```python
net = TiDENet(...)  # 或 MLP
actor = Actor(net, n_actions, device="cpu")
critic = Critic(net, device="cpu")  # 共享同一个 net！
optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=profile.lr)
```

**危害**：
- Value function 学习的是"状态价值"，policy 学习的是"动作偏好"。两个目标的梯度方向经常冲突。
- Policy 更新会干扰 value function 的估计，导致 advantage 计算不准确。
- Critic 被迫学习 policy 不需要的特征，浪费 capacity。

**论文支持**：SPPO (Sequence-Level PPO, 2026) 发现"separate value network"能显著提升性能；Small Critic (1.5B) + Large Policy (7B) 的配置取得最优结果。

### 问题 2：Discount Factor = 0.99，有效视野太短

- Episode length = 300 天
- 0.99^300 ≈ 0.049，意味着 300 天后的 reward 被压缩到不到 5%
- **有效规划视野只有约 3-5 天**，完全无法学习"大波段持仓"这种长周期策略

**对比**：rule-based 策略的核心优势是持有 50-200 天的大波段，但 PPO 的 discount 让它只能看到未来几天。

### 问题 3：GAE Lambda = 0.95，Credit Assignment 偏差大

- GAE(λ=0.95) 在 300 天的 episode 中，credit 会向前传播约 20 步（1/(1-0.95) = 20）
- 一个在第 200 天做出的"减仓避险"决策，其收益要到第 250 天才显现，但 PPO 只能回溯 20 步
- **模型无法学会"提前避险"**，因为它看不到 distant reward

### 问题 4：固定 Entropy Coefficient = 0.01

- Entropy 负责探索。固定 0.01 导致：
  - 前期：探索不足，模型快速陷入局部最优（某个固定仓位）
  - 后期：探索仍然过多，导致不必要的调仓
- SAC 的自动 entropy 调节 (`ent_coef="auto"`) 能根据策略不确定性动态调整探索强度

### 问题 5：On-Policy 样本效率极低

- 当前配置：episode_per_iter=10，每个 episode ~300 步，每轮只有 3000 个样本
- PPO 的 on-policy 特性意味着这些样本只能用一次就丢弃
- 80 iters × 3000 样本 = 240k 总样本
- **对比**：SAC 的 replay buffer 可以复用历史样本，有效样本量可能是 PPO 的 10-100 倍

---

## 三、优化路线图（按 ROI 排序）

### Phase A：PPO 内部调优（保持算法不变）

| 改动 | 预期效果 | 实施难度 | 优先级 |
|------|---------|---------|--------|
| **1. 分离 Actor/Critic 网络** | 价值估计更准确，Advantage 更可靠 | 低 | **P0** |
| **2. Discount factor → 0.999** | 有效视野从 5 天扩展到 50 天 | 低 | **P0** |
| **3. GAE lambda → 0.99** | Credit assignment 从 20 步扩展到 100 步 | 低 | **P0** |
| **4. 动态 entropy schedule** | 前期多探索，后期少探索 | 低 | P1 |
| **5. 增大 batch size / update repeat** | 梯度更稳定 | 低 | P1 |

### Phase B：换算法（tianshou 已支持）

| 算法 | 为什么适合 | 需要改动 | 预期提升 |
|------|-----------|---------|---------|
| **SAC-Discrete** | Off-policy + 自动 entropy + 金融 RL 表现最好 | 需要重写 build_policy，使用 DiscreteSAC | **最大** |
| **A2C** | 同步 A3C，样本效率高于 PPO，金融测试中 top performer | 较小改动，替换 PPOPolicy 为 A2CPolicy | 中等 |
| **Rainbow DQN** | Value-based 更稳定，适合离散动作 | 需要重写为 DQN 风格（无 actor/critic 分离） | 中等 |

### Phase C：训练流程改进

| 方法 | 预期效果 | 难度 |
|------|---------|------|
| **Behavior Cloning 预热** | 用 rule-based 轨迹预训练 policy，避免冷启动 | 中 |
| **Vectorized 并行环境** | 每轮采样量增加 10-50 倍 | 中 |
| **课程学习** | 从低波动 episode 开始，逐步增加难度 | 中 |

---

## 四、关键论文参考

### 金融 RL 综述
1. **"Reinforcement Learning in Financial Decision Making"** (arXiv 2512.10913, 2025)
   - 全面综述金融 RL 的挑战：non-stationarity、sample efficiency、exploration-exploitation
   - 明确指出 PPO 的 sample efficiency 是主要限制

2. **"Revisiting Ensemble Methods for Stock/Crypto Trading"** (ACM ICAIF FinRL 2023-2024)
   - 通过 ensemble PPO+SAC+DDPG 提升稳定性
   - 提出 massively parallel simulation (2048 envs) 解决 sampling bottleneck

### PPO 改进
3. **"Sequence-Level PPO for Long-Horizon Reasoning"** (arXiv 2604.08865, 2026)
   - PPO 的 token-level credit assignment 在长序列中失效
   - **Separate Critic** 是关键改进（Small Critic + Large Policy 取得最优）

4. **"SARD-PPO: Sample-Adaptive Reuse and Dual Clipping"** (Zhao et al., 2023)
   - 改进 PPO 的 clipping 机制和样本复用

### 算法比较
5. **"Deep Reinforcement Learning Strategies in Finance"** (arXiv 2407.09557, 2024)
   - A2C > SAC > TD3 > DDPG > PPO（按 cumulative reward 排序）
   - **A2C 是最适合金融 trading 的 on-policy 算法**

6. **"A Combined Algorithm Approach for Optimizing Portfolio Performance"** (MDPI 2025)
   - PPO 在 COVID-19 市场冲击中 recovery rate 最慢
   - SAC 和 A2C 的 resilience 最强

---

## 五、下一步建议

**最高 ROI 的实验**（今晚可跑 smoke）：

1. **分离 Actor/Critic + 调大 discount/gae**（Phase A 的 1+2+3 组合）
   - 这是 PPO 内部改进中 ROI 最高的组合
   - 不改算法，只改架构和超参数

2. **SAC-Discrete smoke**（Phase B）
   - tianshou 已经支持 `DiscreteSAC`
   - 需要重写 `build_policy()` 和训练循环
   - 但如果 SAC 能在 smoke 中超越 PPO，说明算法是瓶颈

**验证方法**：
- 跑 smoke 对比：current PPO vs 分离网络 PPO vs SAC-Discrete
- 如果分离网络 PPO 有显著提升 → 问题在网络架构
- 如果 SAC-Discrete 显著超越两者 → 问题是 PPO 算法本身

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
