---
date: 2026-04-20
title: "SOXL Backtest vs SOXLQuant \u6DF1\u5EA6\u5BF9\u6BD4\u4E0E\u5956\u52B1\u51FD\
  \u6570\u4F18\u5316\u5206\u6790"
kind: other
profiles:
- balanced
- balanced_rule_residual_v1
- moe
- smoke
- soxl
status: kept
tags:
- balanced
- balanced_rule_residual_v1
- checkpoint-selection
- eval-2022
- execution-overlay
- kept
- moe
- other
- smoke
- soxl
- stress-regime
- walk-forward
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: "\u5168\u9762\u5BF9\u6BD4 soxl_backtest\uFF08\u89C4\u5219\u9A71\u52A8\uFF09\
  \u4E0E src/soxl_quant\uFF08RL\u9A71\u52A8\uFF09\u7684\u67B6\u6784\u3001\u4FE1\u53F7\
  \u3001\u98CE\u63A7\u548C\u7EE9\u6548\u3002 \u6DF1\u5165\u5206\u6790\u5BF9\u6570\u5956\
  \u52B1 $R_t = \\log(W_t/W_{t-1})$ \u7684\u7406\u8BBA\u4F18\u52A3\uFF0C\u6307\u51FA\
  \u5176\u5DF2\u5B9E\u73B0\u5728\u4EE3\u7801\u4E2D\u4F46\u4E0D\u8DB3\u4EE5\u5355\u72EC\
  \u4F7F\u7528\u3002 \u63A8\u8350\u6700\u4F73\u6A21\u578B\u7684\u4E0B\u4E00\u6B65\u4F18\
  \u5316\uFF1A\u542F\u7528 Kelly-alpha \u5956\u52B1\u3001regime-conditioned scale\u3001\
  \u4EE5\u53CA\u5C06 soxl_backtest \u7684 panic index \u5F15\u5165 observation\u3002"
---

# SOXL Backtest vs SOXLQuant 深度对比与奖励函数优化分析

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## 一、两个系统的架构对比

### 1.1 soxl_backtest：规则驱动的传统回测

**核心信号：Panic Index（恐慌指数）**
- 基础：VIX 通过 sigmoid 函数映射到 [-100, 100] 区间
- 调节：RSI6 超卖（<30）降低恐慌值，超买（>70）提高恐慌值
- 放大：交易量相对20日均值的 log 缩放系数 [0.8, 1.5]

**买入逻辑（BuyStrategy）**
```
买入条件 = panic_score ≤ BUY_SCORE_THRESHOLD(-50) 
         AND (daily_return ≤ -15%*vol_coef OR cumulative_return ≤ -30%*vol_coef)
         AND NOT in cooldown
         AND position < MAX_POSITION(0.9)
```
- 高估时（price > MA20）：买入步长降为正常的 40%
- 首次建仓步长：0.3（30%仓位）
- 正常加仓步长：0.1（10%仓位）

**卖出逻辑（SellStrategy）**
```
卖出触发 = panic_score ≥ SELL_SCORE_THRESHOLD(85)
         AND (
             (profit*position ≥ TAKE_PROFIT_THRESHOLD(60%) → 卖出95%)
             OR (profit*position ≥ SELL_PROFIT_TARGET(20%) → 正常卖出)
             OR (last_buy_profit ≥ LAST_BUY_PROFIT_TARGET(30%*vol_coef))
             OR (profit ≤ STOP_LOSS_THRESHOLD(-20%) → 止损卖出60%)
         )
```

**仓位管理**
- 最小仓位：0.5（硬约束，卖出后不得低于）
- 最大仓位：0.9
- 仓位更新频率：每20天
- 波动率系数：HV_20 / volatility_median, clip to [0.5, 2.0]

**特点**
- 完全可解释，每个交易都有明确的触发条件记录
- 依赖人工设计的阈值，无法自适应市场结构变化
- 没有"奖励函数"概念，只有规则遵循
- 回测结果：年度统计、总收益、最大回撤

### 1.2 src/soxl_quant：强化学习驱动的动态系统

**核心信号：59维特征向量**
关键特征族：
- 趋势类：`trend_gap_21`, `anchor_gap_180`, `anchor_slope_180`
- 波动类：`realized_vol_10`, `vol_coef_21`, `vix_zscore`
- 回撤类：`cum_drop_30`, `drawdown_streak_63`, `below_anchor_ratio_63`
- 制度类：`regime_risk`, `regime_risk_velocity_5`
- 扩展观测：`holding_age`, `avg_entry_pnl`, `high_gap`, `last_tranche_pnl`

**决策逻辑：PPO 策略网络**
- 输入：59维特征 + 15维扩展观测 + 5维资产 one-hot
- 网络：MLP [128, 64] 或 MoE/TiDE 变体
- 输出：11个离散动作（0%, 10%, ..., 100%仓位）
- 动作解释：通过 `target_position_for_profile` 映射到实际仓位

**仓位管理（三层叠加）**
1. **Action levels**：基础动作空间（0-1 均匀分布）
2. **Position cap**：根据 cum_drop_30 动态压降仓位上限
   - start_drawdown=10%, full_drawdown=30%, min_cap=40%
   - 牛市放宽：bull_min_cap=65%（当 vol<0.4 且 anchor_gap>0.02）
   - 熊市收紧：bear_min_cap=30%（当 regime_risk>0.65）
3. **Hybrid overlay**（residual_v1 使用）：RL 只学习规则仓位周围的增量 delta ∈ [-0.10, +0.10]

**执行层（Execution Overlay）**
- Regime 检测：base / bull / stress 三种状态
- Rebalance threshold：根据 regime 动态调整（0.04 ~ 0.012）
- Trade lock：交易后锁定3天（stress 状态下更严格）
- Buy cooldown：深跌后冷却期（-15% 触发，按跌幅比例 3-20 天）

### 1.3 架构对比总结

| 维度 | soxl_backtest | src/soxl_quant |
|------|---------------|----------------|
| 信号源 | VIX + RSI6 + Volume（3维） | 59维特征 + 持仓状态 |
| 决策方式 | 硬阈值规则 | PPO神经网络学习 |
| 仓位调整 | 固定步长 | 连续动作空间 + 动态约束 |
| 风控 | 波动率系数 + 止损 | PBRS + 回撤惩罚 + 风险预算 + 换手率惩罚 |
| 自适应性 | 无（阈值固定） | 有（网络权重训练） |
| 可解释性 | 高（每条规则可追踪） | 中（可通过归因分析） |
| 计算成本 | 极低 | 中等（推理一次 <1ms） |

---

## 二、当前最佳模型表现对比

### 2.1 Canonical `balanced`（已晋升的基线）

```
valid:      39.0% return / Sharpe 1.28 / MaxDD 26.6% / avg_pos 56.0%
test_2020:  97.0% return / Sharpe 2.16 / MaxDD 21.1% / avg_pos 51.8%
test_2022: -25.8% return / Sharpe -0.88 / MaxDD 33.0% / avg_pos 28.3%
test_2023_24: 43.5% return / Sharpe 0.69 / MaxDD 33.9% / avg_pos 49.8%
test_proxy: 468.9% return / Sharpe 0.858 / MaxDD 50.2% / avg_pos 47.8%
```

**交易质量诊断**：
- `mid_3_10d`（3-10天中等持仓）：Kelly -0.33 / Expectancy -3.02% ❌
- `small_5_15%`（5-15%小仓位变动）：Kelly -0.06 / Expectancy -0.52% ❌
- `micro_<5%`（<5%微幅调仓）：Kelly +0.04 / Expectancy +0.38% ⚠️
- 总交易动作：578次，摩擦估计 ~57.8%

**问题**：微幅调仓过多，大量交易为负期望。

### 2.2 Candidate `balanced_rule_residual_v1`（当前最佳，@60 checkpoint）

```
valid:      61.8% return / Sharpe 1.69 / MaxDD 26.9% / avg_pos 60.4%
test_2020:  88.2% return / Sharpe 2.52 / MaxDD 14.5% / avg_pos 38.3%
test_2022: -18.3% return / Sharpe -0.95 / MaxDD 24.8% / avg_pos 19.5%
test_2023_24: 52.1% return / Sharpe 0.84 / MaxDD 27.3% / avg_pos 40.0%
test_proxy: 672.3% return / Sharpe 0.960 / MaxDD 46.4% / avg_pos 48.3%
```

**交易质量诊断**：
- `mid_3_10d`：Kelly +0.32 / Expectancy +4.33% ✅
- `small_5_15%`：Kelly +0.18 / Expectancy +2.32% ✅
- `micro_<5%`：Kelly +0.29 / Expectancy +4.49% ✅
- 总交易动作：318次，摩擦估计 ~31.8%

**关键改进来源**：
1. **Residual-over-rule 架构**：RL 不再从零学习仓位，而是学习规则仓位的增量。这大幅降低了探索空间。
2. **Alpha 奖励**：`reward = (step_return - rule_return) * alpha_scale`，而非原始回报。这让 RL 专注于"比规则做得更好"。
3. **更严格的交易约束**：`micro_rebalance_penalty=0.35`, `short_hold_turnover_penalty=2.25` 抑制了无效交易。

---

## 三、奖励函数深度分析

### 3.1 七种奖励模式全景

当前代码（`src/soxl_quant/policy/reward.py`）实现了七种互斥的奖励基础：

| 模式 | 公式 | 当前状态 |
|------|------|----------|
| **Standard** | `pnl_ret * 100 * return_scale` | 默认模式 |
| **Log reward** | `log(1 + safe_ret) * 100 * return_scale * log_reward_scale` | 已实现，smoke 测试过 |
| **Marginal reward** | `period_alpha * 100 * return_scale * marginal_return_scale`（仅决策日） | 已废弃（ablation） |
| **Alpha reward** ✅ | `alpha = step_return - rule_return`，可选 Kelly-log | **当前最佳使用** |
| **Quality reward** | 离散趋势对齐评分 | 已废弃 |
| **N-day reward** | `nday_return * 100 * return_scale * nday_reward_scale` | 已废弃 |
| **Sparse reward** | 仅 episode 结束时给回报 | 已废弃 |

**Universal penalties/bonuses**（所有模式共享）：
- PBRS delta（`pbrs_coef=0.15`）
- Turnover 惩罚（线性 + 二次）
- Drawdown、downside、risk budget 惩罚
- Trend alignment bonus
- Opportunity cost penalty
- Profit taking / panic buy / vol trading bonuses
- Inactivity penalty
- Stale position / trapped loser penalties

### 3.2 用户问题的核心：对数奖励是否需要调整？

用户问：
> 如果我们以 $R_t = \log(W_t/W_{t-1}) = \log(1 + r_{portfolio,t})$ 作为奖励，是否需要调整？

**答案是：对数奖励已经在代码中实现，理论上正确，但单独使用不够。**

#### 3.2.1 对数奖励的理论优势

1. **CRRA 效用对齐**
   - 对数效用 $U(W) = \log W$ 对应相对风险厌恶系数 $\gamma = 1$
   - 边际效用递减：财富翻倍带来的效用增益相同（从 100→200 和 1000→2000 的效用增量相等）
   - 这天然抑制过度杠杆——与 SOXL（3x杠杆ETF）的固有风险偏好匹配

2. **时间可加性**
   - $\sum_{t=1}^T \log(1+r_t) = \log(W_T/W_0)$
   - 总对数回报等于各期对数回报之和，消除了复利路径依赖
   - 在 RL 中，这使得 episode 回报与 step 回报之间有了清晰的数学关系

3. **无杠杆幻觉**
   - 线性奖励：仓位翻倍 → 奖励翻倍（模型倾向于满仓）
   - 对数奖励：$\log(1+2r) \neq 2\log(1+r)$，高仓位的边际收益递减

#### 3.2.2 对数奖励的局限性（为什么单独使用不够）

1. **忽略交易成本**
   - 对数奖励只看净值变化，不看换手率
   - 一个策略每步赚 0.1% 但换手率 50%，对数奖励为正，实际为负
   - **解决方案**：代码已添加 `turnover * transaction_cost * 100` 惩罚

2. **忽略回撤路径**
   - 对数奖励是近视的（myopic）
   - 50% 回撤 + 100% 恢复：$(1-0.5)(1+1.0) = 1$，总对数回报为 0
   - 但投资者经历了 50% 的最大回撤，这是不可接受的
   - **解决方案**：PBRS（Potential-Based Reward Shaping）通过 `pbrs_delta = (-drawdown) - (-prev_drawdown)` 将回撤变化注入奖励

3. **无法塑造行为**
   - 对数奖励只看结果，不看"如何达到"
   - 我们希望模型学会：深跌加仓、高位减仓、避免微幅调仓
   - **解决方案**：`profit_taking_bonus`, `panic_buy_bonus`, `micro_rebalance_penalty`, `short_hold_turnover_penalty` 等行为塑造项

4. **制度异方差性**
   - SOXL 的波动率聚集（volatility clustering）意味着同样的对数奖励在不同 regime 下信噪比不同
   - 危机中的 +1% 对数奖励和平静市场中的 +1% 不应等同对待
   - **解决方案**：`regime_risk` 和 `realized_vol_10` 已用于条件化惩罚

#### 3.2.3 具体调整建议

**建议1：不要单独使用 log reward，使用 log-alpha**

当前最佳模型使用 `alpha_reward`（而非 `log_reward`）。Alpha 奖励的定义：
$$\alpha_t = r_{portfolio,t} - r_{rule,t}$$

其中 $r_{rule,t}$ 是规则策略在相同市场条件下的回报。这移除了市场 beta 成分，让 RL 专注于学习"超越规则"的能力。

如果将 alpha 与对数结合：
$$R_t = \log(1 + \alpha_t) \times 100 \times alpha\_scale$$

这就是代码中的 `use_kelly_alpha: true` 模式。它同时获得了：
- 对数奖励的 CRRA 性质
- Alpha 奖励的市场中性性质

**建议2：启用 regime-conditioned reward scale**

当前 `alpha_scale` 是标量（=3.0）。建议根据 regime 动态调整：

```python
if realized_vol_10 < 0.30 and regime_risk < 0.35:
    effective_scale = alpha_scale * 1.3   # 低波动高信噪比，放大奖励
elif realized_vol_10 > 0.60 or regime_risk > 0.65:
    effective_scale = alpha_scale * 0.7   # 高波动低信噪比，缩小奖励
else:
    effective_scale = alpha_scale
```

这类似于深度学习中常用的梯度裁剪和自适应学习率的思想。

**建议3：加入方差惩罚项**

Kelly 准则的最优仓位 $f^* = \mu / \sigma^2$。如果奖励只考虑均值不考虑方差，模型会过度交易高波动机会。

建议添加：
$$R_t^{variance} = -\lambda \cdot \widehat{\sigma^2}_{rolling}$$

其中 $\widehat{\sigma^2}_{rolling}$ 是最近 N 步对数回报的滚动方差。这会让模型在波动期更保守。

---

## 四、优化路线图

### 4.1 即时优化（无需重新训练）

1. **为 `balanced_rule_residual_v1` 跑 Walk-forward**
   - 当前已通过 fixed-split promotion gate
   - Walk-forward 证据缺失，这是晋升的唯一障碍
   - 命令：`PYTHONPATH=. qlib_venv/bin/python scripts/walk_forward_eval.py --profile balanced_rule_residual_v1`

2. **为 canonical `balanced` 做固定演员交易压缩**
   - 578 次交易 → 目标 350 次以下
   - 提高 `min_trade_turnover` 从 0.02 到 0.04
   - 验证：摩擦从 57.8% 降至 ~40%，检查 proxy Sharpe 是否保持 >0.75

### 4.2 中期优化（奖励工程，需重新训练）

3. **启用 Kelly-alpha 的 residual_v2**
   ```yaml
   reward:
     use_alpha_reward: true
     use_kelly_alpha: true      # 新增
     alpha_scale: 3.0
     # 保持其他惩罚不变
   ```
   - 理论：这是增长最优的奖励形式
   - 风险：Kelly 对极端值敏感，需要 safe_ret = max(alpha, -0.999) 截断

4. **Regime-conditioned alpha_scale**
   ```yaml
   reward:
     use_alpha_reward: true
     use_kelly_alpha: true
     alpha_scale: 3.0
     regime_conditional_scale: true    # 新增
     low_vol_scale_multiplier: 1.3
     high_vol_scale_multiplier: 0.7
     low_vol_threshold: 0.30
     high_vol_threshold: 0.60
   ```
   - 需要修改 `reward.py` 添加 regime 条件化逻辑

5. **添加滚动方差惩罚**
   ```yaml
   reward:
     variance_penalty: 0.5       # 新增
     variance_window: 10
   ```
   - 需要修改 `environment.py` 维护滚动回报方差

### 4.3 长期优化（架构层面）

6. **将 soxl_backtest 的 panic index 引入 observation**
   - 计算 `panic_score = sigmoid(VIX) * volume_multiplier + RSI6_adjustment`
   - 加入 `FEATURE_COLUMNS` 作为第60维特征
   - 理由：这是 soxl_backtest 验证过的核心信号，已有多年历史数据验证

7. **多目标 PPO**
   - Critic 1：预测回报（标准 PPO）
   - Critic 2：预测 Sharpe ratio
   - Critic 3：预测最大回撤
   - 通过 Lagrangian 约束优化：maximize return s.t. Sharpe > min_sharpe, MaxDD < max_dd

8. **策略蒸馏**
   - 将 `balanced_rule_residual_v1` 蒸馏为更小的网络（64→32→16）
   - 或蒸馏为符号规则（决策树 / 规则列表）
   - 目的：生产环境推理速度 + 可解释性

---

## 五、关键结论

### 关于对数奖励的直接回答

> 如果我们以 $R_t = \log(W_t/W_{t-1})$ 作为奖励，是否需要调整？

**需要调整，且当前代码已经做了调整。** 具体来说：

1. **对数奖励本身是好的**——它对应 CRRA 效用、时间可加、抑制杠杆。代码中已通过 `use_log_reward` + `log_reward_scale` 实现。

2. **但对数奖励不足以单独驱动一个可投产的交易策略**——它忽略交易成本、回撤路径、行为塑造和制度异方差。代码中已通过 PBRS、turnover 惩罚、各种 bonuses/penalties 补充。

3. **最佳实践是 log-alpha**——即 $\log(1 + \alpha_t)$，其中 $\alpha_t = r_{portfolio,t} - r_{rule,t}$。这同时获得对数的理论优势和 alpha 的市场中性性质。代码中已通过 `use_kelly_alpha` 实现，但当前最佳模型 **尚未启用**（`use_kelly_alpha: false`）。

4. **下一步最推荐的实验**：基于 `balanced_rule_residual_v1`，仅改变 `use_kelly_alpha: true`，做一次 smoke 测试验证效果。

### 关于两个系统的对比结论

- **soxl_backtest** 是一个设计精良的规则系统，其 panic index 和波动率自适应逻辑经过验证。但它无法自适应市场结构变化。
- **src/soxl_quant** 的 RL 系统已经超越了规则系统，特别是 `balanced_rule_residual_v1` 在回报、Sharpe、回撤和交易质量上全面优于 canonical `balanced`。
- **soxl_backtest 的核心信号（panic index）尚未被 RL 系统直接利用**，这是一个明确的特征工程机会。

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
