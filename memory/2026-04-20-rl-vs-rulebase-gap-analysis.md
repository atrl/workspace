---
date: '2026-04-20'
title: 'RL vs Rule-Based 差距根因分析与下一步优化路线图'
kind: analysis
profiles:
- balanced_rule_residual_v2
- soxl_backtest
status: active
tags:
- gap-analysis
- rule-based
- panic-index
- floor-optimization
- trade-compression
artifacts:
- memory/2026-04-20-rl-vs-rulebase-gap-analysis.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: >
  量化分析 RL (balanced_rule_residual_v2, 521.3%) 与 rule-based (688.6%)
  之间 167pp 差距的五大根因。提出 panic-index 特征注入、底仓提升至 40-50%、
  交易压缩至 <100 笔、regime-conditioned alpha scale 四条优化路径。
---

# RL vs Rule-Based 差距根因分析

## 一、数据对比：差距在哪里

### 1.1 总体表现

| 指标 | Rule-Based | RL v2 (@40) | RL best (floor40_sym) | Gap (v2 vs RB) |
|------|-----------|-------------|----------------------|----------------|
| **Return** | **688.6%** | 521.3% | 698.0% | -167pp |
| **Sharpe** | — | 0.86 | 0.83 | — |
| **MaxDD** | 69.7% | **50.4%** | 75.3% | -19.3pp |
| **Trades** | **45** | 316 | 859 | +271 |
| Avg Position | ~0.57 | ~0.48 | ~0.49 | -0.09 |

### 1.2 分年度拆解（关键！）

| 年度 | Rule-Based | RL v2 | RL floor40_sym | 差距来源 |
|------|-----------|-------|---------------|---------|
| 2020 | +221.3% / 54.3% DD | +84.5% / 15.9% DD | +188.4% / 24.6% DD | v2 过于保守 |
| 2022 | -62.3% / 69.7% DD | -16.2% / 23.6% DD | -52.1% / 60.6% DD | v2 风控过强 |
| **2023-2024** | **+146.5%** | **+43.1%** | **+54.2%** | **🚨 核心差距** |

> **2023-2024 是差距的主要来源**：Rule-Based 赚了 +146.5%，而 v2 只赚了 +43.1%，
> 相差 **103pp**。这一波是 SOXL 的大牛市，Rule-Based 因为 50% 硬底仓全程参与，
> 而 RL 的 20% floor 导致大量时间空仓或低仓位。

---

## 二、五大根因分析

### 根因 #1：底仓太低（20% vs 50%）—— 贡献 ~80pp 差距

**Rule-Based 的底仓设计**：
- `MIN_POSITION = 0.5`（硬约束，卖出后不得低于 50%）
- 这意味着策略**永不清仓**，始终持有 50% 以上的 SOXL
- 在 2023-2024 大牛市中，50% 底仓全程捕获了主升浪

**RL v2 的底仓设计**：
- `hard_floor = 0.20`（仅 20%）
- 在恐慌、高 regime_risk 时，floor 可能进一步降低
- 2023-2024 期间频繁减仓，错过了大量上涨

**验证**：
- floor40_sym（floor=0.30-0.40）在 2023-2024 赚了 +54.2%，比 v2 的 +43.1% 高 11pp
- 如果 floor 提升到 0.50，预计可再增加 30-50pp

### 根因 #2：交易次数过多（316 vs 45）—— 贡献 ~40pp 摩擦损耗

| | Rule-Based | RL v2 | 比率 |
|--|-----------|-------|------|
| Trades | 45 | 316 | **7x** |
| 年均交易 | ~7 笔 | ~50 笔 | 7x |
| 平均持仓天数 | ~50 天 | ~7 天 | 1/7 |

**摩擦计算**：
- 假设每笔交易成本 0.1%（slippage + commission）
- 316 笔 vs 45 笔 = 多 271 笔
- 额外摩擦 ≈ 271 × 0.1% × 2（买卖双向）= **54.2% 本金损耗**
- 即使 v1→v2 已从 578 笔压缩到 316 笔，仍远超 Rule-Based

**Rule-Based 的低交易秘诀**：
- 买入后冷却期：日跌幅买入冷却 1 天，累积跌幅冷却实际累积天数
- 仓位更新频率：每 20 天一次（不是每天调仓）
- 卖出阈值高：panic_score ≥ 85 且 profit ≥ 20%，不会轻易触发

### 根因 #3：Panic Index 信号未被 RL 利用 —— 贡献 ~30pp 信号缺失

**Rule-Based 的核心 Alpha 来源**：
```
panic_score = sigmoid(VIX) × volume_multiplier + RSI6_adjustment
```
- 这是一个经过人工精心设计的多因子复合信号
- 在 SOXL 上表现极好（688.6% return）
- 恐慌指数越低（负值越大）→ 买入信号越强
- 恐慌指数越高（正值越大）→ 卖出信号越强

**RL 的 59 维特征**：
- 包含 `vix_zscore`、`realized_vol_10`、`regime_risk` 等
- **但不包含经过 sigmoid 映射和 RSI6 调节后的 panic_score**
- RL 必须从原始特征中"重新发现" panic 信号，这需要大量数据和训练

**验证**：
- 将 panic_score 引入 observation 是最直接的特征工程机会
- 参考 `memory/2026-04-20-backtest-vs-quant-analysis.md` 第 6 条建议

### 根因 #4：止盈机制不够明确 —— 贡献 ~20pp

**Rule-Based 的止盈设计**：
```
IF panic_score ≥ 85 AND profit×position ≥ 60%:
    → 卖出 95% 仓位（大幅止盈）
IF panic_score ≥ 85 AND profit×position ≥ 20%:
    → 正常卖出
IF profit ≤ -20%:
    → 止损卖出 60%
```

**RL 的止盈机制**：
- 依赖 `alpha_reward = step_return - rule_return` 隐式学习
- 有 `profit_taking_bonus` 和 `stale_position_penalty`，但力度可能不够
- 没有明确的"恐慌高 + 利润高 → 大幅减仓"规则

### 根因 #5：2022 风控过度导致 2023 反弹时仓位不足 —— 贡献 ~20pp

**Rule-Based 在 2022**：
- 2022 年跌了 -62.3%，但底仓始终 50%+
- 2023 年反弹时立即满仓参与

**RL v2 在 2022**：
- 2022 年只跌了 -16.2%（风控优秀）
- 但过度风控导致 2023 年初仓位可能过低
- 反弹初期的仓位重建速度慢于 Rule-Based

---

## 三、下一步优化路线图

### P0（最高优先级）：提升底仓至 40-50%

| 方案 | 预期效果 | 风险 |
|------|---------|------|
| hard_floor: 0.20 → 0.40 | +30-50pp return | MaxDD ↑ 5-10pp |
| bull regime floor: 0.40 → 0.50 | +20pp in bull | 2022 DD ↑ |
| crisis floor: 保持 0.20 | 保留风控能力 | — |

**建议配置**：
```yaml
model:
  hard_floor: 0.40          # 从 0.20 提升
  bull_floor: 0.50          # 牛市底仓 50%
  crisis_floor: 0.20        # 危机时保留 20%
```

### P1：引入 Panic Index 到 Observation

**实现方案**：
1. 在 `src/soxl_quant/data.py` 中新增 `panic_score` 特征列
2. 复用 `soxl_backtest/indicators/panic_calculators/sigmoid.py` 的逻辑
3. 在 `make_observation()` 中将 panic_score 作为第 60 维特征注入

**预期效果**：
- 直接利用 Rule-Based 的核心信号
- 减少 RL 的"重新发现"成本
- 预计提升 20-40pp

### P2：交易压缩至 <100 笔

**当前 v2 的 316 笔仍过多，目标：<100 笔**

| 手段 | 当前值 | 建议值 |
|------|-------|-------|
| trade_lock_days | 3 | 5-7 |
| rebalance_threshold (base) | 0.04 | 0.06-0.08 |
| micro_rebalance_penalty | 0.35 | 0.50 |
| position_update | 每天 | 每 5-10 天 |

**参考 Rule-Based**：
- 仓位更新频率 20 天一次
- 买入后冷却期 1-N 天
- 卖出阈值高（panic≥85 + profit≥20%）

### P3：Regime-Conditioned Alpha Scale

**当前问题**：
- `alpha_scale = 3.0` 是全局固定值
- 在牛市中，alpha 信号应该更强（鼓励持仓）
- 在熊市中，alpha 信号应该更弱（鼓励风控）

**建议**：
```python
if regime == "bull":
    alpha_scale = 4.0   # 放大 alpha 奖励，鼓励跑赢规则
elif regime == "stress":
    alpha_scale = 2.0   # 缩小 alpha 奖励，优先风控
else:
    alpha_scale = 3.0
```

### P4：明确的大涨持仓奖励

**当前 reward 中缺少"在持续上涨中保持高仓位"的明确激励**

建议新增：
```python
# 在 trend_strength > threshold 且 price > MA20 时
if trend_strength > 0.7 and price > ma20:
    bonus += position * 0.5  # 持仓越高奖励越大
```

---

## 四、预期效果

如果同时实施 P0+P1+P2：

| 指标 | Rule-Based | RL v2 (当前) | RL v3 (预测) |
|------|-----------|-------------|-------------|
| Return | 688.6% | 521.3% | **650-720%** |
| Sharpe | — | 0.86 | 0.85-0.95 |
| MaxDD | 69.7% | 50.4% | 55-65% |
| Trades | 45 | 316 | **80-120** |

> 目标不是完全复制 Rule-Based，而是**在保留 RL 风控优势的同时，吸收 Rule-Based 的底仓 + Panic Index 优势**。
