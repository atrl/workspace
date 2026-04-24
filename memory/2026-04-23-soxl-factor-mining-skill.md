---
date: '2026-04-23'
title: "SOXL \u7279\u5F81\u6316\u6398 Skill \u9002\u914D\u5B8C\u6210"
kind: skill
profiles:
- soxl
status: dropped
tags:
- dropped
- skill
- soxl
- stress-regime
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: "Legacy memory note normalized into the repo-wide schema. \u5C06 agentic-alpha\
  \ \u7684\u56E0\u5B50\u6316\u6398\u65B9\u6CD5\u8BBA\u9002\u914D\u5230 SOXL \u5355\
  \u6807\u7684\u62E9\u65F6\u573A\u666F\uFF0C\u5EFA\u7ACB\u72EC\u7ACB\u4E8E RL \u8BAD\
  \u7EC3\u7684\u7279\u5F81\u6316\u6398\u3001\u9A8C\u8BC1\u3001\u5165\u5E93\u4F53\u7CFB\
  \u3002 \u539F\u59CB\uFF1A\u6BCF\u65E5\u622A\u9762 Spearman\uFF08\u591A\u80A1\u7968\
  \u6A2A\u622A\u9762\u6392\u540D\uFF09 \u9002\u914D\uFF1A\u6EDA\u52A8 21 \u65E5\u65F6\
  \u5E8F Pearson \u76F8\u5173\uFF08\u5355\u6807\u7684\u65F6\u95F4\u5E8F\u5217\uFF09\
  \ \u5224\u5B9A\u6807\u51C6\uFF1A`mean_ic > 0.03`, `ic_positive_ratio > 0.55`, `ic_t"
---

# SOXL 特征挖掘 Skill 适配完成

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## 目标
将 agentic-alpha 的因子挖掘方法论适配到 SOXL 单标的择时场景，建立独立于 RL 训练的特征挖掘、验证、入库体系。

## 核心适配

### 1. 从截面 Rank IC 转时序 IC
- 原始：每日截面 Spearman（多股票横截面排名）
- 适配：滚动 21 日时序 Pearson 相关（单标的时间序列）
- 判定标准：`mean_ic > 0.03`, `ic_positive_ratio > 0.55`, `ic_t > 2.0`

### 2. 新增 signal backtest 层
- 独立的 long/flat 信号回测，不依赖 RL 训练
- 支持阈值扫描（threshold_long × threshold_short）
- 核心指标：signal_accuracy, profit_factor, win_rate

### 3. 从 Rulebase 提取 14 个候选特征
通过 `rulebase_signals` feature pack 集成到 `data.py`：
- `RB_PANIC_SCORE_PROXY` — sigmoid(VIX) × volume + RSI6 调节
- `RB_DAILY_DROP_SEVERITY` — 日跌幅 / vol_coef
- `RB_CUMULATIVE_DROP_SEVERITY` — 累积跌幅 / vol_coef
- `RB_RSI6_OVERSOLD` — RSI6 超卖程度
- `RB_RSI6_OVERBOUGHT` — RSI6 超买程度
- `RB_VOL_REGIME` — 波动率系数
- `RB_BUY_COMPOSITE` — 买入条件综合
- `RB_SELL_COMPOSITE` — 卖出条件综合
- `RB_PROFIT_TAKE_PROXY` — 止盈代理
- `RB_COOLDOWN_PROXY` — 冷却期代理
- `RB_STOP_LOSS_PROXY` — 止损代理
- `RB_VOLUME_STRESS` — 成交量 × 压力得分
- `RB_TREND_REVERSAL` — 趋势反转
- `RB_VIX_TERM_STRESS` — VIX 期限结构压力

## 验证结果

### IC 测试（2019-01-01 ~ 2025-12-31）

| 特征 | 5d IC | 5d pos_ratio | 5d t-stat | 状态 |
|------|-------|-------------|-----------|------|
| RB_PANIC_SCORE_PROXY | +0.086 | 61.5% | +10.38 | Accepted |
| RB_RSI6_OVERSOLD | +0.193 | 80.0% | +30.98 | Accepted |
| RB_BUY_COMPOSITE | +0.054 | 58.0% | +6.43 | Accepted |
| RB_COOLDOWN_PROXY | +0.229 | 73.5% | +18.98 | Accepted |
| RB_VOLUME_STRESS | +0.108 | 63.9% | +14.24 | Accepted |
| RB_DAILY_DROP_SEVERITY | -0.157 | 20.9% | -32.61 | Candidate |
| RB_CUMULATIVE_DROP_SEVERITY | -0.415 | 11.7% | -52.35 | Candidate |
| RB_RSI6_OVERBOUGHT | -0.270 | 14.9% | -40.40 | Candidate |
| RB_SELL_COMPOSITE | -0.394 | 9.1% | -55.29 | Candidate |
| RB_PROFIT_TAKE_PROXY | -0.414 | 10.2% | -52.90 | Candidate |
| RB_STOP_LOSS_PROXY | -0.401 | 11.3% | -50.09 | Candidate |
| RB_TREND_REVERSAL | -0.542 | 5.3% | -74.50 | Candidate |
| RB_VIX_TERM_STRESS | NaN | NaN | NaN | Candidate |

### 信号回测（horizon=5d）

| 特征 | 最佳 threshold | accuracy | PF | win_rate | n_sig |
|------|---------------|----------|----|----------|-------|
| RB_RSI6_OVERSOLD | 0.454 | 59.22% | 1.42 | 68.0% | 78 |
| RB_COOLDOWN_PROXY | 0.266, 0.063 | 58.81% | 1.40 | 55.1% | 32 |
| RB_PANIC_SCORE_PROXY | -0.076, -0.746 | 55.34% | 1.15 | 53.0% | 288 |
| RB_VOLUME_STRESS | -0.006, -0.063 | 54.83% | 1.15 | 53.5% | 285 |
| RB_BUY_COMPOSITE | -0.051, -0.259 | 55.16% | 1.11 | 52.4% | 290 |

## 关键发现

1. **RB_RSI6_OVERSOLD 是最强信号**：5d IC = +0.193，signal accuracy = 59.22%，PF = 1.42
2. **RB_COOLDOWN_PROXY 表现优异**：5d IC = +0.229，accuracy = 58.81%，PF = 1.40
3. **负向特征（卖出/止损类）IC 强烈为负**：这是预期行为，说明这些信号确实预测了未来下跌
4. **简单 long/flat 策略 sharpe 为负**：因为 SOXL 长期上涨，空仓错过涨幅。应关注 signal-level 指标而非策略级 sharpe

## 新增文件

- `.codex/skills/soxl-factor-mining/` — 单特征挖掘 skill
- `.codex/skills/soxl-signal-backtest/` — 信号回测 skill
- `.codex/skills/soxl-factor-research/` — 特征研究 skill
- `data/soxl_factor_library.db` — SQLite 因子库
- `docs/factor_research/` — 研究文档和结果

## 修改文件

- `src/soxl_quant/data.py` — 新增 `rulebase_signals` feature pack（14 个特征）

## 兼容性

- `scripts/check_harness.py`：49 passed, 0 failures
- 新 feature pack 不影响现有 RL 训练（默认不启用）

## 下一步

1. 对负向特征（RB_SELL_COMPOSITE, RB_STOP_LOSS_PROXY 等）单独验证其作为“风险预警”信号的价值
2. 将 Accepted 特征通过 profile 的 `feature_packs` 配置注入 RL 观测空间进行训练测试
3. 建立自动化的因子挖掘流水线（cron / github action）

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
