---
date: '2026-04-14'
title: "P0 \u4F18\u5316\u7B2C\u4E00\u6B21\u5C1D\u8BD5\u62A5\u544A"
kind: review
profiles:
- soxl
status: reference
tags:
- reference
- review
- soxl
artifacts: []
workpad: memory/index.json
summary: "Legacy memory note normalized into the repo-wide schema. \u5728 reward \u51FD\
  \u6570\u4E2D\u52A0\u5165\u8E0F\u7A7A\u60E9\u7F5A\uFF08opportunity-cost penalty\uFF09\
  \uFF0C\u5E76\u628A `drawdown_position_cap` \u7684 `min_cap` \u4ECE\u56FA\u5B9A 0.40\
  \ \u6539\u4E3A\u52A8\u6001\u8C03\u6574\uFF1A \u725B\u5E02\uFF08realized_vol < 0.40\
  \ \u4E14 anchor_gap > 0.01\uFF09\uFF1Amin_cap \u63D0\u5347\u5230 0.65 \u9AD8\u98CE\
  \u9669\u5E02\uFF08regime_risk > 0."
---

# P0 优化第一次尝试报告

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## 实验内容
在 reward 函数中加入踏空惩罚（opportunity-cost penalty），并把 `drawdown_position_cap` 的 `min_cap` 从固定 0.40 改为动态调整：
- 牛市（realized_vol < 0.40 且 anchor_gap > 0.01）：min_cap 提升到 0.65
- 高风险市（regime_risk > 0.65）：min_cap 收紧到 0.30

代码改动文件：
- `src/soxl_quant/rl.py`
- `src/soxl_quant/settings.py`
- `configs/soxl/*.yaml`

## 训练结果（SOXL 域内）

| 指标 | 旧模型 | 新模型（P0） | 变化 |
|------|--------|--------------|------|
| return | 368.68% | 372.1% | +3.4% |
| sharpe | 0.805 | 0.82 | +0.015 |
| max_dd | 51.57% | 49.9% | -1.67% |
| trades | 894 | 900 | +6 |

SOXL 域内表现基本稳定，说明训练流程没有被破坏。

## 泛化结果（跨 ETF 零样本）

| ETF | 旧模型 | 新模型（P0） | 变化 |
|-----|--------|--------------|------|
| SOXL | 374.85% | 373.13% | -1.72% |
| TQQQ | 230.51% | 224.71% | **-5.80%** |
| UPRO | 176.53% | 167.85% | **-8.68%** |
| SPXL | 172.53% | 167.70% | **-4.83%** |
| TECL | 223.01% | 223.45% | +0.44% |

## 结论与教训

1. **泛化收益反而下降了**。说明当前 P0 参数组合不够理想，存在以下可能原因：
   - `opportunity_cost_penalty=0.10` 可能过重，导致 actor 在趋势并不十分明确时也被迫加仓，积累了不必要的回调风险。
   - `bull_min_cap=0.65` 把熊市初期的最低仓位抬得太高，在市场转势时无法快速减仓，侵蚀了收益。
   - 踏空惩罚与动态 cap 的交互产生了“追高”行为，而在宽基指数（UPRO/SPXL）中这种追高并不划算。

2. **回撤控制仍然优秀**：所有 ETF 的最大回撤都维持在 35% ~ 50% 区间，与之前基本持平。这说明风险预算框架本身没有被破坏。

3. **SOXL 和 TECL 几乎不受影响**：可能是因为这两只 ETF 的行业属性与训练域（SOXL）更接近，actor 的动作分布已经天然适应。

## 下一轮调参方向

1. **降低踏空惩罚权重**：把 `opportunity_cost_penalty` 从 0.10 降到 0.04 ~ 0.06，只在**强趋势**（anchor_gap > 0.05）时触发。
2. **更保守的动态 cap**：
   - `bull_min_cap` 从 0.65 降到 0.55
   - 增加条件：不仅要求 `anchor_gap > 0.01`，还要求 `trend_gap_21 > 0.02`，避免在震荡市中误识别为牛市。
3. **分品种调参**：宽基指数（UPRO/SPXL）的波动特征与科技股不同，后续做多资产联合训练时，应该让模型自己学习不同品种的最佳 cap，而不是统一的硬编码规则。

## 代码提交
已提交到主分支：`5a53248 feat(P0): adaptive position cap + opportunity-cost penalty, fix regressions`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
