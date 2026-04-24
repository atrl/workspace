---
date: '2026-04-20'
title: 'User Direction Pivot: Stop Trade Compression, Focus on Drawdown + Kelly'
kind: decision
profiles: []
status: kept
tags:
- checkpoint-selection
- decision
- eval-2022
- kept
- stress-regime
- walk-forward
artifacts: []
workpad: memory/index.json
summary: "Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20\
  \ > \"\u4EA4\u6613\u591A\u4E0D\u5F71\u54CD\uFF0C \u4E3B\u8981\u8981\u63A7\u5236\u56DE\
  \u6D4B\u548C\u51EF\u5229\u671F\u671B \u4F18\u5316\u7684\u65B9\u5F0F\u5E94\u8BE5\u662F\
  \u6A21\u578B \u5956\u52B1 \u7279\u5F81\u7ED3\u5408\u7684\u7ED3\u679C\" Translation:\
  \ \"Many trades don't matter. The main thing is to control drawdown and Kelly expectancy.\
  \ The optimization ap"
---

# User Direction Pivot: Stop Trade Compression, Focus on Drawdown + Kelly

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

Date: 2026-04-20

## User Feedback
> "交易多不影响， 主要要控制回测和凯利期望 优化的方式应该是模型 奖励 特征结合的结果"

Translation: "Many trades don't matter. The main thing is to control drawdown and Kelly expectancy. The optimization approach should be the result of combining model, reward, and features."

## Direction Change
- **STOP**: Trade count compression experiments (v3-v6 line)
- **START**: Holistic optimization focusing on:
  1. **Drawdown control** — reduce max drawdown, especially in stress periods
  2. **Kelly expectancy** — improve risk-adjusted alpha
  3. **Model architecture** — larger/more expressive networks
  4. **Reward function** — better alignment with drawdown + Kelly goals
  5. **Feature engineering** — add risk-predictive features

## Current Best Result (v5)
- Return: 780.7% | Sharpe: 0.93 | MaxDD: 55.9% | Trades: 326
- test_2022: return=-24.66% | maxdd=32.74% | sharpe=-0.915
- test_2023_2024: return=55.05% | sharpe=0.783 | maxdd=32.79%

## Promotion Gate Status (v5)
- max_test_2022_drawdown: 32.74% < 35.0% ✓
- min_test_proxy_sharpe: 0.93 > 0.75 ✓
- min_wf_2023_2024_return: 55.05% > 0.0% ✓
- min_wf_2023_2024_sharpe: 0.783 > 0.0 ✓
- **Trade count gate**: NOT part of promotion criteria — user confirmed irrelevant

v5 PASSES all promotion gates! The only blocker was self-imposed trade count target.

## Next Experiment (v7) Design Principles
1. **Model**: Try larger hidden sizes [256, 128] for more capacity
2. **Reward**: Increase drawdown/downside penalties, add circuit-breaker logic
3. **Position cap**: Tighter drawdown-based caps (start earlier, cap lower)
4. **Features**: Consider adding VIX or other risk indicators
5. **Kelly**: Keep log-alpha scaling, maybe tune alpha_scale

## Key Learning
The user is the ultimate arbiter of what matters. Trade count was a self-imposed constraint, not a user requirement. The actual goals are:
- Maximize risk-adjusted returns (Sharpe)
- Control drawdown
- Pass promotion gates
- Good walk-forward performance

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
