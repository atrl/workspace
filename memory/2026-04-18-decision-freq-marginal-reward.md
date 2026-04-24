---
date: '2026-04-18'
title: "2026-04-18: Decision Frequency=3 \u4E2D\u7B49\u8BAD\u7EC3\u9A8C\u8BC1 + Period-Aware\
  \ Marginal Reward"
kind: decision
profiles:
- balanced
- balanced_decision_freq_quick
- smoke
- smoke_decision_freq
- smoke_marginal_freq
- soxl
status: reference
tags:
- balanced
- balanced_decision_freq_quick
- decision
- eval-2022
- reference
- smoke
- smoke_decision_freq
- smoke_marginal_freq
- soxl
artifacts: []
workpad: memory/index.json
summary: "Legacy memory note normalized into the repo-wide schema. 1. \u9A8C\u8BC1\
  \ decision_frequency=3 \u5728 20-iters \u4E2D\u7B49\u8BAD\u7EC3\u4E0B\u7684\u7A33\
  \u5B9A\u6027 2. \u5B9E\u73B0\u5E76\u9A8C\u8BC1 Period-Aware Marginal Reward\uFF08\
  \u8DE8\u5468\u671F alpha reward\uFF09 | Metric | Value | vs Canonical Balanced |\
  \ |--------|-------|------------------"
---

# 2026-04-18: Decision Frequency=3 中等训练验证 + Period-Aware Marginal Reward

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## 实验目标
1. 验证 decision_frequency=3 在 20-iters 中等训练下的稳定性
2. 实现并验证 Period-Aware Marginal Reward（跨周期 alpha reward）

## 结果汇总

### Freq=3 20-iters 中等训练 (`balanced_decision_freq_quick`)
| Metric | Value | vs Canonical Balanced |
|--------|-------|----------------------|
| Test return | 350.7% | ❌ -118pp (468.9%) |
| Sharpe | 0.77 | ~持平 (0.86) |
| MaxDD | 56.3% | ❌ +6pp (50.2%) |
| Trades | 319 | ✅ -681 (1000) |

Slice 对比：
| Year | Freq=3 (20it) | Canonical Balanced |
|------|---------------|-------------------|
| 2020 | +99.7% | +97.0% |
| **2022** | **-36.6%** ❌ | **-25.8%** |
| 2023-24 | +58.3% | +43.5% |

**核心发现**：freq=3 在 2022 年压力年灾难性表现（-36.6% vs -25.8%）。
限制调仓频率后，模型无法及时应对快速反转，回撤放大。
但在趋势年（2020、2023-2024）表现更好——freq=3 实际上提高了 beta 暴露，
改变了风险收益特征，而非提升 alpha。

### Period-Aware Marginal Reward smoke test (`smoke_marginal_freq`)
- Return: 457.5% vs baseline smoke 372.7% → +85pp
- 训练日志 episode reward 为负值，sparse reward 在 2-iters 下学习困难
- 实现保留在代码库中，但需更多训练量验证

## 关键修复
**CLI Bug**: `--smoke` 标志原来会强制覆盖任何 profile 为 `smoke`，
导致所有自定义 smoke profile（smoke_decision_freq, smoke_marginal_freq 等）
完全不生效。已修复：现在 `--smoke` 只控制训练规模参数，保留原 profile 的
reward/freq 设置。

## 结论与下一步
1. **freq=3 不是银弹**。它在趋势年表现好，但在压力年回撤更大。
   对于 SOXL 这种高波动资产，日频调仓可能仍是必要的。
2. **最高 ROI 方向**：回到 freq=1，但用 Period-Aware Marginal Reward
   改进 reward 设计（在日频下需要放大 signal）。
3. **Marginal Reward 的 theory 仍然 valid**，但需更高 scale 和更长训练。

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
