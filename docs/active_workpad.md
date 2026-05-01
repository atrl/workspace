---
date: '2026-05-01'
title: 'P0/P1 Walk-forward 全部完成 — 系统性失败诊断'
kind: experiment
status: active
direction: model
---

# 当前活跃任务: Walk-forward 系统性失败 — 根本性诊断

> 日期: 2026-05-01
> 来源: `runtime/model/exploration_roadmap_2026-04-29.md`

## 重大突破 🎯

**两个候选首次通过 fixed split 全部 promotion checks！**

| Profile | Seed | test_2022 MaxDD | test_2022 avg_pos | test_proxy Sharpe | fixed_split eligible | Smoke Test |
|---------|------|-----------------|-------------------|-------------------|----------------------|------------|
| `balanced_trade_lock` | **123** | **29.9%** ✅ | **0.269** ✅ | **1.007** ✅ | **TRUE** | return=776.4% Sharpe=0.99 MaxDD=50.6% trades=315 |
| `balanced_rule_residual_v19_reg_v22` | **42** | **32.9%** ✅ | **0.320** ✅ | **1.022** ✅ | **TRUE** | return=861.8% Sharpe=1.00 MaxDD=51.5% trades=326 |
| `balanced_core50_v2` | 42 | 40.3% ✅ | 0.365 ❌ | 0.819 ✅ | FALSE | return=481.1% Sharpe=0.81 MaxDD=57.3% trades=306 |

**关键洞察:**
- `trade_lock` **seed=123** 相比 seed=42 (MaxDD=58.4%)，test_2022 MaxDD 从 58.4% → 29.9%（-28.5pp），**seed 对 risk profile 影响巨大**
- `v22` (VIX期限结构特征) 同样通过全部 checks，说明新特征可能有效
- 两者均仅因缺少 walk-forward 而被标记为 `eligible=False`，一旦 walk-forward 通过即可 promotion

---

## 实验计划

### 方向A: `balanced_core50` / `balanced_core50_v2` — 硬底仓
- **状态**: `core50` walk-forward 运行中 (wf_2020 worker PID 619365 进行中)；`core50_v2` smoke test 完成，avg_pos=0.365 > 0.35 仍失败
- **结论**: 单纯降低底仓（50%→40%）对 MaxDD 改善有限（-2.5pp），且收益大幅下降（-90pp）。方向基本证伪。

### 方向B: `balanced_trade_lock` — 交易频率约束 + 多种子验证
- **状态**: seed=42 walk-forward 全部失败；**seed=123 smoke test 通过全部 fixed split checks**
- **下一步**: 启动 seed=123 的 walk-forward 验证

### 方向C: `balanced_curriculum` — 课程学习采样
- **状态**: Walk-forward 全部窗口未通过，已归档

### 方向D: `balanced_rule_residual_v19_reg_v22` — VIX期限结构特征 (P1)
- **状态**: **Smoke test 通过全部 fixed split checks**
- **特征**: 新增 5 个 VIX 期限结构特征 (vvix_zscore_252, vix_term_slope, vix_short_ratio, skew_zscore_252, vix_term_slope_velocity_5)
- **下一步**: 启动 walk-forward 验证

---

## Smoke Test 结果汇总（最新）

| Profile | Seed | Valid Score | test_proxy Return | test_proxy Sharpe | test_2022 MaxDD | test_2022 avg_pos | fixed_split |
|---------|------|-------------|-------------------|-------------------|-----------------|-------------------|-------------|
| v19 baseline | 42 | ~82 | 605% | 0.89 | 38.5% | 0.328 | PASSED |
| trade_lock | 42 | 79.69 | 1012.5% | 0.98 | 58.4% | 0.315 | ❌ MaxDD |
| **trade_lock** | **123** | **91.64** | **776.4%** | **0.99** | **29.9%** | **0.269** | **✅ ALL** |
| **v22** | **42** | **88.93** | **861.8%** | **1.00** | **32.9%** | **0.320** | **✅ ALL** |
| core50_v2 | 42 | — | 481.1% | 0.81 | 40.3% | 0.365 ❌ | ❌ avg_pos |

---

## Walk-Forward 历史结果

### `balanced_curriculum` (seed=42, 全部完成) — 已归档

| Window | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 101.8% | 2.24 | 20.1% | 0.515 | ❌ avg_pos |
| wf_2022 | -29.0% | -0.68 | 35.2% | 0.334 | ❌ return/sharpe |
| wf_2023_2024 | -16.0% | -0.30 | 35.4% | 0.420 | ❌ all |

### `balanced_trade_lock` (seed=42, 全部完成)

| Window | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 90.2% | 2.09 | 24.2% | 0.497 | ❌ avg_pos |
| wf_2022 | -29.8% | -1.08 | 32.1% | 0.280 | ❌ return/sharpe |
| wf_2023_2024 | -10.2% | -0.12 | 33.9% | 0.412 | ❌ all |

### `balanced_core50` (seed=42, 已完成)

| Window | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 115.0% | 2.05 | 20.5% | 0.567 | ❌ avg_pos |
| wf_2022 | -41.1% | -1.28 | 43.4% | 0.362 | ❌ all |
| wf_2023_2024 | -14.2% | -0.24 | 32.7% | — | ❌ return |

### `balanced_rule_residual_v19_reg_v22` (seed=42, 已完成)

| Window | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 97.7% | 1.96 | 19.3% | — | ✅ |
| wf_2022 | -30.3% | -0.92 | 35.3% | — | ❌ return/sharpe |
| wf_2023_2024 | -15.3% | -0.21 | 35.4% | — | ❌ return/sharpe |

### `balanced_trade_lock` (seed=123, 已完成)

| Window | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 96.1% | 1.74 | 25.0% | — | ✅ |
| wf_2022 | -28.7% | -0.89 | 33.8% | — | ❌ return/sharpe |
| wf_2023_2024 | -16.0% | -0.29 | 34.5% | — | ❌ return/sharpe |

---

## 关键发现与诊断

1. **Seed 对 risk profile 影响巨大**
   - `trade_lock` seed=42: test_2022 MaxDD=58.4%，seed=123: 29.9%
   - 同样的超参数，不同的随机初始化导致完全不同的风险控制行为
   - 这暗示模型对 initialization 敏感，或存在多个局部最优解

2. **VIX 期限结构特征可能有效**
   - `v22` (新增 5 个 VIX 特征) 首次通过全部 fixed split checks
   - 特征从 59 → 64，外部数据文件确认存在 (VIX3M.csv, VIX9D.csv, VVIX.csv, SKEW.csv)
   - 需要 walk-forward 验证才能确认不是 overfit

3. **core50 方向基本证伪**
   - core50 (50%底仓): MaxDD=59.8%, avg_pos=0.366
   - core50_v2 (40%底仓): MaxDD=57.3%, avg_pos=0.365
   - 降低底仓 10pp 仅改善 MaxDD 2.5pp，收益却下降 90pp
   - 硬底仓不是解决 MaxDD 的有效手段

4. **Walk-forward 根本性困难 — 系统性失败确认**
   - **所有三个方向（core50, trade_lock, v22）的 walk-forward 均失败**
   - 共同模式: wf_2020 表现优秀 (+96~115%), wf_2022 亏损 (-28~41%), wf_2023_2024 亏损 (-14~16%)
   - **关键发现**: wf_2022 和 wf_2023_2024 均为负收益，说明模型无法泛化到 2022+ 的市场环境
   - 这不是超参数问题，而是 **model capacity / validation design / regime adaptation** 的根本性缺陷
   - fixed split 通过但 walk-forward 失败，说明 fixed split 的 test_2022 并非真正 OOS

5. **Smoke test 种子稳健性验证完成**
   - 4个额外 smoke test 已完成（存在 model file contention，结果仅供参考）
   - trade_lock seed=456: return=1012.5% Sharpe=0.98 MaxDD=58.4% → blocked
   - trade_lock seed=789: return=776.4% Sharpe=0.99 MaxDD=50.6% → blocked
   - v22 seed=123: return=1099.3% Sharpe=1.00 MaxDD=59.6% → blocked
   - v22 seed=456: return=795.5% Sharpe=0.96 MaxDD=56.0% → blocked
   - 观察: 不同 seed 确实产生不同 risk profile，但并发写入同一 model 文件导致结果不可靠

---

## Next Action

- [x] 1-15. 历史任务已完成
- [x] 16. Walk-forward: `balanced_core50` wf_2023_2024 — **已完成，wf_2023_2024 return=-14.2% 失败**
- [x] 17. 分析三个方向walk-forward结果 — 完成
- [x] 18. 创建 `balanced_core50_v2` profile — 已完成
- [x] 19. Smoke test: `balanced_core50_v2` — 已完成，avg_pos=0.365 仍失败
- [x] 20. 创建 `balanced_rule_residual_v19_reg_v22` profile — 已完成
- [x] 21. Smoke test: `balanced_rule_residual_v19_reg_v22` — **已完成，fixed_split eligible=TRUE!**
- [x] 22. Smoke test: `balanced_trade_lock` seed=123 — **已完成，fixed_split eligible=TRUE!**
- [x] 23. Walk-forward: `balanced_trade_lock` seed=123 — **已完成，wf_2023_2024 return=-16.0% 失败**
- [x] 24. Walk-forward: `balanced_rule_residual_v19_reg_v22` seed=42 — **已完成，wf_2023_2024 return=-15.3% 失败**
- [x] 25. 等待 core50 walk-forward 完成 — **已完成**
- [x] 26. 等待 4个额外 smoke test 完成 (trade_lock seeds 456/789, v22 seeds 123/456) — **已完成，全部 blocked**
- [ ] 27. **根本性诊断**: 所有 walk-forward 在 2022+ 均失败，需分析原因并制定下一步策略

---

## 证据 / Blockers

- `balanced_trade_lock` seed=123 summary: `runtime/model/card/balanced_trade_lock/soxl_balanced_trade_lock_summary.json`
- `balanced_rule_residual_v19_reg_v22` summary: `runtime/model/card/balanced_rule_residual_v19_reg_v22/soxl_balanced_rule_residual_v19_reg_v22_summary.json`
- `balanced_core50_v2` summary: `runtime/model/card/balanced_core50_v2/soxl_balanced_core50_v2_summary.json`
- `balanced_core50` walk-forward log: background task `bash-4ljlyb37`

## 决策

- **`balanced_core50` / `core50_v2` 方向降级为 P2**，硬底仓不是有效解决方案
- **`balanced_curriculum` 已归档**，不再投入资源
- **Walk-forward 系统性失败 — 所有方向在 2022+ 市场环境均失效**
- **P0 优先级转移**: 从 "超参数/特征搜索" → **"根本性诊断: 为什么所有模型在 2022+ 失败"**
- **假设**: 2022-2024 市场 regime 与训练期（2010-2021）存在结构性差异，模型缺乏 regime adaptation 能力
- **下一步**: 分析失败原因（regime shift? overfitting? action space? reward design?），制定针对性修复策略
