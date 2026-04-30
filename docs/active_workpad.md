---
date: '2026-04-30'
title: 'P0 实验方向并行推进 — core50, trade_lock, curriculum'
kind: experiment
status: active
direction: model
---

# 当前活跃任务: P0 三个方向并行推进

> 日期: 2026-04-30
> 来源: `runtime/model/exploration_roadmap_2026-04-29.md`

## 背景

Macro特征实验已完成并确认放弃。基于最新方向梳理，P0三个方向（硬底仓、交易锁、课程学习）预期ROI最高且互不冲突，决定并行启动。

## 实验计划

### 方向A: `balanced_core50` — 硬底仓50% + RL战术叠加
- **Changed layer**: execution overlay / position mapping
- **Baseline**: `balanced_rule_residual_v19`
- **Role**: diagnose
- **Hypothesis**: 强制底仓50%可改善recovery期踏空问题
- **Accepted trade-off**: 2022 MaxDD可能恶化至45%
- **Target window**: test_2023_2024 return > 80%
- **状态**: Smoke test ✅ 通过，promotion blocked（avg_pos=0.366 > 0.35，预期内）

### 方向B: `balanced_trade_lock` — 交易频率硬约束
- **Changed layer**: execution overlay
- **Baseline**: `balanced_rule_residual_v19`
- **Role**: diagnose
- **Hypothesis**: threshold=0.05 + trade_lock=5 + min_turnover=0.05 可压缩摩擦成本
- **Accepted trade-off**: 危机反应可能变慢
- **Target window**: test_proxy trades < 400, Sharpe > 0.80
- **状态**: Smoke test ✅ 通过，fixed split gate 全通过，待 walk-forward

### 方向C: `balanced_curriculum` — 课程学习采样
- **Changed layer**: data / sampling
- **Baseline**: `balanced_rule_residual_v19`
- **Role**: diagnose
- **Hypothesis**: regime_stratified + phase_balanced 采样可改善bear market generalization
- **Accepted trade-off**: 训练时间可能增加，valid score 可能下降
- **Target window**: wf_2022 Sharpe > -0.80, wf_2024 Sharpe > -0.23
- **状态**: Smoke test ✅ 通过，fixed split gate 全通过，待 walk-forward

---

## Smoke Test 结果汇总

| 指标 | v19 baseline | core50 | trade_lock | curriculum |
|------|-------------|--------|------------|------------|
| Valid score | ~82 | 72.85 | 79.69 | **86.36** |
| test_proxy Return | 605% | 574% | 695% | **702%** |
| test_proxy Sharpe | 0.89 | 0.85 | 0.92 | **0.97** |
| test_2022 Return | -36.7% | -34.5% | -37.5% | **-28.9%** |
| test_2022 avg_pos | 0.328 | 0.366 ❌ | 0.315 ✅ | **0.283** ✅ |
| test_2023_2024 Return | 62.4% | 39.1% | 54.7% | 44.9% |
| Fixed split gate | PASSED | BLOCKED | PASSED | PASSED |

### curriculum 关键发现
- **Bear market 防御显著改善**：test_2022 return -28.9% vs v19 -36.7%，avg_pos 0.283（更防御）
- **Recovery 期表现下降**：test_2023_2024 return 44.9% vs v19 62.4%
- 呈现清晰的 trade-off：防御性提升 vs recovery 踏空

---

## Next Action

- [x] 1. 完成方向梳理文档 `runtime/model/exploration_roadmap_2026-04-29.md`
- [x] 2. 搭建自动化实验基础设施 (scripts, systemd, prompt)
- [x] 3. 创建 `balanced_core50` profile YAML [auto] — 已完成
- [x] 4. 创建 `balanced_trade_lock` profile YAML [auto] — 已完成
- [x] 5. 在 `src/soxl_quant/policy/positioning.py` 中新增 `core50` position mapping [auto] — 已完成
- [x] 6. Smoke test: `balanced_core50` 快速训练验证代码正确性 [auto] — 已完成
- [x] 7. Smoke test: `balanced_trade_lock` 回测验证 [auto] — 已完成
- [x] 8. 创建 `balanced_curriculum` profile YAML — 已完成
- [x] 9. Smoke test: `balanced_curriculum` 快速训练验证 — 已完成
- [ ] 10. Walk-forward: `balanced_trade_lock`
- [ ] 11. Walk-forward: `balanced_core50`（调参后）
- [ ] 12. Walk-forward: `balanced_curriculum`

## 证据 / Blockers

- 方向梳理文档: `runtime/model/exploration_roadmap_2026-04-29.md`
- 自动化脚本: `scripts/kimi_auto_experiment.sh`
- Prompt模板: `.codex/prompts/auto_experiment.md`
- `balanced_curriculum` profile: `runtime/model/card/balanced_curriculum/profile.yaml`
- `balanced_curriculum` summary: `runtime/model/card/balanced_curriculum/soxl_balanced_curriculum_summary.json`

## 决策

- **P0三个方向并行**，不互相等待
- **优先完成 walk-forward 验证**
- `balanced_curriculum` smoke test 结果 promising，优先安排 walk-forward
- `balanced_trade_lock` fixed split 表现稳定，次优先 walk-forward
- `balanced_core50` 需要调参优化 avg_pos 后才能 walk-forward

---

## 历史档案

- `balanced_macro_v1` profile: `runtime/model/card/balanced_macro_v1/profile.yaml`
- `balanced` baseline (seed 42): `runtime/model/card/balanced/soxl_balanced_summary.json`
- `balanced_macro_v1` summary: `runtime/model/card/balanced_macro_v1/soxl_balanced_macro_v1_summary.json`
- IC 分析脚本: `scripts/compute_macro_ic.py`
- IC 结果: `runtime/model/card/macro_ic_analysis/ic_results.json`
- IC 摘要: `runtime/model/card/macro_ic_analysis/ic_summary.txt`
