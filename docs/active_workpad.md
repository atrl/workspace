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
- **状态**: Smoke test ✅ 通过，fixed split gate 全通过，walk-forward 运行中 (wf_2020 ✅, wf_2022 ✅, wf_2023_2024 ⏳)

### 方向C: `balanced_curriculum` — 课程学习采样
- **Changed layer**: data / sampling
- **Baseline**: `balanced_rule_residual_v19`
- **Role**: diagnose
- **Hypothesis**: regime_stratified + phase_balanced 采样可改善bear market generalization
- **Accepted trade-off**: 训练时间可能增加，valid score 可能下降
- **Target window**: wf_2022 Sharpe > -0.80, wf_2024 Sharpe > -0.23
- **状态**: Walk-forward ❌ 全部窗口均未通过promotion gate

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

## Walk-Forward 结果

### `balanced_curriculum` (全部完成)

| Window | valid_score | valid_return | valid_sharpe | valid_avg_pos | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|--------------|--------------|---------------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 95.16 | 62.1% | 1.79 | 0.564 | 101.8% | 2.24 | 20.1% | 0.515 | ❌ |
| wf_2022 | 52.28 | 34.3% | 1.04 | 0.657 | -29.0% | -0.68 | 35.2% | 0.334 | ❌ |
| wf_2023_2024 | 59.12 | 40.7% | 1.15 | 0.620 | -16.0% | -0.30 | 35.4% | 0.420 | ❌ |

**失败原因拆解:**
- wf_2020: avg_pos 0.515 > 0.35 gate, 尽管test表现优秀
- wf_2022: test_return -29%, sharpe -0.68 < 0.50 gate, 但avg_pos 0.334通过
- wf_2023_2024: test_return -16% < 0% gate, sharpe -0.30 < 0 gate, avg_pos 0.420 > 0.35

### `balanced_trade_lock` (全部完成)

| Window | valid_score | valid_return | valid_sharpe | valid_avg_pos | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|--------------|--------------|---------------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 96.17 | 62.3% | 1.83 | 0.555 | 90.2% | 2.09 | 24.2% | 0.497 | ❌ |
| wf_2022 | 64.14 | 42.3% | 1.23 | 0.634 | -29.8% | -1.08 | 32.1% | 0.280 | ❌ |
| wf_2023_2024 | 57.72 | 38.9% | 1.14 | 0.560 | -10.2% | -0.12 | 33.9% | 0.412 | ❌ |

---

## 关键发现与诊断

1. **Curriculum 采样并未改善walk-forward generalization**
   - wf_2022和wf_2023_2024均出现负收益，与fixed split的promising结果形成巨大反差
   - Fixed split test_2023_2024: +44.9% → Walk-forward: -16.0% (差距60pp!)
   - 说明课程采样可能overfit到valid set的regime分布

2. **Trade lock表现与curriculum惊人相似**
   - wf_2022: -29.8% vs curriculum -29.0%, 几乎相同
   - 两个完全不同的干预手段(different changed layers)却产生几乎相同的test结果
   - 这强烈暗示问题不在sampling或execution，而在更底层的model capacity / feature set

3. **所有window的valid score都很高(52-96)，但test表现分化**
   - 说明validation set不能有效预测test set性能
   - 可能存在validation leakage或validation set与test set分布差异过大

4. **wf_2020的avg_pos普遍过高(0.497-0.515)**
   - 即使是bull market window，模型也几乎全仓
   - 硬底仓core50的设计可能反而更合理（强制50%保留）

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
- [x] 10. Walk-forward: `balanced_curriculum` — 已完成，全部窗口未通过
- [x] 11. Walk-forward: `balanced_trade_lock` — 已完成，全部窗口未通过
- [x] 12. Walk-forward: `balanced_core50`（调参后）— 优先级降低，因其他两个方向已揭示walk-forward根本性困难
- [x] 13. 分析 walk-forward 失败根因 — 完成，见下方诊断
- [x] 14. Walk-forward: `balanced_core50` wf_2020 [auto] — 已完成 ( return=109.8% Sharpe=1.86 MaxDD=23.0% avg_pos=0.585 ❌ )
- [x] 15. Walk-forward: `balanced_core50` wf_2022 [auto] — 已完成 ( return=-41.6% Sharpe=-1.28 MaxDD=43.9% avg_pos=0.360 ❌ )
- [ ] 16. Walk-forward: `balanced_core50` wf_2023_2024 [auto] — 运行中 (第一次timeout后已重启，2h timeout)
- [ ] 17. 分析三个方向walk-forward结果，决定下一步 [auto]
- [ ] 18. 创建 `balanced_core50_v2` profile (底仓降至40%) [auto]
- [ ] 19. Smoke test: `balanced_core50_v2` [auto]
- [ ] 20. 创建 `balanced_rule_residual_v19_reg_v22` profile (P1方向D) [auto]
- [ ] 21. Smoke test: `balanced_rule_residual_v19_reg_v22` [auto]

## 证据 / Blockers

- 方向梳理文档: `runtime/model/exploration_roadmap_2026-04-29.md`
- 自动化脚本: `scripts/kimi_auto_experiment.sh`
- Prompt模板: `.codex/prompts/auto_experiment.md`
- `balanced_curriculum` profile: `runtime/model/card/balanced_curriculum/profile.yaml`
- `balanced_curriculum` summary: `runtime/model/card/balanced_curriculum/soxl_balanced_curriculum_summary.json`
- `balanced_curriculum` wf summaries: `runtime/model/card/balanced_curriculum/soxl_balanced_curriculum_wf_20{20,22,23_2024}_summary.json`
- `balanced_trade_lock` wf summaries: `runtime/model/card/balanced_trade_lock/soxl_balanced_trade_lock_wf_20{20,22}_summary.json`

## 决策

- **P0三个方向并行**，不互相等待
- `balanced_curriculum` walk-forward 失败，需要诊断是否继续迭代
- `balanced_trade_lock` walk-forward 即将完成，等待wf_2023_2024结果
- `balanced_core50` 需要调参优化 avg_pos 后才能 walk-forward
- **关键问题**: valid score高但test差，可能表明validation set设计有根本性问题

---

## 历史档案

- `balanced_macro_v1` profile: `runtime/model/card/balanced_macro_v1/profile.yaml`
- `balanced` baseline (seed 42): `runtime/model/card/balanced/soxl_balanced_summary.json`
- `balanced_macro_v1` summary: `runtime/model/card/balanced_macro_v1/soxl_balanced_macro_v1_summary.json`
- IC 分析脚本: `scripts/compute_macro_ic.py`
- IC 结果: `runtime/model/card/macro_ic_analysis/ic_results.json`
- IC 摘要: `runtime/model/card/macro_ic_analysis/ic_summary.txt`
