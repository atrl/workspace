---
date: '2026-04-24'
title: 'Next Actions'
kind: ops
status: active
direction: harness
---

# Next Actions

## 本次已完成（代码整理迭代）

- ✅ 删除 codex_exec_plans/，迁移至 docs/active_workpad.md
- ✅ memory/ 已迁移到 runtime/<direction>/knowledge.md 并删除
- ✅ experiments/ 精简（删除 archive，合并重复分支）
- ✅ experiments/ 合并到 runtime/：每个实验包精确链接到对应 model card，knowledge.md 新增实验包索引表
- ✅ risk_budget_usage_v1 card 添加完整 Mermaid 架构图（观测层→Actor→ModeRouter→BudgetUsage→PositionCalc→Execution）
- ✅ 创建 skill `.codex/skills/risk-budget-architecture/SKILL.md`，记录架构组件、历史表现、待挖掘方向
- ✅ risk_budget_usage_v2 扩展实现：
  - Mode-Conditional Reward Shaping（config.py + reward.py）
  - Dynamic Band Scaling（config.py + hierarchical_composer.py）
  - 新建实验包 `experiments/candidate/risk_budget_usage_v2/`（notes.md + expectations.md + profile.yaml）
  - 新建 runtime card `runtime/model/card/risk_budget_usage_v2/card.md`
- ✅ risk_budget_usage_v3 扩展实现：
  - Mode-Conditioned Actor Input（config.py + observation.py + environment.py + backtest.py + selection.py）
  - 新建实验包 `experiments/candidate/risk_budget_usage_v3/`（notes.md + expectations.md + profile.yaml + card.md）
- ✅ 实验卡片共置重组：将 runtime/model/card/ 下有对应实验包的 card.md 移动到 experiments/ 下
  - 12 个卡片移动，14 个跳过（合并实验共享或纯 artifact summary）
  - runtime/model/knowledge.md 链接更新指向 experiments/
  - build_runtime_index.py 支持扫描 experiments/ 目录
- ✅ **risk_budget_usage_v2 训练完成** (seed=42)
  - test_proxy: Sharpe=0.884, MaxDD=20.6%, avg_pos=0.180
  - test_2022: MaxDD=7.4%, avg_pos=0.070（显著改善但过度保守）
  - Promotion: blocked (Sharpe < 0.95)
  - 诊断: flat_wait_pos_penalty 太强，需调参
- ✅ **risk_budget_usage_v3 训练完成** (seed=42) — 🎉 **BREAKTHROUGH**
  - test_proxy: Sharpe=**1.074**, MaxDD=21.4%, avg_pos=0.208
  - test_2022: MaxDD=**12.0%**, avg_pos=**0.143**（通过 gate!）
  - **Fixed Split Gate: PASSED** (Sharpe>0.95, MaxDD<20%, avg_pos<0.15)
  - 首次突破: mode-conditioned input 让 Actor 直接"看到"mode，效果显著优于 reward shaping
- ✅ runtime/model/card/ 按 family 拆分（canonical/balanced/smoke/risk_budget/other）
- ✅ skills 重命名（soxl-factor-* → leveraged-etf-factor-*）+ _index.json
- ✅ 知识卡片重组：每个实验独立文件夹 + 统一格式 + 结论
- ✅ 网络卡片按架构合并（GRU/Attention/TiDE/MLP），含结构代码和参数表
- ✅ 奖励卡片按类型合并（Base Log/Kelly/Imitation），含参数和结果表
- ✅ 因子卡片添加论文引用（Amihud 2002, Bollerslev et al. 2009 等）
- ✅ 清理：空 backtest CSV / log/ / wf_workers / diagnosis temp files
- ✅ 修复 .gitignore（包含 runtime/reward/ multi_asset/ harness/）
- ✅ check_harness.py: 59 passed, 0 failures

---

## 当前待验证假设

### 0. Crypto-style 单资产因子参考（factor）— 本次完成

**问题**: 用户提出 SOXL 个股/单ETF择时更像币圈单一资产量化，询问可参考的因子。

**本次执行**:
- 新增 `SIGNED_TE_REVERSION_*` 到 `transition_signals` feature pack：
  `-zscore(rolling_sum(SOXL_return - 3 * SOXX_return, window), 252d)`，窗口为 5d / 10d / 21d。
- 新增 `scripts/evaluate_signed_te_reversion.py`，输出 IC、standalone backtest、core correlation evidence。
- 新增因子卡片 `runtime/factor/card/signed_te_reversion/card.md`。

**结果**:
- Best IC: `SIGNED_TE_REVERSION_21`, h=20, IC mean=+0.014, positive ratio=0.516 → 未过 IC gate。
- Best standalone: `SIGNED_TE_REVERSION_10`, threshold=0.5, Sharpe=+0.561, PF=1.228, accuracy=0.546 → 有正收益但方向稳定性不足。
- Correlation: 10d max core rho=0.577 通过；5d max core rho=0.811 未过相关性预算。

**结论**: 不加入 core RL schema。若继续，只围绕 `SIGNED_TE_REVERSION_10` 做 regime-gated ablation，不推广 5/10/21 全家族。

### 0b. Crypto-style wick rejection 因子（factor）— 本次完成

**本次执行**:
- 新增 `INTRADAY_REJECTION_EVENT` 到 `transition_signals` feature pack：
  `lower_wick_ratio * close_location * positive_intraday_range_zscore * (1 + positive_relative_volume_zscore)`。
- 新增 `scripts/evaluate_intraday_rejection.py`，输出 IC、standalone backtest、core correlation evidence。
- 新增因子卡片 `runtime/factor/card/intraday_rejection_event/card.md`。

**结果**:
- Best IC: h=20, IC mean=+0.0157, positive ratio=0.215 → 未过 IC gate。
- Best standalone: threshold=0.0, Sharpe=+0.261, PF=1.078, accuracy=0.529, total_return=-53.9% → 未过 backtest gate。
- Correlation: max core rho=0.444 vs `volume_shock` → 相关性预算通过。

**结论**: 裸 wick rejection 不适合 SOXL daily timing；后续除非加 stress-release / VIX divergence / post-crash compression gate，否则不要重复裸测。

### 1. 2022 Stress Window 过仓问题（P0）

**问题**: canonical `balanced` 在 test_2022 的 avg_pos ~0.38，promotion gate 要求 < 0.35。
**已尝试**: floor40 overlay, mode routing, penalty tuning, phase-balanced sampling — 均未同时满足 return + MaxDD + avg_pos。
**待尝试方向**:
- a) **Stress-specific reward shaping**: stress 态下仓位惩罚系数动态增大
- b) **Hard floor 提升**: 从 0.20 提升到 0.30-0.35，限制最低仓位
- c) **Regime-conditioned action space**: stress 态时减少 action levels（如只保留 [0, 0.2, 0.4]）
- d) **Stress episode 过采样**: 在 training batch 中增加 2022 类 episode 的比例

### 2. Multi-Asset 泛化（P1）— 部分完成

**已完成**:
- ✅ **Zero-shot migration v1**: SOXL actor 在 TQQQ/UPRO/SPXL/TECL 上回测
  - TQQQ Sharpe=0.867（甚至优于 SOXL 的 0.846）
  - UPRO/SPXL Sharpe~0.83-0.84，MaxDD 更低（~38%）
  - TECL Sharpe=0.716（相对最弱）
- 结论: 零样本迁移可行，核心特征具有跨资产通用性

**待尝试方向**:
- a) **联合训练**: shared backbone + asset-specific head，对比零样本 gap
- b) **Volatility normalization**: 将 realized_vol 按资产历史中位数标准化
- c) **分析 TECL 弱势原因**: 成份股差异 vs 特征适配度

---

## 下一步建议（选一个推进）

| 优先级 | 方向 | 假设 | 验证方式 | 预期时间 |
|--------|------|------|----------|----------|
| P0 | 2022 avg_pos 修复 | hard floor 0.30 或 stress action space 限制 | 修改 profile → 训练 → test_2022 gate | 2-3h |
| P1 | TQQQ 迁移测试 | SOXL actor 在 TQQQ 上直接回测 | 使用现有 actor + TQQQ qlib store | 30min |
| P2 | 自动化卡片检查 | 脚本验证每个 card.md 格式完整性 | 新增 check_cards.py | 1h |
| P2 | Skill 过期清理 | 删除 30 天未使用的 skill | 运行 check_skills.py + 手动审查 | 30min |

**推荐**: 先跑 P1（TQQQ 迁移测试），30 分钟就能得到多资产 gap 的量化数据，为后续联合训练提供依据。如果 2022 问题更紧急，则选 P0。
