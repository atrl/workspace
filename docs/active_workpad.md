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
- ✅ memory/ 整合到 runtime/<direction>/knowledge.md
- ✅ experiments/ 精简（删除 archive，合并重复分支）
- ✅ artifacts/ 按 family 拆分（canonical/balanced/smoke/risk_budget/other）
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

### 1. 2022 Stress Window 过仓问题（P0）

**问题**: canonical `balanced` 在 test_2022 的 avg_pos ~0.38，promotion gate 要求 < 0.35。
**已尝试**: floor40 overlay, mode routing, penalty tuning, phase-balanced sampling — 均未同时满足 return + MaxDD + avg_pos。
**待尝试方向**:
- a) **Stress-specific reward shaping**: stress 态下仓位惩罚系数动态增大
- b) **Hard floor 提升**: 从 0.20 提升到 0.30-0.35，限制最低仓位
- c) **Regime-conditioned action space**: stress 态时减少 action levels（如只保留 [0, 0.2, 0.4]）
- d) **Stress episode 过采样**: 在 training batch 中增加 2022 类 episode 的比例

### 2. Multi-Asset 泛化（P1）

**问题**: 当前仅 SOXL 单标训练，多资产迁移未启动。
**已知失败教训**: 需要 generic reward + volatility normalization + asset-specific heads。
**待尝试方向**:
- a) **TQQQ 迁移测试**: 使用现有 SOXL actor 直接回测 TQQQ，量化 gap
- b) **共享 backbone 实验**: GRU 共享 + 每个资产一个 MLP head
- c) **Volatility normalization**: 将 realized_vol 按资产历史中位数标准化

---

## 下一步建议（选一个推进）

| 优先级 | 方向 | 假设 | 验证方式 | 预期时间 |
|--------|------|------|----------|----------|
| P0 | 2022 avg_pos 修复 | hard floor 0.30 或 stress action space 限制 | 修改 profile → 训练 → test_2022 gate | 2-3h |
| P1 | TQQQ 迁移测试 | SOXL actor 在 TQQQ 上直接回测 | 使用现有 actor + TQQQ qlib store | 30min |
| P2 | 自动化卡片检查 | 脚本验证每个 card.md 格式完整性 | 新增 check_cards.py | 1h |
| P2 | Skill 过期清理 | 删除 30 天未使用的 skill | 运行 check_skills.py + 手动审查 | 30min |

**推荐**: 先跑 P1（TQQQ 迁移测试），30 分钟就能得到多资产 gap 的量化数据，为后续联合训练提供依据。如果 2022 问题更紧急，则选 P0。
