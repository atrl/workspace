---
date: '2026-04-24'
title: 'Next Actions'
kind: ops
status: active
direction: harness
---

# Next Actions

接替 `codex_exec_plans/active/current.md` 的活任务面。
只保留尚未完成的验证项和待启动的实验方向。

## 当前待办

1. **项目代码整理**（本次任务本身）
   - 清理 codex_exec_plans
   - 按方向拆分 memory
   - 精简 experiments
   - 重命名 skill 并适配多 ETF
   - 拆分 artifacts

2. **待验证假设**
   - canonical `balanced` 在 2022 stress window 的 avg_pos 仍偏高（~0.38），需要新的 stress sampling 或 regime-specific reward 来修复
   - multi-asset joint training 的失败教训：需要 generic reward + volatility normalization + asset-specific heads

## 最近完成的验证

- `scripts/check_harness.py`: passed=62, failures=0
- `scripts/check_experiment_cards.py`: packages=64, registry_rows=110
- `scripts/check_research_knowledge.py`: files=7, active_budget=300
