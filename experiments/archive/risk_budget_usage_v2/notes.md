---
title: risk_budget_usage_v2
direction: model
status: ablation
---

# risk_budget_usage_v2

## Motivation

v1 proxy Sharpe 0.942 有提升空间。v2 测试 mode-conditional reward shaping +
dynamic band scaling 是否能引导 PPO 发现更少保守的策略。

## Configuration Delta vs v1

```yaml
reward:
  mode_conditional_reward_enabled: true
  flat_wait_pos_penalty: 0.15
  crash_rebuild_bonus: 0.20
mode_router:
  use_dynamic_bands: true
```

## Result

| 切片 | Return | Sharpe | MaxDD |
|------|--------|--------|-------|
| test_proxy | 158.3% | 0.884 | 20.6% |
| test_2022 | -3.2% | -0.272 | **7.4%** |

## Decision

**Ablation.** Reward shaping 导致过度保守（2022 avg_pos 0.070），proxy Sharpe
未达 0.95。Dynamic bands 未能补偿保守性损失。

## Evidence Links

- Landscape: `docs/experiment-landscape-review.md`
