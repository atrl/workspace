---
title: aggressive
direction: reward
status: candidate
---

# aggressive

## Motivation

Higher risk-tolerance variant of `balanced` with elevated return_scale,
weaker drawdown penalties, and stronger activity incentives.

## Configuration Delta vs balanced

```yaml
reward:
  return_scale: 1.40          # vs 1.15
  drawdown_penalty: 0.18      # vs 0.28
  activity_bonus: 1.40        # vs 1.25
position_cap:
  bull_min_cap: 0.60          # vs 0.65
```

## Result

TBD — no recent run on current harness.

## Evidence Links

- Landscape: `docs/experiment-landscape-review.md`
