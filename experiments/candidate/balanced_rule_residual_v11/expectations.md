# v11 Expectations

## Promotion Gate Targets

| Check | Threshold | v2 Actual | v8/v9/v10 Actual | v11 Target |
|-------|-----------|-----------|------------------|------------|
| test_2022_max_dd | < 35.0% | 23.6% ✓ | 31.3% ✓ | < 28.0% |
| test_2022_avg_pos | < 0.35 | 0.18 ✓ | 0.27 ✓ | < 0.25 |
| test_proxy_sharpe | > 0.75 | 0.86 ✓ | 0.85 ✓ | > 0.80 |
| test_proxy_max_dd | < 50%* | 50.4% | 53.4% | **< 48%** |

*Primary optimization target.

## Success Criteria

1. **test_proxy_maxdd < 48%** — per-step penalty must produce visible drawdown improvement
2. **test return > 450%** — agent must not become too conservative
3. **valid best score > 85** — maintain learning capability
4. **test_2022_max_dd < 30%** — keep 2022 control advantage

## Failure Modes & Next Steps

- If maxdd > 52%: per-step penalty still insufficient → increase to 10.0+ or add GRU layer
- If return < 350%: agent too conservative → raise threshold to 0.15 or reduce penalty to 4.0
- If valid score < 80: observation_window:3 causes overfitting → revert to 1 or add regularization
- If identical to v10: bug not fully fixed → emergency debug

## Critical Validation

Before declaring v11 complete, verify:
1. Training log shows `episode_max_dd` penalty being applied (rewards should be more negative than v10)
2. Test metrics are DIFFERENT from v8/v9/v10 (539.3%/0.85/53.4%)
3. If test metrics are still identical, the bug fix failed
