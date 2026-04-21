# v13 Expectations

## Targets

| Metric | v2 Baseline | v12 Actual | v13 Target |
|--------|-------------|------------|------------|
| test_proxy_max_dd | 50.4% | 48.8% | **< 50%** |
| test return | 521.3% | 492.3% | **> 510%** |
| test_proxy_sharpe | 0.86 | 0.85 | **> 0.85** |

## Success Criteria

1. **MaxDD < 50%** — must keep v12's breakthrough
2. **Return > 510%** — recover meaningful portion of lost return
3. **Sharpe > 0.85** — maintain risk-adjusted performance

## Failure Modes

- If MaxDD > 52%: penalty too weak → increase to 5.0
- If Return < 500%: still too conservative → increase threshold to 0.20
- If MaxDD < 46% and Return < 500%: agent too conservative → reduce asymmetry to 1.0
