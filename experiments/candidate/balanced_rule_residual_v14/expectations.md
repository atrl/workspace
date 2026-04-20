# v14 Expectations

## Clean Ablation Test

| Metric | v2 Baseline (MLP) | v12 (GRU+penalty) | v14 (GRU only) |
|--------|-------------------|-------------------|----------------|
| test_proxy_max_dd | 50.4% | 48.8% | ??? |
| test return | 521.3% | 492.3% | ??? |
| test_proxy_sharpe | 0.86 | 0.85 | ??? |

## Interpretation

- **If MaxDD < 50%**: GRU temporal memory alone controls drawdown. Episode penalty unnecessary. Architecture is the key.
- **If MaxDD 50-52%**: GRU helps modestly; penalty adds marginal benefit.
- **If MaxDD > 53%**: Penalty is essential even with GRU. Need both.
