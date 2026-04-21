# v12 Expectations

## Promotion Gate Targets

| Check | Threshold | v2 Actual | v11 Actual | v12 Target |
|-------|-----------|-----------|------------|------------|
| test_2022_max_dd | < 35.0% | 23.6% ✓ | 31.3% ✓ | < 28.0% |
| test_2022_avg_pos | < 0.35 | 0.18 ✓ | 0.27 ✓ | < 0.25 |
| test_proxy_sharpe | > 0.75 | 0.86 ✓ | 0.85 ✓ | > 0.80 |
| test_proxy_max_dd | < 50%* | 50.4% | 54.4% | **< 50%** |

*Primary optimization target.

## Success Criteria

1. **test_proxy_maxdd < 50%** — GRU must enable drawdown control that MLP couldn't achieve
2. **test return > 450%** — agent must not become too conservative
3. **valid best score > 85** — maintain learning capability
4. **Metrics DIFFERENT from v11** — confirms GRU is actually changing behavior

## Failure Modes & Next Steps

- If maxdd > 53%: GRU insufficient for this problem → try LSTM or larger GRU hidden size
- If return < 350%: terminal penalty too strong → reduce to 4.0
- If identical to v2: GRU not learning temporal patterns → debug hidden state
- If valid score < 80: GRU overfitting → add dropout or reduce hidden size
