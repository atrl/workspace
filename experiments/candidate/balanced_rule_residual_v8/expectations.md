# v8 Expectations

## Promotion Gate Targets

| Check | Threshold | v2 Actual | v5 Actual | v7 Actual | v8 Target |
|-------|-----------|-----------|-----------|-----------|-----------|
| test_2022_max_dd | < 35.0% | 23.6% ✓ | 32.7% ✓ | 33.9% ✓ | < 35.0% |
| test_2022_avg_pos | < 0.35 | 0.18 ✓ | 0.27 ✓ | 0.27 ✓ | < 0.30 |
| test_proxy_sharpe | > 0.75 | 0.86 ✓ | 0.86 ✓ | 0.87 ✓ | > 0.80 |
| test_proxy_max_dd | < 50%* | 50.4% | 55.9% | 58.4% | < 50% |

*Not a formal gate but the user's primary optimization target.

## Success Criteria

1. **test_proxy_maxdd < 50%** — must beat v2's 50.4%
2. **test_proxy_sharpe > 0.80** — maintain competitiveness
3. **test_2022_avg_pos < 0.25** — restore v2's bear-market de-risking behavior
4. **valid best score > 95** — comparable to v5/v7

## Failure Modes

- If maxdd > 55%: step-level penalties insufficient, need code-level episode drawdown penalty
- If valid overfits early (score drops after @20): [128,64] may still be too large, consider [64,32]
- If return < 400%: penalties too strong, dial back drawdown_penalty to 0.50
