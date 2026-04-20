# v9 Expectations

## Promotion Gate Targets

| Check | Threshold | v2 Actual | v5 Actual | v7 Actual | v8 Actual | v9 Target |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| test_2022_max_dd | < 35.0% | 23.6% ✓ | 32.7% ✓ | 33.9% ✓ | 31.3% ✓ | < 35.0% |
| test_2022_avg_pos | < 0.35 | 0.18 ✓ | 0.27 ✓ | 0.27 ✓ | 0.27 ✓ | < 0.25 |
| test_proxy_sharpe | > 0.75 | 0.86 ✓ | 0.86 ✓ | 0.87 ✓ | 0.88 ✓ | > 0.80 |
| test_proxy_max_dd | < 50%* | 50.4% | 55.9% | 58.4% | 53.4% | < 50% |

*Not a formal gate but the user's primary optimization target.

## Success Criteria

1. **test_proxy_maxdd < 50%** — must beat v2's 50.4% and v8's 53.4%
2. **valid best score > 95** — restore learning quality vs v8's 89.54
3. **test_2022_avg_pos < 0.25** — hard_floor=0.10 should enable deeper de-risking
4. **test_proxy_sharpe > 0.80** — maintain competitiveness

## Failure Modes

- If valid score < 90: penalties still too strong, dial back drawdown_penalty to 0.30
- If test_2022_max_dd > 35%: penalties too weak for 2022 control
- If maxdd > 55%: step-level penalties fundamentally insufficient, need episode-level redesign
- If test_2022_avg_pos > 0.25: hard_floor reduction ineffective, agent not learning to de-risk
