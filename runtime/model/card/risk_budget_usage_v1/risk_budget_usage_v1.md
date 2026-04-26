# risk_budget_usage_v1

Fixed-signal counterfactual combining the slow `mode_v1` risk governor with the fast `imitation_v4` path as a budget-usage signal.

| variant | test_2020 ret/sh/dd | test_2022 ret/sh/dd/pos | test_2023_2024 ret/sh/dd | proxy ret/sh/dd/trades |
| --- | --- | --- | --- | --- |
| mode_v1_slow | 177.1% / 1.936 / 29.9% / pos 0.502 / trades 45 | -24.8% / -0.461 / 37.5% / pos 0.303 / trades 32 | 103.6% / 1.082 / 32.7% / pos 0.439 / trades 82 | 1156.6% / 1.076 / 53.3% / pos 0.426 / trades 244 |
| imitation_v4_fast | 103.9% / 1.401 / 43.7% / pos 0.524 / trades 37 | -19.3% / -0.279 / 34.8% / pos 0.303 / trades 12 | 151.2% / 1.260 / 32.4% / pos 0.484 / trades 54 | 1200.9% / 1.046 / 55.8% / pos 0.457 / trades 161 |
| risk_budget_usage_v1 | 239.0% / 2.285 / 21.7% / pos 0.490 / trades 34 | -17.1% / -0.358 / 25.6% / pos 0.214 / trades 27 | 127.8% / 1.278 / 32.1% / pos 0.389 / trades 60 | 2003.6% / 1.248 / 45.0% / pos 0.391 / trades 188 |

## Decision

candidate
