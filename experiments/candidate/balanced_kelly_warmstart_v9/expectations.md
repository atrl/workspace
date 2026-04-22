# Expectations: balanced_kelly_warmstart_v9

## What Must Improve

1. **Walk-forward 2023-2024 Return**: Must exceed v4's 51.76%.
   The hypothesis is that Kelly-positive windows in early 2023 will receive
   enough bonus to nudge the policy toward higher exposure.

2. **No regression on test_2022 MaxDD**: Must remain < 35%.
   Since we start from v4 (31.3%) and use very low lr, catastrophic
   regression is unlikely.

## Accepted Trade-off

- test_2020 Return may drop slightly (v4 was 111.6%).
- Turnover may increase slightly if the model becomes more dynamic.

## Go/No-Go Criteria

- **GO**: WF 2023-2024 Return > 60% AND test_2022 MaxDD < 35%
- **NO-GO**: WF 2023-2024 Return < 50% (no improvement) OR test_2022 MaxDD > 38%
