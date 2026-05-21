# v19reg_v22_budget_usage_v1 Walk-Forward Daily-Path Review

## Summary

The fixed-path budget-usage mechanism survives the stress and recovery windows
but gives up too much `wf_2020` bull participation to promote directly.

This review uses regenerated daily paths from saved walk-forward actors and the
same daily-position replay method for the slow path, fast path, and composed
path. It is a composition diagnostic, not a full broker/executor promotion
review.

## Results

| window | slow v19_reg | fast v22 | composed | composed vs v22 | decision |
| --- | --- | --- | --- | --- | --- |
| `wf_2020` | `69.45% / 1.186 / 28.64% / pos 0.387 / trades 171` | `61.42% / 1.077 / 34.35% / pos 0.402 / trades 177` | `49.86% / 0.965 / 29.85% / pos 0.394 / trades 23` | return `-11.56pp`, MaxDD `-4.50pp`, trades `-154` | fails bull participation |
| `wf_2022` | `-47.26% / -1.357 / 49.52% / pos 0.273 / trades 182` | `-44.11% / -1.500 / 45.95% / pos 0.228 / trades 181` | `-20.68% / -0.624 / 26.65% / pos 0.184 / trades 15` | return `+23.42pp`, MaxDD `-19.30pp`, trades `-166` | strong defense improvement |
| `wf_2023_2024` | `42.15% / 1.084 / 30.70% / pos 0.335 / trades 180` | `32.56% / 0.866 / 33.21% / pos 0.362 / trades 178` | `62.64% / 1.424 / 30.18% / pos 0.316 / trades 27` | return `+30.08pp`, MaxDD `-3.03pp`, trades `-151` | recovery improvement |

## Interpretation

- The mechanism meaningfully reduces churn in every window: composed trades are
  `23`, `15`, and `27` versus roughly `178-182` for the raw WF paths.
- `wf_2022` confirms the fixed replay's defensive intuition: v19_reg mode bands
  plus v22 usage reduce average exposure and drawdown.
- `wf_2023_2024` also improves, so this is not merely an over-conservative
  replay like `risk_budget_usage_v2`.
- `wf_2020` is the blocker. The same low-churn execution path lowers drawdown
  versus v22, but it also loses `11.56pp` return versus v22 and `19.59pp`
  versus v19_reg.

## Decision

Keep as a diagnostic candidate, but do not bind runtime.

The next valid follow-up is not another PPO continuation or SOXX threshold
tune. If this line continues, it needs a distinct bull-participation guard or
mode-band exception for strong trend windows, then the same three walk-forward
daily-path checks must be rerun.

## Evidence

- `walkforward/wf_2020/v19reg_v22_budget_usage_v1_wf_2020.md`
- `walkforward/wf_2022/v19reg_v22_budget_usage_v1_wf_2022.md`
- `walkforward/wf_2023_2024/v19reg_v22_budget_usage_v1_wf_2023_2024.md`
