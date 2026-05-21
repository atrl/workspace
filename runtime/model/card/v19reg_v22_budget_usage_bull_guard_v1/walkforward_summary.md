# v19reg_v22_budget_usage_bull_guard_v1 Walk-Forward Daily-Path Review

## Summary

The bull-participation guard fixes the `v19reg_v22_budget_usage_v1` 2020
participation gap while preserving the 2022 defensive result. It costs a small
amount of 2024 recovery return versus the original composed path, but remains
materially better than the raw v22 daily path.

This remains a composition diagnostic, not a runtime profile.

## Results

| window | v22 daily path | v1 composed | bull guard composed | bull guard vs v1 | bull guard vs v22 | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wf_2020` | `61.42% / 1.077 / 34.35% / pos 0.402 / trades 177` | `49.86% / 0.965 / 29.85% / pos 0.394 / trades 23` | `98.27% / 1.456 / 26.14% / pos 0.399 / trades 25` | return `+48.41pp`, MaxDD `-3.71pp` | return `+36.85pp`, MaxDD `-8.21pp` | fixes bull participation |
| `wf_2022` | `-44.11% / -1.500 / 45.95% / pos 0.228 / trades 181` | `-20.68% / -0.624 / 26.65% / pos 0.184 / trades 15` | `-20.68% / -0.624 / 26.65% / pos 0.184 / trades 15` | unchanged | return `+23.42pp`, MaxDD `-19.30pp` | defense preserved |
| `wf_2023_2024` | `32.56% / 0.866 / 33.21% / pos 0.362 / trades 178` | `62.64% / 1.424 / 30.18% / pos 0.316 / trades 27` | `58.19% / 1.326 / 30.18% / pos 0.337 / trades 29` | return `-4.45pp`, MaxDD unchanged | return `+25.64pp`, MaxDD `-3.03pp` | acceptable recovery giveback |

## Mechanism Readout

- Guard rule: `price > MA50`, `20d momentum >= 3%`, and 50d rolling-high
  drawdown no worse than `8%`.
- Guard action: floor raw composed target to the fast v22 position.
- Guard days: `113` in `wf_2020`, `8` in `wf_2022`, and `49` in
  `wf_2023_2024`.
- The `wf_2022` guard days did not alter the final daily path after threshold
  and trade-lock processing.

## Decision

Keep as a promoted diagnostic candidate and use it as the next runtime
packaging candidate if this line continues.

Do not bind runtime directly yet. The next step must convert the composition and
price-derived guard into governed runtime/profile semantics, then verify the
same post-close/paper/dashboard path used by the current production loop.

## Evidence

- `walkforward/wf_2020/v19reg_v22_budget_usage_bull_guard_v1_wf_2020.json`
- `walkforward/wf_2022/v19reg_v22_budget_usage_bull_guard_v1_wf_2022.json`
- `walkforward/wf_2023_2024/v19reg_v22_budget_usage_bull_guard_v1_wf_2023_2024.json`
