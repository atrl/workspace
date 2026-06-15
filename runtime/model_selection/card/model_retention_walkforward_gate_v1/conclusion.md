# Model Retention Walk-Forward Gate

- Reference baseline: `soxl_backtest_policy_reimplementation_v1` return `146.043798%`, MaxDD `13.554089%`.
- Current best YTD model return: `224.361539%`; required alignment floor at ratio `0.95` is `213.143462%`.
- Walk-forward requirement: windows `wf_2020, wf_2022, wf_2023_2024` all present and MaxDD strictly below `20.00%`.
- Passing promotable profiles: `none`.

This card is a retention/promotion audit. It does not delete actors or override production runtime protection.
