# Model Selection Knowledge

## Retained Insight

Checkpoint choice often matters more than longer training. Several branches peaked early, and promotion should use fixed-split slices plus proxy metrics rather than headline return.

## Failed Pattern

Do not assume a branch is undertrained when validation improves but `test_2022` average position or max drawdown worsens.

## Use With Skills

- `post-train-backtest-review` for checkpoint comparison, fixed-split diagnosis, and promotion review.

## Evidence Path

- `runtime/model/knowledge.md`
- `runtime/model/knowledge.md`
- `runtime/model/`

## Reusable Method

Review `selection_history`, fixed-split `test_2020`, `test_2022`, `test_2023_2024`, and proxy MaxDD together. Prefer checkpoint review and fixed-actor counterfactuals before opening another broad retrain.

## Next Valid Experiment Type

A valid model-selection experiment changes checkpoint scoring or selection evidence, not reward coefficients or thresholds under a new name.

The v22/post-release-cap runtime-selection loop exposed a second selection
issue after the model and overlay gates passed: training `best_score` still
favored v19 even though corrected walk-forward evidence favored v22. Runtime
auto-selection now uses a complete walk-forward report when all standard
windows are present, scoring each window with the profile's selection weights
and relative outperformance as the return term. If no complete report exists it
falls back to the legacy summary `best_score`.

Current canonical runtime selection is `balanced_rule_residual_v22` with
`selective_release_recovery_post_release_cap_v1` explicitly approved for
auto-selection. The live walk-forward runtime selection scores are v22 `64.88`
versus v19 `46.17`.

## 2026-05-21 Continued-Training Diagnostic

`balanced_rule_residual_v22_continued_retention_v1` tested whether v22 needed
more PPO steps after checkpoint retention became available. It did not.
Continuing from the v22 actor retained iter `0`, `10`, and `20`, but iter `0`
remained best (`86.24` score) while iter `10` and iter `20` fell to `77.14` and
`69.26`. Proxy return increased, but risk widened: proxy MaxDD rose from v22
`54.52%` to `59.13%`, and `test_2022` worsened to `-51.24%` with MaxDD
`55.26%`, failing the fixed-split limit.

Conclusion: do not treat v22 as undertrained. Future model-selection work
should change checkpoint scoring, retained-evidence review, or a distinct
risk-control mechanism; a short continuation of v22 is not a valid promotion
path.

## 2026-05-21 Fixed-Path Budget-Usage Diagnostic

`v19reg_v22_budget_usage_v1` tested a model-selection / composition mechanism
instead of retraining: `balanced_rule_residual_v19_reg` defines the slow
risk-mode band, and `balanced_rule_residual_v22` is interpreted as fast usage
inside that band.

The first replay exposed stale baseline daily CSVs whose `test_2022` and
`test_2023_2024` positions were zero. Regenerate actor daily paths before
trusting fixed-path composition when slice coverage looks suspicious.

With fresh actor daily exports through `2026-05-19`, the composed path improved
versus fresh v22 on every fixed replay slice: proxy `2264.14% / 1.260 /
40.89% MaxDD`, `test_2022` `-21.20% / 31.09% MaxDD / 0.238 avg_pos`, and
`test_2023_2024` `244.80% / 1.606 / 30.67% MaxDD`.

Conclusion: keep as a diagnostic candidate and validate by walk-forward daily
composition before any runtime binding. This is a distinct mechanism from
continued PPO training, MoE routing, SOXX threshold tuning, or direct feature
retraining.

Walk-forward daily-path validation was mixed. `wf_2022` improved versus v22
from `-44.11% / 45.95% MaxDD / 0.228 avg_pos` to `-20.68% / 26.65% MaxDD /
0.184 avg_pos`, and `wf_2023_2024` improved from `32.56% / 33.21% MaxDD` to
`62.64% / 30.18% MaxDD`. But `wf_2020` return fell to `49.86%`, below both
v22 (`61.42%`) and v19_reg (`69.45%`) daily paths.

Conclusion update: diagnostic keep, no runtime binding. If this line continues,
the next mechanism must add a bull-participation guard or mode-band exception
and rerun all three walk-forward daily-path checks.

## 2026-05-21 Budget-Usage Bull-Participation Guard

`v19reg_v22_budget_usage_bull_guard_v1` completed that follow-up without a new
PPO run. The composer keeps v19_reg as the slow mode band and v22 as fast usage,
then applies a price-confirmed trend guard: `price > MA50`, `20d momentum >=
3%`, and 50d rolling drawdown no worse than `8%`. When the guard is true, the
raw composed target is floored to the fast v22 position before the existing
threshold and trade-lock logic.

Walk-forward daily-path readout:

- `wf_2020`: `98.27% / 1.456 / 26.14% MaxDD`, improving v1 by `+48.41pp`
  return and beating v22 by `+36.85pp`.
- `wf_2022`: unchanged from v1 at `-20.68% / -0.624 / 26.65% MaxDD /
  0.184 avg_pos`, preserving the defensive result.
- `wf_2023_2024`: `58.19% / 1.326 / 30.18% MaxDD`, giving back `4.45pp`
  versus v1 but still improving v22 by `+25.64pp` return and `-3.03pp` MaxDD.

Conclusion: this is now the best budget-usage continuation candidate, but still
not runtime evidence. The next valid step is a governance package that expresses
the slow/fast composition and price-derived guard as a real runtime/profile
interface, then verifies it through the production paper/dashboard path.

Follow-up slow-path ablation:

`v19reglight_v22_budget_usage_bull_guard_ablation_v1` replaced the slow
`balanced_rule_residual_v19_reg` walk-forward actors with
`balanced_rule_residual_v19_reg_light` while keeping v22 as fast usage and
keeping the same bull guard. It did not improve the approved composer:
`wf_2020` tied exactly at `98.27% / 1.456 / 26.14% MaxDD`, `wf_2022` gave up
return/sharpe (`-23.22% / -1.231` versus `-20.68% / -0.624`) despite lower
MaxDD, and `wf_2023_2024` regressed to `33.34% / 0.873 / 36.91% MaxDD` versus
approved `58.19% / 1.326 / 30.18% MaxDD`.

Conclusion update: keep `balanced_rule_residual_v19_reg` as the slow path.
Do not repeat regularization slow-path swaps unless a new mechanism explains
why the recovery drawdown/return trade-off should change.
