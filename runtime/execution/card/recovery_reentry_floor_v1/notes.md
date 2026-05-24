# Recovery Reentry Floor v1

Date: 2026-05-24

## Governance

- Baseline: `v19reg_v22_budget_usage_bull_guard_v1`
- Direct comparison: current Futu paper composer from
  `runtime/production/card/paper_model_oos_risk_check_v1`
- Role: `diagnose`
- Changed layer: `execution`
- Target window: strict OOS poor windows in 2025-2026, with walk-forward
  validation on `wf_2020`, `wf_2022`, and `wf_2023_2024`
- Accepted trade-off: improve recent recovery/upside capture only if saved
  walk-forward actors do not lose 2022 defense or increase drawdown materially
- Failure mode: a high-volatility rebound floor becomes a hidden 2022 exposure
  increase or a 2026-specific overfit
- Comparison-table row key: `recovery_reentry_floor_v1`

## External Basis

- [Momentum Crashes](https://econpapers.repec.org/RePEc:chf:rpseri:rp1361):
  panic states after market declines and high volatility can coincide with
  rebounds, so recovery logic must be state-aware rather than a static leverage
  increase.
- [Isolating momentum crashes](https://www.sciencedirect.com/science/article/abs/pii/S0927539821000979):
  rebounding bear markets and high momentum volatility are distinct from normal
  bull momentum, motivating a separate crash/rebound indicator.
- [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912):
  separates trend estimation and position sizing inside a volatility-scaled
  framework, matching this fixed-actor sizing counterfactual.
- [Trend-following with better adaptation to large downside risks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276322):
  trend-following can underperform buy-and-hold in positive trend phases when
  adaptation is sluggish.
- [Qlib GitHub](https://github.com/microsoft/qlib): keep the test inside the
  qlib research-to-production harness rather than a standalone notebook.

## Historical Context

Closest success:

- `v19reg_v22_budget_usage_bull_guard_v1` showed that a fixed-actor,
  price-confirmed participation guard can improve bull-window capture without
  retraining.

Closest failures:

- `selective_release_trade_lock_bypass_v1` and later guarded variants showed
  that relaxing execution in recovery states can improve order generation but
  widen drawdown.
- `soxx_relative_weakness_walkforward_v1` showed that fixed-window gains can be
  absorbed or reversed on saved walk-forward actors.

Why this is not a duplicate:

- This guard fires when both actors are effectively flat but qlib features show
  high-volatility positive-momentum rebound (`mom_10`, `mom_21`,
  `relative_strength_21`, `vol_of_vol_21`, `vix_percentile_252`), then floors
  the low raw target to a bounded `0.35` rather than raising all bull states.

## Command

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/compose_budget_usage_counterfactual.py \
  --slow runtime/model/card/balanced_rule_residual_v19_reg/balanced_rule_residual_v19_reg_soxl_daily.csv \
  --fast runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv \
  --output-dir runtime/execution/card/recovery_reentry_floor_v1 \
  --profile recovery_reentry_floor_v1 \
  --slow-label balanced_rule_residual_v19_reg_current \
  --fast-label balanced_rule_residual_v22_current \
  --composed-label recovery_reentry_floor_v1 \
  --description "Fixed-actor recovery/reentry floor counterfactual after OOS poor-window diagnostics." \
  --bull-participation-guard \
  --recovery-reentry-guard \
  --reentry-min-target 0.35 \
  --reentry-max-target 0.55 \
  --reentry-max-base-target 0.25 \
  --reentry-min-mom-10 0.08 \
  --reentry-min-mom-21 0.12 \
  --reentry-min-relative-strength-21 1.0 \
  --reentry-min-vol-of-vol-21 0.10 \
  --reentry-max-vix-percentile-252 0.85
```

The same guard was replayed on saved walk-forward inputs under:

```text
runtime/execution/card/recovery_reentry_floor_v1/walkforward/
```

## Results

| window | baseline return | candidate return | baseline max_dd | candidate max_dd | baseline avg_pos | candidate avg_pos | gate_days | changed_days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict OOS 2020-01-02..2026-05-21 | 1542.89% | 2091.90% | 41.59% | 41.59% | 0.462 | 0.468 | 35 | 73 |
| wf_2020 | 98.27% | 108.00% | 26.14% | 26.14% | 0.399 | 0.434 | 23 | 25 |
| wf_2022 | -20.68% | -22.79% | 26.65% | 26.10% | 0.184 | 0.212 | 21 | 99 |
| wf_2023_2024 | 58.19% | 58.19% | 30.18% | 30.18% | 0.337 | 0.337 | 0 | 0 |

Worst 2026 poor-window read-through:

| window | baseline strategy | candidate strategy | buy-hold | baseline excess | candidate excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-02-10..2026-05-11 | 1.74% | 33.78% | 200.63% | -198.89pp | -166.85pp |
| 2026-02-05..2026-05-06 | 17.69% | 46.84% | 211.46% | -193.77pp | -164.61pp |

## Decision

Keep `recovery_reentry_floor_v1` as diagnostic evidence, but do not promote it
to runtime.

The guard proves the current 2026 recovery gap is actionable through fixed-actor
execution mapping: strict OOS return improves by `+549.01pp`, and the worst
2026 poor-window excess improves by roughly `+32pp`. However, it does not reduce
full-period max drawdown below the `15%` target, and saved `wf_2022` return
falls while average exposure rises from `0.184` to `0.212`.

Next valid loop: add a narrower reentry state or paired post-reentry de-risk
that explicitly blocks the 2022 false positives while keeping the 2026 recovery
target change. Do not bind this floor to paper/live runtime as-is.

## Artifacts

```text
runtime/execution/card/recovery_reentry_floor_v1/recovery_reentry_floor_v1.json
runtime/execution/card/recovery_reentry_floor_v1/recovery_reentry_floor_v1.md
runtime/execution/card/recovery_reentry_floor_v1/recovery_reentry_floor_v1_soxl_daily.csv
runtime/execution/card/recovery_reentry_floor_v1/oos_poor_window_diagnostics/
runtime/execution/card/recovery_reentry_floor_v1/walkforward/
```
