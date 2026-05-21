# v19reg_v22_budget_usage_bull_guard_v1

## Governance

- Canonical baseline: `balanced_rule_residual_v22`.
- Direct comparison baseline: `v19reg_v22_budget_usage_v1`.
- Role: `diagnose`.
- Target window: saved walk-forward daily paths for `wf_2020`, `wf_2022`, and
  `wf_2023_2024`.
- Accepted trade-off: recover `wf_2020` bull participation while preserving the
  `wf_2022` defensive improvement; a small recovery-window return giveback is
  acceptable only if it remains materially better than v22.
- Changed layer: `selection` / fixed-path composer.
- External reference: existing `risk_budget_usage_v1` two-speed architecture and
  trend-following mode-band expansion family.
- Hypothesis: the v19_reg slow path correctly defends stress windows, but
  over-compresses shallow strong-trend windows. A price-confirmed bull guard can
  floor the raw composed target to the fast v22 position only during strong,
  shallow-drawdown trends.
- Acceptance criteria:
  - `wf_2020` return exceeds both v22 and the original composed path.
  - `wf_2022` MaxDD stays below `30%` and avg position stays below `0.20`.
  - `wf_2023_2024` remains better than v22 on return and MaxDD.
- Failure mode: the guard restores 2020 by reopening 2022 risk or erasing the
  2024 recovery gain.
- Comparison-table row key: `v19reg_v22_budget_usage_bull_guard_v1`.

## Historical Contrast

1. Closest success case: `v19reg_v22_budget_usage_v1` improved `wf_2022` and
   `wf_2023_2024` by using v19_reg as slow risk band and v22 as budget usage.
2. Closest failure case: the same v1 composer failed bull participation because
   `wf_2020` return fell below both raw daily paths.
3. Why this is not a duplicate: it keeps the same slow/fast actors but adds a
   distinct price-confirmed bull participation guard, rather than retraining,
   changing SOXX thresholds, or loosening all mode bands.

## Mechanism

The composer now supports `BullParticipationGuardConfig`. When enabled, it
computes:

- `price > MA50`
- `20d momentum >= 3%`
- drawdown from the 50d rolling high no worse than `8%`

When all conditions hold, the raw composed target is floored to the fast v22
position. The execution threshold and trade lock still apply after that raw
target change.

## Commands

All commands reused the saved daily inputs under
`runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/`.

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/compose_budget_usage_counterfactual.py \
  --slow runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2020/balanced_rule_residual_v19_reg_input.csv \
  --fast runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2020/balanced_rule_residual_v22_input.csv \
  --output-dir runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/walkforward/wf_2020 \
  --profile v19reg_v22_budget_usage_bull_guard_v1_wf_2020 \
  --slow-label balanced_rule_residual_v19_reg_wf_2020 \
  --fast-label balanced_rule_residual_v22_wf_2020 \
  --composed-label v19reg_v22_budget_usage_bull_guard_v1_wf_2020 \
  --bull-participation-guard --bull-ma-window 50 --bull-momentum-window 20 \
  --bull-high-window 50 --bull-min-periods 20 --bull-min-momentum 0.03 \
  --bull-max-drawdown 0.08
```

The same command shape was run for `wf_2022` and `wf_2023_2024` with the
matching input and output paths.

## Results

| window | original composed | bull guard | delta vs v1 | delta vs v22 | guard days | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wf_2020` | `49.86% / 0.965 / 29.85% / pos 0.394 / trades 23` | `98.27% / 1.456 / 26.14% / pos 0.399 / trades 25` | return `+48.41pp`, MaxDD `-3.71pp` | return `+36.85pp`, MaxDD `-8.21pp` | 113 | fixes bull participation |
| `wf_2022` | `-20.68% / -0.624 / 26.65% / pos 0.184 / trades 15` | `-20.68% / -0.624 / 26.65% / pos 0.184 / trades 15` | unchanged | return `+23.42pp`, MaxDD `-19.30pp` | 8 | defense preserved |
| `wf_2023_2024` | `62.64% / 1.424 / 30.18% / pos 0.316 / trades 27` | `58.19% / 1.326 / 30.18% / pos 0.337 / trades 29` | return `-4.45pp`, MaxDD unchanged | return `+25.64pp`, MaxDD `-3.03pp` | 49 | acceptable recovery giveback |

## Decision

Keep as a promoted diagnostic candidate, but do not bind runtime directly from
this script-level composition.

This is the first budget-usage follow-up that fixes the `wf_2020` participation
gap while preserving the `wf_2022` defensive path. The next valid continuation
is a runtime-governance packaging review for the composer/guard path, including
how to express it as a real runtime profile without hiding the price-derived
guard inside an ad-hoc script.

## Evidence

- `walkforward_summary.md`
- `walkforward/wf_2020/v19reg_v22_budget_usage_bull_guard_v1_wf_2020.md`
- `walkforward/wf_2022/v19reg_v22_budget_usage_bull_guard_v1_wf_2022.md`
- `walkforward/wf_2023_2024/v19reg_v22_budget_usage_bull_guard_v1_wf_2023_2024.md`

## 2026-05-21 Runtime Packaging Step

Decision: `policy_layer_ready_runtime_binding_pending`.

The price-derived bull guard has been moved out of the counterfactual app layer
and into `src/soxl_quant/policy/two_speed_budget.py`. The counterfactual card
now imports `BullParticipationGuardConfig` and guarded
`compose_budget_usage_policy_frame` from the policy layer, which gives future
runtime/profile code a shared implementation to call instead of duplicating the
diagnostic logic.

This is still not runtime auto-select evidence. The standard paper trading path
still selects and plans one actor profile; no governed runtime profile config
exists yet for `slow_profile + fast_profile + bull_guard`.

Evidence:

- `runtime_packaging_review.md`
- `runtime_packaging_review.json`
- `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_two_speed_budget.py tests/test_budget_counterfactual.py -q` -> `12 passed`
