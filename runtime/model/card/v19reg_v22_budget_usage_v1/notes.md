# v19reg_v22_budget_usage_v1

## Governance

- Canonical baseline: `balanced_rule_residual_v22`
- Direct comparison baseline: `balanced_rule_residual_v19_reg`
- Role: `diagnose`
- Target window: fixed daily-path replay across `test_proxy`, `test_2020`,
  `test_2022`, and `test_2023_2024`; walk-forward only if the replay improves
  the v22/v19_reg risk-recovery trade-off without a large recovery collapse.
- Accepted trade-off: may give up some proxy return versus v22 if `test_2022`
  drawdown and average position improve, but should not push
  `test_2023_2024` below the v19_reg recovery frontier.
- Changed layer: `selection`
- External reference: hierarchical risk-budget allocation, reusing the existing
  `risk_budget_usage_v1` architecture where a slow path sets the risk mode band
  and a fast path supplies usage inside that band.
- Comparison-table row key: `v19reg_v22_budget_usage_v1`

## Hypothesis

`balanced_rule_residual_v19_reg` is the more defensive fixed-split baseline,
while `balanced_rule_residual_v22` is the current canonical runtime actor with
stronger production readiness. A fixed-path budget-usage composer may preserve
v22's participation signal while letting v19_reg anchor risk mode and cap
stress exposure.

## Acceptance Criteria

- A reusable, non-hardcoded counterfactual artifact path writes JSON, daily CSV,
  and Markdown review output under this card.
- The composed path improves `test_2022` MaxDD and average position versus v22.
- `test_2023_2024` return stays at or above the v19_reg baseline, or any loss is
  offset by a materially better `test_2022` drawdown profile.
- The result is interpreted as a diagnostic only; no runtime profile is bound
  unless a later walk-forward validation passes.

## Failure Mode

- v19_reg mode anchoring suppresses v22 recovery timing and repeats the
  over-conservative `risk_budget_usage_v2` failure.
- v22 as a usage ratio increases exposure inside favorable bands and repeats the
  `balanced_rule_residual_v19_reg_v22_retest_v1` stress-risk failure.
- Fixed-split replay looks good but cannot be reproduced by walk-forward daily
  paths.

## Historical Contrast

Closest success:

- `risk_budget_usage_v1` showed that slow-mode plus fast-usage composition can
  improve the 2022 risk/recovery frontier without retraining.
- `balanced_rule_residual_v22` remains the canonical runtime after corrected
  walk-forward selection and scheduled auto-selection proof.

Closest failure:

- `risk_budget_usage_v2` confirmed that simply making budget usage more
  conservative can destroy recovery.
- `risk_budget_usage_v3` failed walk-forward after adding mode-conditioned model
  inputs.
- `balanced_rule_residual_v19_reg_v22_retest_v1` increased 2022 exposure and
  failed fixed-split risk checks.

Why this is not a duplicate:

- It keeps both actors fixed and changes only selection/composition semantics.
- It does not add SOXX thresholds, MoE routing, VIX feature retraining, or a
  stronger penalty.
- The output path is a generalized harness artifact rather than a renamed
  profile or a hardcoded `risk_budget_usage_v1` replay.

## Run Log

2026-05-21:

Focused TDD / harness support:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_budget_counterfactual.py -q
```

Result:

- Red phase: missing `src.soxl_quant.app.budget_counterfactual` failed
  collection as expected.
- Green phase: `3 passed`.

Initial stale-input attempt:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/compose_budget_usage_counterfactual.py \
  --slow runtime/model/card/balanced_rule_residual_v19_reg/balanced_rule_residual_v19_reg_soxl_daily.csv \
  --fast runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv \
  --output-dir runtime/model/card/v19reg_v22_budget_usage_v1 \
  --profile v19reg_v22_budget_usage_v1 \
  --slow-label balanced_rule_residual_v19_reg \
  --fast-label balanced_rule_residual_v22 \
  --composed-label v19reg_v22_budget_usage_v1
```

The existing source daily CSVs were invalid for this replay because both paths
went to zero after 2020, producing zero-position `test_2022` and
`test_2023_2024` slices. Treat that attempt as input-provenance evidence only,
not as a model result.

Fresh input exports:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v19_reg \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/inputs/balanced_rule_residual_v19_reg_fresh_input.csv

PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v22 \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/inputs/balanced_rule_residual_v22_fresh_input.csv
```

Fresh-input composition:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/compose_budget_usage_counterfactual.py \
  --slow runtime/model/card/v19reg_v22_budget_usage_v1/inputs/balanced_rule_residual_v19_reg_fresh_input.csv \
  --fast runtime/model/card/v19reg_v22_budget_usage_v1/inputs/balanced_rule_residual_v22_fresh_input.csv \
  --output-dir runtime/model/card/v19reg_v22_budget_usage_v1 \
  --profile v19reg_v22_budget_usage_v1 \
  --slow-label balanced_rule_residual_v19_reg_fresh \
  --fast-label balanced_rule_residual_v22_fresh \
  --composed-label v19reg_v22_budget_usage_v1
```

Artifacts:

- JSON:
  `runtime/model/card/v19reg_v22_budget_usage_v1/v19reg_v22_budget_usage_v1.json`
- Review:
  `runtime/model/card/v19reg_v22_budget_usage_v1/v19reg_v22_budget_usage_v1.md`
- Daily path:
  `runtime/model/card/v19reg_v22_budget_usage_v1/v19reg_v22_budget_usage_v1_soxl_daily.csv`
- Fresh inputs:
  `runtime/model/card/v19reg_v22_budget_usage_v1/inputs/`

Fresh fixed-path replay result:

- Proxy: `2264.14%` return / `1.260` Sharpe / `40.89%` MaxDD /
  `0.437` avg position / `185` trades.
- Versus fresh v22 daily export: proxy return `+1135.83pp`, Sharpe `+0.287`,
  MaxDD `-12.33pp`, avg position `-0.063`, trades `-1251`.
- `test_2020`: `131.95%` return / `1.774` Sharpe / `25.19%` MaxDD.
- `test_2022`: `-21.20%` return / `-0.458` Sharpe / `31.09%` MaxDD /
  `0.238` avg position; versus fresh v22, return `+20.54pp`, MaxDD
  `-21.75pp`, avg position `-0.161`.
- `test_2023_2024`: `244.80%` return / `1.606` Sharpe / `30.67%` MaxDD;
  versus fresh v22, return `+51.72pp`, MaxDD `-1.59pp`, avg position
  `-0.055`.

Decision:

- Fixed-path replay passes the diagnostic acceptance criteria and is worth
  walk-forward daily-path validation.
- Do not bind runtime from this result. It is a fixed full-window replay, not a
  saved walk-forward actor/composer proof.
- Next valid loop: export saved walk-forward daily paths for v19_reg and v22,
  compose each window with the same generalized script, and compare against the
  retained walk-forward actor baselines.

## Walk-Forward Daily-Path Validation

2026-05-21 continuation:

Harness fix:

- `scripts/export_profile_daily_backtest.py` now resolves the saved
  walk-forward actor layout
  `runtime/model/card/<profile>_<window>/soxl_<profile>_<window>_actor.pt`.
- `tests/test_artifact_paths.py` covers the fallback layout and explicit
  `--model` override.

Commands:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v19_reg --tag wf_2020 \
  --test-start 2020-01-01 --test-end 2020-12-31 \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2020/balanced_rule_residual_v19_reg_input.csv

PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v22 --tag wf_2020 \
  --test-start 2020-01-01 --test-end 2020-12-31 \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2020/balanced_rule_residual_v22_input.csv

PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v19_reg --tag wf_2022 \
  --test-start 2022-01-01 --test-end 2022-12-31 \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2022/balanced_rule_residual_v19_reg_input.csv

PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v22 --tag wf_2022 \
  --test-start 2022-01-01 --test-end 2022-12-31 \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2022/balanced_rule_residual_v22_input.csv

PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v19_reg --tag wf_2023_2024 \
  --test-start 2024-01-01 --test-end 2024-12-31 \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2023_2024/balanced_rule_residual_v19_reg_input.csv

PYTHONPATH=src:. qlib_venv/bin/python scripts/export_profile_daily_backtest.py \
  --profile balanced_rule_residual_v22 --tag wf_2023_2024 \
  --test-start 2024-01-01 --test-end 2024-12-31 \
  --output runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/inputs/wf_2023_2024/balanced_rule_residual_v22_input.csv
```

Then each window was composed with
`scripts/compose_budget_usage_counterfactual.py` into:

- `runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/wf_2020/`
- `runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/wf_2022/`
- `runtime/model/card/v19reg_v22_budget_usage_v1/walkforward/wf_2023_2024/`

Readout:

- `wf_2020`: composed `49.86%` return / `0.965` Sharpe / `29.85%` MaxDD /
  `0.394` avg position / `23` trades. Versus v22 daily path, return is
  `-11.56pp` but MaxDD improves `-4.50pp`.
- `wf_2022`: composed `-20.68%` return / `-0.624` Sharpe / `26.65%` MaxDD /
  `0.184` avg position / `15` trades. Versus v22 daily path, return improves
  `+23.42pp`, MaxDD improves `-19.30pp`, and avg position drops `-0.044`.
- `wf_2023_2024`: composed `62.64%` return / `1.424` Sharpe / `30.18%` MaxDD /
  `0.316` avg position / `27` trades. Versus v22 daily path, return improves
  `+30.08pp`, MaxDD improves `-3.03pp`, and avg position drops `-0.045`.

Decision:

- Keep as a diagnostic candidate / no runtime binding.
- The 2022 defense and 2024 recovery improvements are real in daily-path replay,
  but `wf_2020` loses too much bull participation.
- Next valid continuation would need a distinct bull-participation guard or
  mode-band exception, then the same three WF daily-path checks. Do not continue
  this line by generic PPO training, SOXX cap tuning, or MoE hyperparameter
  sweeps.

Summary:

- `runtime/model/card/v19reg_v22_budget_usage_v1/walkforward_summary.md`
