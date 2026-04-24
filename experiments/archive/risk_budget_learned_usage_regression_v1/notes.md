# risk_budget_learned_usage_regression_v1

## Motivation

The discrete learned-router oracle probes may be throwing away information by
forcing a 5-bin action label. This run keeps the same replay teacher and the
same deployable learned-router runtime, but replaces the classification head
with direct continuous usage regression.

## External Reference

- Continuous action / sizing literature in trading and control suggests that a
  direct scalar regression head can preserve amplitude information that is lost
  under coarse action quantization.

## Hypothesis

If the missing edge is mainly caused by quantizing usage into 5 bins, then a
continuous usage regression head should beat the learned-router oracle
classification probes on `test_proxy` and `test_2023_2024`.

## Target Slices / Windows

- Diagnostic only on `test_2020`, `test_2022`, `test_2023_2024`, and
  `test_proxy`.
- Compare against `risk_budget_learned_oracle_distill_v1`,
  `risk_budget_learned_conditioned_oracle_distill_v1`, and `hier_learned_slow5`.

## Acceptance Criteria

- The output is marked no-promotion.
- Replay source remains `soxl_hier_learned_slow5.csv`.
- The probe uses continuous usage targets rather than discrete action labels.

## Expected Trade-Off

This run still uses held-out replay labels. A good result only proves that
continuous target regression is a better diagnostic head than discrete
classification.

## Failure Mode

If regression still does not materially improve the learned-router probe, the
next loop should change backbone capacity or sequence structure, not just the
head type.

## Comparison-Table Row Key

`risk_budget_learned_usage_regression_v1`

## Result

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/build_budget_usage_regression_probe.py \
  --profile risk_budget_learned_usage_regression_v1 \
  --replay-daily runtime/model/reviews/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 2000 \
  --sample-size 0 \
  --output-json runtime/model/reviews/risk_budget_learned_usage_regression_v1.json \
  --output-md runtime/model/reviews/risk_budget_learned_usage_regression_v1.md
```

Evidence:

- samples: `1508`
- target mean / std: `0.247 / 0.177`
- train loss / MAE: `0.00230 / 0.0351`
- `test_2022`: `-15.1% / Sharpe -1.436 / MaxDD 17.6% / avg_pos 0.102`
- `test_2023_2024`: `63.0% / Sharpe 0.978 / MaxDD 25.7% / avg_pos 0.332`
- `test_proxy`: `222.8% / Sharpe 0.772 / MaxDD 34.7% / avg_pos 0.303`

## Decision

Drop as promotable by construction. Despite low regression loss, the continuous
usage head was materially worse than the learned-router oracle classification
probe (`222.8% / 0.772 / 34.7%` vs `412.9% / 1.042 / 28.7%`). That is strong
evidence that the current bottleneck is not just discrete action quantization.
