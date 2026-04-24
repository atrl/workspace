# risk_budget_oracle_distill_v1

## Motivation

`risk_budget_usage_v1` fixed replay passed every replay gate, but actor-backed
parity, native PPO, and train-split BC warm-start all failed to reproduce the
fixed replay upside. This diagnostic asks a narrower question: can the current
single actor even represent the replay budget-usage path if trained directly on
that path?

## External Reference

- DAgger / behavior-cloning literature: one-shot imitation can fail under
  distribution shift, so this run is framed as an oracle capacity probe rather
  than a promotable training method
  (`https://proceedings.mlr.press/v15/ross11a.html`).
- Hierarchical RL / temporal abstraction: the successful replay path separates
  slow risk-budget selection from fast participation
  (`https://proceedings.neurips.cc/paper/2016/hash/f442d33fa06832082290ad8544a8da27-Abstract.html`).

## Hypothesis

If full replay labels still do not produce a strong actor, then the bottleneck
is model representation / observation state rather than train-split label
coverage. If full replay labels do produce a strong actor, the next valid path
is not promotion; it is a non-leaking policy-composer or dataset-aggregation
design.

## Target Slices / Windows

- Diagnostic only on `test_2020`, `test_2022`, `test_2023_2024`, and
  `test_proxy`.
- Compare against native `risk_budget_usage_v1`, `risk_budget_bc_warmstart_v1`,
  and fixed replay `risk_budget_usage_v1`.

## Acceptance Criteria

- The script must mark the output as no-promotion.
- The label distribution must report whether action-4 full usage exists.
- The result must be recorded in `evidence/post_train_review/` and the landscape
  table.

## Expected Trade-Off

This run knowingly uses held-out replay labels, so any good result only proves
capacity. It cannot be promoted or used as a canonical actor.

## Failure Mode

If the actor remains conservative or fails to beat native `risk_budget_usage_v1`,
stop single-actor distillation and move to explicit policy composition.

## Comparison-Table Row Key

`risk_budget_oracle_distill_v1`

## Result

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_oracle_distill_v1 \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0
```

Evidence:

- samples: `1508`
- label distribution: `{0: 127, 1: 436, 2: 383, 3: 423, 4: 139}`
- BC loss / accuracy: `1.195 / 0.711`
- `test_2022`: `-5.5% / Sharpe -0.350 / MaxDD 11.0% / avg_pos 0.117`
- `test_2023_2024`: `58.5% / Sharpe 1.180 / MaxDD 19.3% / avg_pos 0.271`
- `test_proxy`: `363.8% / Sharpe 1.090 / MaxDD 23.7% / avg_pos 0.264`

## Decision

Drop as promotable by construction. The actor saw full replay labels, including
action-4 usage examples, but still did not approach the fixed replay's `2003.6%`
proxy return. This supports moving to explicit policy composition rather than
more single-actor PPO/BC attempts.
