# risk_budget_learned_attention_oracle_distill_v1

## Motivation

The learned-router oracle probes suggest the bottleneck is no longer the head
type or explicit regime identity. The next cheap question is whether the fast
actor simply needs a stronger temporal backbone than the current GRU.

## External Reference

- Attention-based sequence models can preserve longer and more selective
  temporal context than a small GRU, especially when the useful signal is not
  uniformly distributed across the observation window.

## Hypothesis

If backbone capacity is the bottleneck, then an attention oracle probe trained
on the same learned-router teacher path should beat the GRU oracle probe on
`test_proxy` and `test_2023_2024`.

## Target Slices / Windows

- Diagnostic only on `test_2020`, `test_2022`, `test_2023_2024`, and
  `test_proxy`.
- Compare against `risk_budget_learned_oracle_distill_v1`,
  `risk_budget_learned_conditioned_oracle_distill_v1`,
  `risk_budget_learned_usage_regression_v1`, and `hier_learned_slow5`.

## Acceptance Criteria

- The output is marked no-promotion.
- Replay source remains `soxl_hier_learned_slow5.csv`.
- Only the backbone changes; head type and teacher path stay aligned with the
  best existing learned-router oracle probe.

## Expected Trade-Off

This run still uses held-out replay labels. A good result only proves that an
attention backbone is a better capacity probe than the current GRU.

## Failure Mode

If attention does not materially improve the oracle probe, the next loop should
move beyond small backbone swaps and look at broader sequence/state redesign.

## Comparison-Table Row Key

`risk_budget_learned_attention_oracle_distill_v1`

## Result

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_attention_oracle_distill_v1 \
  --replay-daily evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json evidence/post_train_review/risk_budget_learned_attention_oracle_distill_v1.json \
  --output-md evidence/post_train_review/risk_budget_learned_attention_oracle_distill_v1.md
```

Evidence:

- samples: `1508`
- label distribution: `{0: 238, 1: 1138, 2: 48, 3: 52, 4: 32}`
- BC loss / accuracy: `1.186 / 0.719`
- `test_2022`: `-11.7% / Sharpe -1.054 / MaxDD 13.2% / avg_pos 0.105`
- `test_2023_2024`: `78.6% / Sharpe 1.130 / MaxDD 24.9% / avg_pos 0.336`
- `test_proxy`: `412.9% / Sharpe 1.042 / MaxDD 28.7% / avg_pos 0.296`

## Decision

Drop as promotable by construction. The attention backbone produced effectively
the same policy metrics as the GRU learned-router oracle probe, despite slightly
worse BC fit. That means this scale of backbone swap is not the missing
ingredient.
