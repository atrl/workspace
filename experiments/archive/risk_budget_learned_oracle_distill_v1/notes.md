# risk_budget_learned_oracle_distill_v1

## Motivation

`hierarchical_online_budget_generalization` showed that the deployable
hierarchical runtime helps, and the strongest line is a learned router plus slow
refresh. The unresolved question is whether the fast actor can represent that
learned-router usage policy at all when trained directly on the resulting daily
path.

## External Reference

- DAgger / behavior-cloning literature: this is a no-promotion capacity probe,
  not a production training method
  (`https://proceedings.mlr.press/v15/ross11a.html`).
- Hierarchical RL / temporal abstraction: this run keeps the learned router as
  the slow layer and tests whether a single fast actor can compress the usage
  policy under that regime
  (`https://proceedings.neurips.cc/paper/2016/hash/f442d33fa06832082290ad8544a8da27-Abstract.html`).

## Hypothesis

If a single actor trained on the deployable `hier_learned_slow5` daily path
still cannot approach that path's return profile, then the bottleneck remains
actor representation / observation sufficiency even after moving to the learned
router regime.

## Target Slices / Windows

- Diagnostic only on `test_2020`, `test_2022`, `test_2023_2024`, and
  `test_proxy`.
- Compare against `actor_runtime`, `hier_learned_slow5`, and the earlier
  `risk_budget_oracle_distill_v1`.

## Acceptance Criteria

- The script must mark the result as no-promotion.
- The replay source must be the deployable `soxl_hier_learned_slow5.csv` path,
  not the fixed replay counterfactual.
- The result must be recorded under `evidence/post_train_review/` and the
  landscape table.

## Expected Trade-Off

This run uses held-out replay labels and therefore cannot be promoted. A strong
result only proves capacity under the learned-router runtime shape.

## Failure Mode

If the actor remains materially weaker than the `hier_learned_slow5` daily
path, do not spend the next loop on more cadence or threshold drift. The next
work should change the fast actor architecture or the observation/training
objective in that learned-router regime.

## Comparison-Table Row Key

`risk_budget_learned_oracle_distill_v1`

## Result

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_oracle_distill_v1 \
  --replay-daily evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json evidence/post_train_review/risk_budget_learned_oracle_distill_v1.json \
  --output-md evidence/post_train_review/risk_budget_learned_oracle_distill_v1.md
```

Evidence:

- samples: `1508`
- label distribution: `{0: 238, 1: 1138, 2: 48, 3: 52, 4: 32}`
- BC loss / accuracy: `1.124 / 0.781`
- `test_2020`: `72.5% / Sharpe 3.162 / MaxDD 8.0% / avg_pos 0.286`
- `test_2022`: `-11.7% / Sharpe -1.054 / MaxDD 13.2% / avg_pos 0.105`
- `test_2023_2024`: `78.6% / Sharpe 1.130 / MaxDD 24.9% / avg_pos 0.336`
- `test_proxy`: `412.9% / Sharpe 1.042 / MaxDD 28.7% / avg_pos 0.296`

## Decision

Drop as promotable by construction. This learned-router oracle is stronger than
the earlier fixed-replay oracle probe (`412.9%` vs `363.8%` proxy return), but
it is still far below the deployable `hier_learned_slow5` target path
(`195.3% / 0.741 / 36.6%` on SOXL with higher drawdown budget) and nowhere near
the fixed replay ceiling. That keeps the bottleneck on single-actor capacity /
state representation rather than just router cadence.
