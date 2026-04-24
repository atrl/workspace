# balanced_rule_residual_v24 — Low-Frequency Coarse Residual Probe

## Hypothesis

The current repo frontier is not another feature pack or overlay retune. This
branch tests whether `v19` improves when the actor is forced to make lower-
frequency, lower-resolution residual decisions:

- `decision_frequency: 2`
- hybrid residual deltas compressed from 5 actions to 3 actions

## Baselines

- Direct comparison baseline:
  - `balanced_rule_residual_v19`
- Historical success anchor:
  - `balanced_decision_freq_quick`
- Historical failure anchor:
  - `balanced_rule_residual_v23`

## Why This Exists

- `balanced_decision_freq_quick` showed that changing decision timescale can
  improve drawdown and generalization, but it lived on an older branch and gave
  up too much return.
- `v23` showed that adding more transition-state features still collapses into
  broad de-risking.
- This branch asks a narrower question: can temporal abstraction help the
  retained `v19` family without reopening failed feature or overlay directions?

## Evidence Links

- Active workpad:
  - `memory/index.json`
- Historical trade-compression note:
  - `memory/2026-04-18-trade-compression-experiments.md`
- Frontier summary:
  - `docs/experiment-landscape-review.md`

## Result

- dropped after evaluation
- kept only as a temporal-compression failure reference

## Outcome

- recovered actor evaluation after fixing a summary persistence bug:
  - valid: `61.2% / 1.67 / 27.3% / avg_pos 0.601`
  - full test: `439.9% / 0.775 / 58.35% / 176 trades`
  - `test_2022`: `-22.24% / -0.86 / 30.67% / avg_pos 0.257`
  - `test_2023_2024`: `38.23% / 0.66 / 31.58%`
  - `test_proxy`: `758.6% / 0.961 / 54.58%`

## Readout

- the intended gain did show up weakly:
  - `test_2022` drawdown and avg_pos both improved slightly vs `v19`
- the branch failed the core hypothesis:
  - trades did not compress; review markers expanded from roughly `35` on `v19`
    to `176`
  - `test_2023_2024` return fell from `43.6%` to `38.2%`
- conclusion:
  - coarse residual deltas on the current execution path do not behave like
    cleaner temporal abstraction
  - they behave like noisier, chunkier rebalancing

## Current Role

- archived dropped probe
- use as the closest failure case for future temporal-abstraction ideas
- only reopen if the next branch adds action persistence, dwell constraints, or
  a hierarchical rule/actor split
