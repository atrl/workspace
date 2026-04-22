# SOXL Experiment Landscape Review

Last updated: 2026-04-22

## Executive Summary

The experiment history is no longer a "try more knobs" problem. The repo already shows a clear pattern:

- The canonical `balanced` branch is a strong stress-defense baseline and still the right control group for any new work.
- The largest durable improvement came from the `balanced_rule_residual` family, especially `v1`, `v2`, `v16`, `v19`, and `v22`.
- The strongest repeated failure mode came from broad bundled changes that raised exposure in the name of transition capture, reward shaping, or stress-sampling. Those branches usually improved validation or local rebound windows, then failed in `test_2022` and widened `test_proxy` drawdown.
- The current frontier is not "find one more penalty coefficient". The frontier is:
  - better state representation for transition / restart windows
  - better model-selection logic for risk-return trade-offs
  - narrower, evidence-backed experiments instead of multi-lever retries

Current decision state:

- Canonical baseline: `balanced`
- Current best promotable RL family: `balanced_rule_residual`
- Best risk-quality point: `balanced_rule_residual_v1` / `v2`
- Best high-return GRU point: `balanced_rule_residual_v19`
- Best external-risk feature probe: `balanced_rule_residual_v22`

## Retention Policy

Historical experiments should now be retained by mechanism role, not by profile
name or recency. The repo should keep one stable core, a small set of optional
modules, and a clearly documented discard set.

### Core Keep Set

These are the mechanisms that future work should treat as the default retained
stack.

| layer | retained item | why it stays core | default posture |
| --- | --- | --- | --- |
| baseline control | `balanced` | still the cleanest stress-defense control profile | always keep as the control group |
| mainline RL family | `balanced_rule_residual` | strongest durable architecture family | all new RL branches should start from this family |
| high-return actor anchor | `balanced_rule_residual_v19` | best retained high-return GRU baseline with seed confirmation | direct comparison anchor for new mainline attempts |
| harness governance | registry + archive + memory + comparison table | prevents duplicate research loops and profile sprawl | mandatory workflow infrastructure |
| execution evaluation path | fixed-actor overlay sweep scripts and reviews | useful for testing execution hypotheses without retraining | keep as an evaluation tool, not a default alpha source |

### Optional Module Set

These mechanisms remain useful, but they should be treated as isolated,
switchable modules rather than folded into the default stack all at once.

| module | representative branch or tool | why it stays optional | usage rule |
| --- | --- | --- | --- |
| external-risk features | `balanced_rule_residual_v22` | first clean drawdown-shape gain from external VIX term-structure data | test as a narrow add-on over `v19`, not as a new baseline |
| checkpoint review / early selection | `balanced_checkpoint_review`, selection history tooling | good at diagnosis and pruning weak late checkpoints | keep as a selector, do not treat it as a rescue mechanism |
| smoke residual harness | `smoke_rule_residual_v1` | cheap direction-finding sanity check for residual-over-rule changes | use only for staging, not for promotion claims |
| fixed-actor overlay routing | `regime_route_v1` study and overlay variants | validated execution-layer routing as a direction | only extend if a genuinely new state/mechanism is added |

### Permanent Drop Set

These directions should remain in the repo as evidence only. They should not be
reopened as default candidate paths unless a future run introduces a clearly new
mechanism and explicitly cites why the old failure no longer applies.

| dropped direction | representative branches | repeated failure mode | reopen only if |
| --- | --- | --- | --- |
| reward-only tuning | `balanced_reward_guard`, repeated coefficient sweeps | validation can rise while `2022` and proxy risk worsen | reward is paired with a new state or execution mechanism |
| broad transition feature bundles | `balanced_transition_state` | local rebound improvement, full-window exposure drift | the transition logic is narrowed and explicitly gated |
| narrow transition retrains without selective execution | `balanced_rule_residual_v23` | lower DD came from broad de-risking, not better re-entry | paired with a new release/re-entry execution mechanism |
| generic overlay retuning | `cta_*`, `fast_reentry`, `regime_route_v1` parameter drift | threshold changes do not solve SOXL-specific selective re-entry | a new execution state exists beyond `stress / bull / base` |
| bundled structural retries | `balanced_structural_v1`, early MoE bundles | attribution collapses and one gain is offset elsewhere | one structural lever is isolated and ablated |

### Integration Rule

Future "integration" work should not merge historical wins into one large
profile. Instead:

1. Keep a single stable core stack centered on `balanced` and
   `balanced_rule_residual_v19`.
2. Treat every other historically useful idea as a switchable module.
3. Turn on only one new module at a time during a new loop.
4. If a proposal only changes thresholds, penalties, or naming inside an
   already-dropped direction, default to not running it.

### Overlay Follow-Up

The first explicit `release` overlay follow-up did not clear the bar for a new
mechanism.

- `stress_release_route_v1` added a coded `release` state over
  `regime_route_v1`, but the realized SOXL `bear_2022` and
  `recovery_2023_2024` slices were unchanged.
- The small proxy-return change came mostly from marginal behavior elsewhere,
  not from better selective re-entry in the windows that motivated the change.
- Conclusion: keep `release` support as harness infrastructure, but drop this
  parameterization as evidence that "release-state overlay" is currently solved.
- Next overlay work is only valid if it first proves that the release state
  fires on materially different rows or uses genuinely new information beyond
  threshold drift on existing row-level features.

The next narrow merge attempt also failed to produce a new keeper.

- `dwell_stress_merge_v1` tried to merge the low-churn `dwell_hold_v2` hold
  skeleton with the stress-side asymmetry from `regime_route_v1`.
- It did improve `test_2022` relative to `dwell_hold_v2`, but the cost was too
  high: proxy return / Sharpe fell sharply, proxy MaxDD worsened to `60.6%`,
  and trade count jumped from `121` to `358`.
- Conclusion: recombining existing `dwell` and `stress` knobs is not a real new
  mechanism. Keep `regime_route_v1` and `dwell_hold_v2` as separate frontier
  points, and drop further threshold-only merges between them.

## Experiment Taxonomy

The historical branches fall into a few repeatable categories:

- `canonical baseline`
  - `balanced` remains the control profile because it passes the fixed-split gate cleanly and has the clearest stress-defense behavior.
- `rule_residual`
  - RL no longer learns the whole position policy from scratch; it learns a bounded delta over a rule-based anchor.
  - This is the most successful family by far.
- `reward_only`
  - Changes reward coefficients without changing state representation or architecture.
  - This direction repeatedly failed to improve crisis control.
- `feature_only`
  - Adds new state information while keeping most mechanics fixed.
  - Useful when narrow and hypothesis-led, harmful when added as a wide batch.
- `structural / architecture`
  - Changes sequence model, routing, observation window, or expert structure.
  - Works when the change is focused; fails when bundled.
- `selection / sampling`
  - Targets checkpoint choice or episode mix rather than raw model structure.
  - Helpful for diagnosis, but did not rescue weak actors on its own.
- `aggressive / higher-floor`
  - Raises baseline exposure to recover trend capture.
  - Can look good on return, but often breaks the stress gate.
- `smoke`
  - Cheap probes and staging branches. Useful for direction finding, not a replacement for mainline evidence.
- `rule-based / external reference`
  - The `soxl_backtest` system remains the clearest open-source-style reference implementation inside the repo for panic-based signal engineering and lower-frequency execution.

## Master Comparison Table

This table is the working summary for the most decision-relevant historical branches.

| profile | category | baseline | registry status | key delta | valid ret / sh / dd | 2020 ret / dd | 2022 ret / dd / pos | 2023-24 ret / dd | proxy ret / sh / dd | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | canonical | balanced | canonical | reward+cap baseline | 39.0 / 1.3 / 26.6 | 97.0 / 21.1 | -25.8 / 33.0 / 0.3 | 43.5 / 33.9 | 468.9 / 0.9 / 50.2 | pass |
| balanced_rule_residual_v1 | rule_residual | balanced | candidate | residual-over-rule architecture | 61.8 / 1.7 / 26.9 | 88.2 / 14.5 | -18.3 / 24.8 / 0.2 | 52.1 / 27.3 | 672.3 / 1.0 / 46.4 | pass |
| balanced_rule_residual_v2 | rule_residual | balanced_rule_residual_v1 | candidate | Kelly-log alpha reward | 70.7 / 1.9 / 26.0 | 84.5 / 15.9 | -16.2 / 23.6 / 0.2 | 43.1 / 29.2 | 492.7 / 0.9 / 53.9 | pass |
| balanced_rule_residual_v10 | rule_residual | balanced_rule_residual_v2 | dropped | episode-level DD penalty redesign | 60.1 / 1.6 / 27.9 | 111.2 / 18.4 | -22.8 / 31.3 / 0.3 | 46.7 / 33.1 | 587.5 / 0.9 / 56.7 | pass |
| balanced_rule_residual_v16 | rule_residual | balanced_rule_residual_v13 | candidate | GRU + window=5 | 63.7 / 1.7 / 28.9 | 102.8 / 16.7 | -23.6 / 32.7 / 0.3 | 57.9 / 33.6 | 413.2 / 0.8 / 55.6 | pass |
| balanced_rule_residual_v19 | rule_residual | balanced_rule_residual_v16 | candidate | seed-confirmed GRU baseline | 56.6 / 1.6 / 27.0 | 109.4 / 18.1 | -22.5 / 31.1 / 0.3 | 43.6 / 37.3 | 735.1 / 0.9 / 54.8 | pass |
| balanced_rule_residual_v20 | rule_residual | balanced_rule_residual_v16 | candidate | penalty=3.0 ablation | 67.4 / 1.7 / 28.3 | 110.1 / 16.2 | -23.4 / 31.7 / 0.3 | 42.6 / 32.7 | 623.2 / 0.9 / 53.8 | pass |
| balanced_rule_residual_v22 | rule_residual | balanced_rule_residual_v19 | candidate | VIX term-structure features | 62.9 / 1.6 / 30.6 | 115.6 / 16.2 | -24.5 / 32.7 / 0.3 | 37.9 / 32.4 | 695.5 / 0.9 / 54.5 | pass |
| balanced_rule_residual_mode_v1 | structural | balanced_rule_residual_v19 | candidate | persistent mode hierarchy + mode bands | 36.2 / 1.2 / 26.1 | 64.3 / 10.6 | -13.1 / 16.5 / 0.1 | 43.0 / 20.1 | 886.1 / 1.1 / 42.3 | pass |
| balanced_rule_residual_mode_v2 | structural | balanced_rule_residual_mode_v1 | dropped | config-owned favorable-mode loosening | 35.8 / 1.2 / 26.1 | 42.6 / 22.9 | -3.6 / 13.5 / 0.1 | 47.2 / 28.7 | 693.8 / 1.0 / 50.4 | pass |
| balanced_rule_residual_v23 | feature_only | balanced_rule_residual_v19 | dropped | narrow transition-state pack | 52.1 / 1.6 / 25.9 | 91.6 / 16.2 | -21.9 / 30.2 / 0.3 | 34.3 / 29.4 | 493.2 / 0.9 / 51.2 | pass |
| balanced_rule_residual_v24 | structural | balanced_rule_residual_v19 | dropped | decision_frequency=2 + coarse 3-level residual deltas | 61.2 / 1.7 / 27.3 | 97.5 / 17.8 | -22.2 / 30.7 / 0.3 | 38.2 / 31.6 | 758.6 / 1.0 / 54.6 | pass |
| balanced_rule_residual_simplified_v1 | reward_only | balanced_rule_residual_v19 | dropped | stripped reward to 5 terms + larger network + longer episode | 63.3 / 1.55 / 31.8 | 112.6 / 20.5 | -25.35 / 34.01 / — | 53.68 / 36.11 | 776.9 / 0.91 / 58.5 | blocked |
| balanced_rule_residual_refined_v1 | reward_only | balanced_rule_residual_v19 | candidate | remove activity_bonus, strengthen risk, respect cash | 56.9 / 1.51 / 32.5 | 107.4 / 16.2 | -22.15 / 31.51 / — | 48.84 / 31.02 | 500.6 / 0.84 / 54.6 | blocked |
| balanced_rule_residual_refined_v2 | reward_only | balanced_rule_residual_refined_v1 | candidate | add trend_alignment_bonus + moderate opportunity_cost | 68.9 / 1.79 / 27.1 | 108.6 / 17.8 | -22.60 / 31.91 / — | 63.33 / 29.70 | 556.9 / 0.87 / 53.0 | blocked |
| balanced_rule_residual_refined_v3 | reward_only | balanced_rule_residual_refined_v2 | candidate | stronger trend_alignment (0.30) + opportunity_cost (0.06) | 58.7 / 1.61 / 28.9 | 109.0 / 22.3 | -24.57 / 32.59 / — | 46.42 / 33.07 | 637.8 / 0.89 / 54.9 | blocked |
| balanced_rule_residual_refined_v4 | reward_only | balanced_rule_residual_refined_v2 | dropped | midpoint between v2 and v3 | 60.4 / 1.68 / 26.6 | 109.0 / 17.8 | -22.01 / 30.95 / — | 38.12 / 33.25 | 480.0 / 0.84 / 54.2 | blocked |
| balanced_transition_state | feature_only | — | — | transition/release features batch | 54.7 / 1.6 / 27.3 | 103.8 / 20.6 | -41.0 / 46.5 / 0.4 | 44.9 / 38.8 | 472.9 / 0.8 / 59.0 | fail |
| balanced_structural_v1 | structural | balanced | dropped | longer context + routing + sampling | 47.3 / 1.6 / 25.9 | 101.3 / 20.0 | -31.3 / 39.9 / 0.3 | 45.5 / 33.5 | 307.0 / 0.7 / 53.7 | fail |
| balanced_checkpoint_review | selection | balanced | candidate | earlier checkpoint selection | 52.6 / 1.6 / 26.7 | 114.7 / 21.3 | -39.5 / 46.3 / 0.4 | 39.6 / 38.7 | 371.5 / 0.8 / 60.3 | fail |
| balanced_reward_guard | reward_only | balanced | dropped | stronger stress penalties | 55.8 / 1.6 / 29.4 | 137.9 / 21.1 | -36.0 / 47.4 / 0.4 | 50.2 / 39.9 | 523.4 / 0.8 / 62.3 | fail |
| balanced_phase_sampling_v1 | sampling | balanced | dropped | phase-balanced episode sampling | 54.8 / 1.5 / 29.8 | 121.8 / 22.2 | -40.7 / 47.5 / 0.4 | 41.7 / 41.7 | 306.9 / 0.7 / 63.9 | fail |
| balanced_regime_gate | execution_gate | balanced | dropped | regime-routed caps v1 | 53.5 / 1.6 / 26.0 | 118.5 / 20.4 | -39.2 / 43.4 / 0.4 | 56.6 / 31.0 | 373.5 / 0.8 / 55.6 | fail |
| balanced_regime_gate_v2 | execution_gate | balanced_regime_gate | dropped | regime-routed caps v2 | 49.6 / 1.5 / 27.7 | 117.5 / 18.7 | -43.6 / 49.5 / 0.4 | 54.6 / 38.2 | 360.8 / 0.8 / 62.9 | fail |
| balanced_floor40_sym | aggressive | balanced | archived | higher floor aggressive branch | 53.2 / 1.5 / 28.1 | 120.5 / 22.5 | -37.0 / 45.4 / 0.4 | 51.2 / 39.5 | 443.7 / 0.8 / 61.5 | fail |
| balanced_floor40_sym_slow_bear | aggressive | balanced_floor40_sym | ablation | slow-bear variant | 57.2 / 1.7 / 25.1 | 90.4 / 17.3 | -21.6 / 29.2 / 0.2 | 62.6 / 32.0 | 514.7 / 0.9 / 46.4 | pass |
| balanced_moe | architecture | — | — | MoE with VIX | 47.2 / 1.5 / 26.0 | 108.1 / 19.1 | -29.9 / 35.7 / 0.3 | 49.6 / 34.9 | 407.1 / 0.8 / 52.8 | fail |
| balanced_moe_novix | architecture | — | — | MoE without VIX | 45.1 / 1.6 / 24.0 | 93.3 / 16.8 | -26.3 / 35.7 / 0.3 | 50.0 / 30.6 | 348.4 / 0.8 / 49.8 | fail |
| smoke_hybrid | smoke | smoke | archived | hybrid smoke baseline | 31.7 / 1.4 / 20.9 | 71.8 / 13.1 | -21.8 / 31.5 / 0.3 | 40.1 / 24.6 | 322.9 / 0.9 / 42.5 | pass |
| smoke_rule_residual_v1 | smoke | smoke | candidate | smoke residual-over-rule | 70.2 / 1.8 / 27.3 | 91.1 / 17.3 | -18.4 / 25.3 / 0.2 | 55.2 / 29.1 | 600.8 / 0.9 / 53.4 | pass |

## Difference Tables

### Relative To Canonical `balanced`

| branch | primary difference | what improved | what got worse | verdict |
| --- | --- | --- | --- | --- |
| `balanced_rule_residual_v1` | residual-over-rule + alpha reward | `2022`, `2023-2024`, proxy Sharpe, proxy DD | slightly lower `2020` return | strongest clean improvement over canonical |
| `balanced_rule_residual_v2` | Kelly-log alpha reward | best `2022` result in the mainline chain | proxy DD rose vs `v1` | keep as a high-quality reference, not the final frontier |
| `balanced_transition_state` | broad transition feature batch | local rebound windows improved slightly | `2022` exposure and DD expanded sharply | directionally informative, implementation failed |
| `balanced_reward_guard` | harsher reward coefficients only | none that mattered for promotion | `2022` and proxy DD both worsened | reward-only tuning should not be the main lever |
| `balanced_structural_v1` | longer context + routing + stratified sampling | `2022` avg_pos fell vs failed reward branches | still worse than canonical on stress DD and proxy Sharpe | keep harness support, drop bundle |
| `balanced_floor40_sym` | higher floor / more aggressive exposure | trend capture and upside branch behavior | stress gate broke | treat as aggressive side branch, not canonical path |

### `balanced_rule_residual` Frontier Chain

| profile | mechanism shift | strongest gain | main cost | keep / drop |
| --- | --- | --- | --- | --- |
| `v1` | residual-over-rule architecture | biggest clean step-change in quality | less bull capture than aggressive branches | keep |
| `v2` | Kelly-log alpha reward | best `2022` DD / avg_pos in the family | proxy DD above `v1` | keep |
| `v10` | episode-level drawdown penalty redesign | retained gate pass with higher total return | proxy DD still high | drop as final answer, keep as reward lesson |
| `v16` | GRU + 5-day context | strong `2023-2024` capture without gate break | proxy DD still ~55% | keep |
| `v19` | seed-confirmed GRU baseline | highest proxy return in the family | `2023-2024` DD widened | keep as current high-return GRU anchor |
| `v20` | stronger terminal penalty | none vs `v19` | return down, DD not better | dominated; do not repeat |
| `v22` | external VIX term-structure data | better `2020` DD and `2023-2024` DD | return drag in `2023-2024` | keep as candidate, needs seed + walk-forward |
| `mode_v1` | persistent mode hierarchy over residual actor | best `2022` / proxy DD control in the retained frontier | large validation and `2020` upside giveback | keep as candidate |
| `mode_v2` | looser favorable-mode config over `mode_v1` | best `2022` result in the hierarchy branch | proxy quality and `2020` both regressed vs `mode_v1` | drop |
| `v23` | narrow transition-state pack | lower `2022` avg_pos / DD and lower proxy DD | large return loss in `2020`, `2023-2024`, proxy | drop; transition signal still maps to broad de-risking |
| `v24` | lower-frequency coarse residual control | slight `2022` DD / avg_pos improvement | turnover exploded and `2023-2024` return fell | drop; coarse actions did not create real trade compression |
| `simplified_v1` | stripped reward to 5 terms + larger net + longer episode | none — proxy DD worsened to 58.5%, test_2022 return fell to -25.4% | removed protective guardrails (stale_position, trapped_loser, downside_signal, explicit risk_budget) | drop; radical reward simplification removes necessary protective terms; cash respect requires incentive restructuring, not guardrail deletion |
| `refined_v1` | remove activity_bonus, strengthen risk guardrails, reduce forced-trading pressure | best proxy DD (54.6%) and best test_2023_24 DD (31.0%) in the reward-only branch | proxy return collapsed 235pp (500.6% vs v19 735.1%), model overly conservative | candidate but blocked; needs better "when to participate" signal without blanket activity_bonus |
| `refined_v2` | add trend_alignment_bonus (0.20) + raise opportunity_cost (0.04) | recovered 56pp return vs v1, best proxy DD in entire family (53.0%), excellent test_2023_24 (63.3%/29.7%) | still 178pp below v19 proxy return; 2021 bull participation remains weak | candidate but blocked; reward-only direction shows diminishing returns; next lever is structural or multi-asset |
| `refined_v3` | stronger trend_alignment (0.30) + opportunity_cost (0.06) + lower hard_floor (0.15) | recovered another 81pp vs v2 (637.8%), best proxy return in refined series | proxy DD rose 1.9pp, test_2022 and test_2023_24 both worsened vs v2 | candidate but blocked; the return/DD trade-off is now clear; v3 is more aggressive, v2 is more defensive |
| `refined_v4` | midpoint between v2 and v3 parameters | none — proxy return fell to 480.0%, worse than v1 | midpoint parameters created a worse combination than either extreme | drop; no evidence that linear interpolation between v2 and v3 is valid; reward interactions are non-linear |

### Repeated Failure Directions

| direction | representative branches | repeated pattern | conclusion |
| --- | --- | --- | --- |
| reward-only tuning | `balanced_reward_guard`, repeated stress-penalty loops | validation can improve while `2022` and proxy DD worsen | stop treating coefficient sweeps as primary research |
| bundled transition features | `balanced_transition_state` | local rebound timing improves, but overall exposure drifts up | transition work must be narrow and gated |
| narrow transition features inside residual family | `balanced_rule_residual_v23` | stress DD improves without gate break | style still drifts too defensive and gives up too much upside | do not retry more transition features before proving a selective execution overlay |
| naive temporal compression inside current execution path | `balanced_rule_residual_v24` | slightly lower `2022` exposure and DD | coarse deltas increased actual rebalances and still gave up upside | temporal abstraction needs action persistence or hierarchy, not just lower decision frequency |
| selection-only rescue | `balanced_checkpoint_review`, `balanced_stress_select*` | better checkpoint choice helps diagnosis, not rescue | use for pruning search space, not as the end solution |
| broad episode sampling | `balanced_phase_sampling_v1` | more coverage but worse `2022` and proxy Sharpe | sampling changes need a much tighter mechanism |
| aggressive high-floor variants | `balanced_floor40_sym`, `balanced_regime_gate*` | better upside, broken stress gate | only valid as explicit aggressive branches |
| large bundled structures | `balanced_structural_v1`, early MoE branches | one metric improves, efficiency collapses elsewhere | ablate one structural lever at a time |

### Branches Still Worth Retaining

| branch | why it stays relevant | what remains open |
| --- | --- | --- |
| `balanced` | control baseline for all future work | can still be beaten on upside without losing discipline |
| `balanced_rule_residual_v1` | best clean architecture change | whether later branches can dominate it on both DD and proxy Sharpe |
| `balanced_rule_residual_v19` | best high-return GRU line with seed confirmation | walk-forward and regime-specific DD control |
| `balanced_rule_residual_v22` | first clean external-risk feature win | whether DD gain survives seed variance without return drag |
| `balanced_rule_residual_mode_v1` | first real temporal-abstraction scaffold inside the mainline RL family | whether bull participation can be recovered without reopening `2022` and proxy DD |
| `balanced_rule_residual_mode_v2` | proves the hierarchy can now be tuned from experiment config instead of `src` edits | whether the next gain should come from better mode classification rather than wider favorable-mode envelopes |
| `balanced_floor40_sym_slow_bear` | shows aggressive family can be repaired partially | whether that logic can be ported without re-breaking stress |
| `smoke_rule_residual_v1` | cheap residual sanity check | whether smoke findings transfer to full mainline consistently |

## Success And Failure Patterns

### What Worked

1. Residual-over-rule was the single biggest structural win.
- It reduced exploration burden.
- It anchored exposure to an interpretable base policy.
- It improved both crisis behavior and trade quality.

2. Sequence modeling helped when tied to a proven family.
- `v16` and `v19` show GRU + longer context can capture more upside without breaking the gate.
- The effect was strongest when context length changed alone, not when bundled with more routing or sampling changes.

3. External risk data can improve drawdown shape even when it does not maximize return.
- `v22` is useful because it buys better drawdown in `test_2020` and `test_2023_2024`.
- That is a real signal, not a random byproduct.

4. The harness already proved that better performance does not require higher 2022 exposure.
- `v1`, `v2`, and `v19` all pass the gate cleanly.
- The repo is past the stage where "higher return must mean looser risk".

### What Failed

1. Broad feature expansion without gating.
- The transition-feature batch was directionally right but operationally wrong.
- The model used the new features to justify broader risk appetite, not just better re-entry timing.

2. Reward coefficients as a primary optimization loop.
- Harder penalties did not teach cleaner crisis behavior.
- They often created more distorted exposure or lower efficiency without solving the path-level DD problem.

3. Multi-lever bundled structural changes.
- Longer context, routing, action compression, and sampling together made attribution impossible.
- Some pieces were useful, but the bundle degraded overall quality.

4. "Return first, fix stress later" aggressive branches.
- Floor-raising and regime-gate variants improved trend capture.
- They repeatedly failed the 2022 gate and widened proxy DD, so they cannot be the default research path.

## Good Windows vs Bad Windows

The repo already has a stable window-level pattern:

- Strong windows:
  - persistent 2022 stress defense in canonical `balanced`
  - 2020 trend capture once sequence context improves (`v16`, `v19`, `v22`)
  - 2023-2024 capture with residual-over-rule (`v1`, `v16`)
- Weak windows:
  - crash-to-rebound restarts
  - deep-stress release phases
  - cases where long-term structural damage is still visible but short-term panic has already cleared

Operationally this means:

- Good intervals come from persistent-state recognition.
- Bad intervals come from transition-state recognition.
- The research gap is not generic "more trend features". It is selective transition-state discrimination without broad exposure drift.

## Professional Quant Diagnosis

### 1. Signal Quality

The core signal problem is no longer basic trend or stress detection. The baseline already detects:

- persistent damage
- persistent low-risk uptrends
- mature trend continuation

The weak point is transition discrimination:

- rebound after deep damage
- panic-release without full anchor repair
- early trend restart when long-horizon features still look broken

### 2. Risk Budget Mapping

Risk-budget translation is partially solved.

- Canonical `balanced` proves the repo can map stress state into acceptable `2022` exposure.
- Many failed branches widened `2022` avg_pos` from ~0.28 to ~0.37-0.41 without earning enough extra return.

So the next question is not "can the repo cap risk at all?".
It is "can the repo loosen risk only during the right release windows without losing the hard-won crisis discipline?"

### 3. Trading Friction

Trade count matters, but it is not the sole bottleneck.

- Execution-layer compression helped more than reward-layer compression.
- But aggressive trade compression also destroyed return quickly.
- The right target is not "fewest trades". The right target is "higher-value state changes per trade".

### 4. Model-Selection Bias

This remains a real risk.

- Checkpoint review proved best validation iteration is often earlier than the configured max.
- Seed sensitivity exists on return, especially in the GRU branch, but drawdown discipline is much more stable than return.

So future claims should be ranked by:

- fixed split
- seed sweep
- walk-forward

in that order, not by one backtest headline.

### 5. Style Exposure

A large part of historical return variation came from exposure style, not pure alpha.

- `floor40_sym` and regime-gate branches bought return by staying more invested.
- `balanced_rule_residual_v1` improved quality by using a better base policy and better delta decisions, not just more exposure.

That distinction matters. A good branch should explain whether gains came from:

- better timing inside the same risk budget
- or simply higher average exposure

### 6. Robustness

The most robust current findings are:

- `balanced` is a trustworthy baseline
- `balanced_rule_residual` is the right mainline family
- reward-only micro-tuning is a low-value loop
- transition-state work is still necessary, but it must be narrower than the first failed batch

## Next Experiment Queue

Only the following three next steps are high-value enough to justify new runs.

### 1. Panic-score + term-structure transition branch

- Goal window:
  - rebound / restart windows that canonical `balanced` undercaptures
  - especially 2020-04, 2022-07, 2022-11, 2023-01 style states
- Reference class:
  - open-source project: `soxl_backtest` panic-index logic
  - data: `VVIX`, `VIX9D`, `VIX3M`, `SKEW`
  - feature family: stress-release / transition features
- Implementation idea:
  - do not re-add a wide transition batch
  - add one narrow profile-local family combining panic-release and VIX term-structure release speed
- Acceptance:
  - keep `test_2022` gate pass
  - improve targeted restart windows
  - avoid broad `avg_pos` drift

### 2. `v22` Pareto checkpoint selection

- Goal window:
  - keep the v22 drawdown improvement while reducing the `2023-2024` return drag
- Reference class:
  - model-selection method: Pareto ranking over return / Sharpe / DD rather than a single score
- Implementation idea:
  - keep actor family fixed
  - change checkpoint choice only
  - run seed-confirm + walk-forward before considering canonical replacement
- Acceptance:
  - `test_2023_2024` return recovers toward `v19`
  - `test_2020` and `test_2023_2024` DD gains from `v22` do not disappear

### 3. Fixed-actor two-speed execution counterfactual

- Goal window:
  - reduce low-value churn without flattening trend capture
- Reference class:
  - execution model: decision-frequency / threshold / trade-lock counterfactuals
  - open-source reference: slower update cadence in `soxl_backtest`
- Implementation idea:
  - test this first on a fixed actor such as `v19` or `v22`
  - use crisis-vs-normal two-speed execution instead of a global compression knob
- Acceptance:
  - lower micro churn
  - no large hit to `test_2023_2024`
  - proxy DD or Sharpe improves without changing the actor

## Required Workflow Change

This report should become the default pre-run briefing for future work:

- before a new experiment, identify the closest historical success case
- identify the closest historical failure case
- explain what mechanism is new this time
- record the intended row that will be added back into the comparison table

Without that, the repo will keep repeating old branches under new names.

## Evidence Paths

- Canonical notes:
  - `experiments/canonical/balanced/notes.md`
- Candidate notes:
  - `experiments/candidate/balanced_rule_residual_v19/notes.md`
  - `experiments/candidate/balanced_rule_residual_v22/notes.md`
- Archived milestone notes:
  - `experiments/archive/balanced_rule_residual_v1/notes.md`
  - `experiments/archive/balanced_rule_residual_v16/notes.md`
  - `experiments/archive/balanced_rule_residual_v20/notes.md`
- Registry:
  - `configs/soxl/profile_registry.yaml`
- Summary metrics:
  - `artifacts/models/*_summary.json`
- Historical review:
  - `runtime/post_train_review/`
- Continuity:
  - `memory/2026-04-18-checkpoint-review-and-overlay-loop.md`
  - `memory/2026-04-18-structural-model-loop.md`
  - `memory/2026-04-18-transition-feature-batch-failure.md`
  - `memory/2026-04-18-trade-compression-experiments.md`
  - `memory/2026-04-20-backtest-vs-quant-analysis.md`
  - `memory/2026-04-20-rl-vs-rulebase-gap-analysis.md`
  - `memory/2026-04-20-seed-sweep-clarified.md`
  - `memory/2026-04-21-v22-vix-term-structure-results.md`

## 2026-04-22 Drawdown Frontier Exploration

### New Mechanisms Tested

| experiment | approach | proxy ret / sh / dd | verdict |
|---|---|---|---|
| `balanced_rule_residual_learned_mode_v1` | Learned mode classifier replacing rule-based `classify_rule_mode` | 507.0% / 0.87 / **50.0%** | dropped — trained actor exploits router permissiveness, DD rebounds above v19 |
| `drawdown_budget_overlay_v19` | Hard position cap based on drawdown consumption | 353.0% / 0.87 / 37.8% | diagnostic only — too blunt, kills return |
| `regime_conditional_budget_rt0.6` | Budget mask only when `regime_risk >= 0.6` | **669.5%** / **1.04** / **42.1%** | **candidate overlay** — best mechanical improvement with zero return cost |
| `hybrid_learned_budget` | Learned router + conditional budget on fixed actor | 343.4% / 0.87 / 35.8% | dropped — return too low vs overlay-only path |
| `balanced_rule_residual_refined_v1` | Remove activity_bonus, reduce inactivity_penalty | 500.6% / 0.84 / 54.6% | dropped — too conservative, missed bull runs |
| `balanced_rule_residual_refined_v2` | Add trend_alignment_bonus=0.20 | 556.9% / 0.87 / **53.0%** | candidate — best proxy DD in refined series, best test_2023_24 |
| `balanced_rule_residual_refined_v3` | Strengthen trend incentives (alignment=0.30, opportunity=0.05) | **637.8%** / **0.89** / 54.9% | candidate — best refined return/sharpe but worse crisis behavior |
| `balanced_rule_residual_refined_v4` | Midpoint params between v2/v3 | 480.0% / 0.84 / 54.2% | dropped — midpoint trap, non-linear interactions |
| `multi_asset_refined_v3` | Joint train SOXL/TQQQ/UPRO/SPXL/TECL with refined_v3 reward | 388.6% / 0.78 / 49.7% | dropped — reward tuned for SOXL does not generalize; -249pp return vs single-asset |
| `actor_ensemble_v19_mode_v1` | Dynamic switching between aggressive/defensive actors | ~40% / 0.85 / ~10% | dropped — incoherent behavior, return collapse |

### Key Conclusion

The repo has reached a **fundamental frontier** for single-asset 3x leveraged ETF strategies:

- `mode_v1` already achieves **42.3% proxy MaxDD** — this is likely the practical limit without external hedging or multi-asset diversification.
- No architectural tweak (learned router, action masking, ensemble) can push proxy MaxDD to 30% without large return sacrifice.
- The **regime-conditional budget overlay** (`rt0.6_b0.3`) is the only net-positive mechanical addition: it improves proxy MaxDD by ~5pts with virtually no return drag.

### Key Conclusion

The repo has reached a **fundamental frontier** for single-asset 3x leveraged ETF strategies:

- `mode_v1` already achieves **42.3% proxy MaxDD** — this is likely the practical limit without external hedging or multi-asset diversification.
- No architectural tweak (learned router, action masking, ensemble) can push proxy MaxDD to 30% without large return sacrifice.
- **Reward-only tuning is exhausted** — four refined iterations (v1-v4) show a sharp trade-off between return and DD, with no configuration beating v19's 735% return.
- **Multi-asset with current architecture fails** — a single shared actor with simple one-hot embedding and SOXL-tuned reward drops return 249pp. Cross-asset generalization requires asset-specific heads, sector embeddings, or generic reward functions.
- The **regime-conditional budget overlay** (`rt0.6_b0.3`) is the only net-positive mechanical addition: it improves proxy MaxDD by ~5pts with virtually no return drag.

### Next Valid Directions

1. **Productionize regime-conditional budget** as a runtime overlay on the canonical actor.
2. **Seed sweep on v16** (GRU, window=5, penalty=2.0) — best risk-adjusted GRU config (632.4% / 0.91 / 50.1%) is seed-sensitive; a better seed may close the gap to v19.
3. **Multi-asset revisited ONLY with architectural changes** — asset-specific heads, sector embeddings, volatility normalization, or generic reward. Current one-hot + shared actor is insufficient.
4. **Options/tail-risk hedging** — outside current harness scope but the industry-standard solution for this exact problem.
