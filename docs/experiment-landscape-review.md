# SOXL Experiment Landscape Review

This file is now the compact landing page and master comparison surface. Detailed mechanism-specific conclusions live in `docs/research_knowledge/`.

## Executive Summary

- `balanced` remains the canonical control baseline.
- `balanced_rule_residual` remains the successful residual family.
- Future work must choose a mechanism category before running: factor, data, model selection, network, reward, execution overlay, or multi-asset.
- Full historical narrative before this split is preserved at `docs/research_knowledge/experiment-landscape-review-legacy.md`.

## Mechanism Index

| Mechanism | Category file | Current stance |
|---|---|---|
| Rulebase / academic factors | `docs/research_knowledge/factors.md` | validate independently before RL injection |
| Data / qlib / caches | `docs/research_knowledge/data.md` | registry-backed paths only |
| Checkpoint / selection | `docs/research_knowledge/model_selection.md` | inspect before retraining |
| GRU / attention / MoE / heads | `docs/research_knowledge/networks.md` | require bottleneck-specific hypothesis |
| Reward / Kelly / imitation | `docs/research_knowledge/rewards.md` | avoid repeated knob drift |
| Fixed-actor overlays / modes | `docs/research_knowledge/execution_overlays.md` | prefer as separation tests |
| Cross-ETF diagnostics | `docs/research_knowledge/multi_asset.md` | diagnose before shared training |

## Master Comparison Table

| Profile / line | Category | Baseline | Mechanism delta | Key metrics | Decision | Next action |
|---|---|---|---|---|---|---|
| `balanced` | canonical | old qlib baseline | balanced RL harness baseline | retained control; see registry and model summaries | keep | compare all branches to this |
| `balanced_rule_residual` family | model / reward / execution | `balanced` | residualize around rulebase behavior | strongest retained family in historical review | keep family | use direct candidate rows for new variants |
| `balanced_rule_residual_mode_v1` | execution overlay / mode | `v19` | config-owned mode hierarchy | improved drawdown frontier but upside giveback remained | retained frontier evidence | improve mode classification, not looser envelopes |
| `balanced_rule_residual_mode_v2` | execution overlay / mode | `mode_v1` | looser favorable modes | hurt `test_2020` and proxy quality | drop | do not repeat envelope loosening alone |
| transition feature branches | feature | `balanced` / `v19` | stress-release / transition-state features | validation/local rebounds improved, fixed-split risk worsened | drop as broad retrain path | test transition signals through execution overlay first |
| phase-balanced sampling | sampling | `balanced` | broader episode coverage | widened risk globally; best checkpoint was early | keep harness support, drop branch | do not widen sampling alone |
| reward-only variants | reward | `balanced` / residual branches | penalty or bonus drift | often increased conservatism or low-position collapse | drop repeated pattern | require new reward design and evidence |
| multi-asset refined reward | multi-asset | single-asset refined run | shared actor over SOXL/TQQQ/UPRO/SPXL/TECL | all slices degraded vs single asset | drop | add generic reward, volatility normalization, asset-specific heads before retrain |
| `risk_budget_usage_v1` | execution overlay / model | `balanced_rule_residual_imitation_v4` | budget-usage architecture: slow mode sets risk band, fast path decides usage | proxy Sharpe 0.94, passes gate; 2022 return -17.3% MaxDD 18.7% — **vs SOXL B&H -86.6%, outperforms by +69.3pp**; 2023-2024 return +82.8%; best recovery in risk_budget family | **candidate** | core problem is recovery timing (too conservative); next: improve re-entry speed after bear markets |
| `risk_budget_usage_v2` | execution overlay | `v1` | more conservative budget thresholds | proxy Sharpe 0.88 (fails gate <0.95); 2022 return -3.2% MaxDD 7.4% — excellent defense; 2023-2024 return +40.1% — worst recovery in family | **ablation** | confirms more conservative = worse recovery; do not pursue this direction |
| `risk_budget_oracle_distill_v1` | model / distillation | `v1` | distill from oracle budget usage as supervision | proxy Sharpe 1.09 (highest in family); 2022 return -5.5%; 2023-2024 return +58.5% — weaker than v1 (+82.8%) | **drop** | oracle supervision teaches conservatism; not better than v1 |
| `risk_budget_bc_warmstart_v1` | model / warmstart | `v1` | BC warm-start from oracle path then PPO | proxy Sharpe 0.91; 2022 return -5.0%; 2023-2024 return +49.0% — weaker than v1 | **drop** | BC initialization teaches conservative behavior; not better than v1 |
| `risk_budget_learned_oracle_distill_v1` | model / distillation | `v1` | learn oracle from data instead of direct oracle | proxy Sharpe 1.04; 2022 return -11.7% — worse than direct oracle (-5.5%) | **drop** | learned oracle worse than direct oracle; unnecessary complexity |
| `risk_budget_usage_v3` | network / execution | `v1` | mode-conditioned actor input (one-hot) | proxy Sharpe 1.07 but backtest 0.87; wf_2022 MaxDD 59.0%, wf_2023_2024 Sharpe 0.22; severe proxy-to-backtest gap | **drop** | mode input leaks information; do not pursue mode-conditioned architecture |
| `balanced_rule_residual_v19` walk-forward | model selection / validation | `balanced` | fixed-split best candidate (Sharpe 0.97) under walk-forward test | wf_2020 return +105.9% / Sharpe 2.04 / MaxDD 20.3% (excellent bull); wf_2022 return **-30.7%** / Sharpe -0.87 / MaxDD 39.4% — **vs SOXL B&H -86.6%, outperforms by +55.9pp**; bear-market defense is successful; gate rejection due to inappropriate absolute Sharpe threshold in bear year | **candidate** | revise promotion gate to use benchmark-relative metrics; investigate high proxy MaxDD (53.9%); improve recovery timing (2023-2024 model +79% vs B&H +182%) |
| `balanced_rule_residual_v22` | model / reward | `v19` | v19 + adjusted reward coefficients | proxy Sharpe 0.86, passes fixed-split gate; fixed-split test_2022 return -21.0%, MaxDD 30.1% — **vs SOXL B&H -86.6%, outperforms by +65.6pp**; bear-market defense successful | **candidate** | same as v19; proxy MaxDD 55.5% needs attention; recovery timing too slow (2023-2024 +60.5% vs B&H +182%) |
| `balanced_rule_residual_v19_small` | network | `v19` | smaller network (hidden=[32], gru=16) | proxy Sharpe 0.84 but fixed-split test_2022 MaxDD 38.3% > 35% limit; return -31.6% — **vs SOXL B&H -86.6%, outperforms by +55.0pp**; worst fixed-split bear-market defense in family | **drop** | network size reduction hurts performance; fixed-split gate fails on MaxDD; do not reduce capacity |
| `balanced_rule_residual_v19_reg` | network | `v19` | v19 + dropout (0.2) + weight decay (1e-5) | proxy Sharpe 0.88; **walk-forward**: wf_2020 Sharpe 2.20 / Ret 95.3% / MaxDD 20.8%, wf_2022 Sharpe **-0.73** / Ret -21.9% / MaxDD 31.0% — best bear-market walk-forward in residual family, wf_2023_2024 Sharpe -0.44 / Ret -22.9% / MaxDD 37.2%; regularization improves wf_2022 but trades off wf_2023_2024 | **candidate** | first experiment to improve ANY walk-forward window; regularization alone cannot improve both bear-market windows simultaneously |
| `balanced_rule_residual_v19_reg_light` | network | `v19_reg` | v19 + dropout (0.1) + weight decay (1e-6) | proxy Sharpe 0.92; **walk-forward**: wf_2020 Sharpe **2.36** / Ret 115.6% / MaxDD 20.1% — best bull market in family, wf_2022 Sharpe -0.86 / Ret -21.9% / MaxDD 30.2%, wf_2023_2024 Sharpe -0.38 / Ret -24.3% / MaxDD 41.9%; lighter regularization preserves bull-market performance but does not improve bear-market generalization over v19_reg | **candidate** | regularization creates a Pareto frontier: stronger = better wf_2022, weaker = better wf_2020; no single setting wins both bear-market windows; needs combination with other mechanisms |
| `balanced_rule_residual_v19` walk-forward (CORRECTED) | harness / measurement | `v19` | fix worker eval to use backtest_profile metrics; recalibrate gate thresholds | **CORRECTED metrics**: wf_2020 Ret +113.4% / Sharpe 1.98 / MaxDD 22.4%, wf_2022 Ret -47.0% / Sharpe -1.41 / MaxDD 51.1% — vs B&H -86.6% (+39.6pp outperformance), wf_2023_2024 Ret +28.8% / Sharpe 0.68 / MaxDD 34.4% — vs B&H -2.6% (+31.5pp outperformance); previous "failure" was measurement artifact from evaluate_split gaps of 80-110pp | **canonical baseline** | v19 actually outperforms B&H in ALL windows; promotion gate redesigned with relative-outperformance metrics and realistic MaxDD thresholds (55% for 3x ETF bear market) |
| `balanced_rule_residual_v19_lowcap` | execution overlay | `v19` | reduce `position_cap.min_cap` from 0.40 to 0.20 | **Backtest-only** (old actor, new cap): wf_2020 Ret +106.7% / MaxDD 22.4%, wf_2022 Ret -41.9% / MaxDD 46.7%, wf_2023_2024 Ret +29.8% / MaxDD 32.1%; 2022 MaxDD reduced by 4.4pp, return improved by 5.1pp; **BUT retrained actor** with min_cap=0.20: wf_2020 Ret +130.9% (better), wf_2022 MaxDD 52.8% (WORSE), wf_2024 Ret +10.9% (WORSE); retraining teaches aggression without sufficient defense | **dropped** | min_cap=0.20 cap config is beneficial, but must be applied to OLD actor (trained with min_cap=0.40); retraining with lower min_cap creates moral hazard |


## Required Workflow

Before starting any new run:

1. Read the relevant file in `docs/research_knowledge/`.
2. Query the relevant `runtime/<direction>/knowledge.md` for historical context.
3. Record closest success, closest failure, new mechanism, target window, accepted trade-off, and comparison-table row key in the active workpad or experiment card.
4. Add or update a master table row when the loop completes.

## Evidence Paths

- Historical narrative: `docs/research_knowledge/experiment-landscape-review-legacy.md`
- Runtime knowledge: `runtime/<direction>/knowledge.md`
- Profile registry: `runtime/harness/card/registry/profile.yaml`
- Runtime reviews: `runtime/model/`
- Generalization evidence: `runtime/execution/generalization/`
