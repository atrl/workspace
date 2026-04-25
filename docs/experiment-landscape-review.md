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
| `balanced_rule_residual_v19` walk-forward | model selection / validation | `balanced` | fixed-split best candidate (Sharpe 0.97) under walk-forward test | wf_2020 Sharpe 2.35 (excellent bull); wf_2022 Sharpe -0.80, return -25.2%, MaxDD 35.2% — **vs SOXL B&H -86.6%, outperforms by +61.4pp**; bear-market defense is successful; gate rejection due to inappropriate absolute Sharpe threshold in bear year | **candidate** | revise promotion gate to use benchmark-relative metrics; investigate high proxy MaxDD (53.9%); improve recovery timing (2023-2024 model +79% vs B&H +182%) |
| `balanced_rule_residual_v22` | model / reward | `v19` | v19 + adjusted reward coefficients | proxy Sharpe 0.86, passes fixed-split gate; fixed-split test_2022 return -21.0%, MaxDD 30.1% — **vs SOXL B&H -86.6%, outperforms by +65.6pp**; bear-market defense successful | **candidate** | same as v19; proxy MaxDD 55.5% needs attention; recovery timing too slow (2023-2024 +60.5% vs B&H +182%) |
| `balanced_rule_residual_v19_small` | network | `v19` | smaller network (hidden=[32], gru=16) | proxy Sharpe 0.84 but fixed-split test_2022 MaxDD 38.3% > 35% limit; return -31.6% — **vs SOXL B&H -86.6%, outperforms by +55.0pp**; worst fixed-split bear-market defense in family | **drop** | network size reduction hurts performance; fixed-split gate fails on MaxDD; do not reduce capacity |
| `balanced_rule_residual_v19_reg` | network | `v19` | v19 + dropout (0.2) + weight decay (1e-5) | proxy Sharpe 0.88; **walk-forward (main backtest)**: wf_2020 +181.3% / 0.59 / 76.7%, wf_2022 +42.2% / 0.34 / 61.6%, wf_2023_2024 -7.1% / 0.05 / 43.4%; vs v19 baseline: wf_2022 v19=+67.6% > v19_reg=+42.2%; **critical finding**: worker internal eval (-23.9%) differs from backtest (+42.2%) by 66pp | **candidate** | regularization does NOT improve walk-forward; worker eval is not a reliable predictor of backtest; promotion blocked on all models due to high MaxDD (61-77%) |


## Required Workflow

Before starting any new run:

1. Read the relevant file in `docs/research_knowledge/`.
2. Query `memory/_archive_index.json` with `runtime/<direction>/knowledge.md`.
3. Record closest success, closest failure, new mechanism, target window, accepted trade-off, and comparison-table row key in the active workpad or experiment card.
4. Add or update a master table row when the loop completes.

## Evidence Paths

- Historical narrative: `docs/research_knowledge/experiment-landscape-review-legacy.md`
- Memory index: `memory/_archive_index.json`
- Profile registry: `configs/soxl/profile_registry.yaml`
- Runtime reviews: `runtime/model/reviews/`
- Generalization evidence: `runtime/execution/generalization/`
