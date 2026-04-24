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

## Required Workflow

Before starting any new run:

1. Read the relevant file in `docs/research_knowledge/`.
2. Query `memory/index.json` with `scripts/memory_search.py`.
3. Record closest success, closest failure, new mechanism, target window, accepted trade-off, and comparison-table row key in the active workpad or experiment card.
4. Add or update a master table row when the loop completes.

## Evidence Paths

- Historical narrative: `docs/research_knowledge/experiment-landscape-review-legacy.md`
- Memory index: `memory/index.json`
- Profile registry: `configs/soxl/profile_registry.yaml`
- Runtime reviews: `runtime/model/reviews/`
- Generalization evidence: `runtime/execution/generalization/`
