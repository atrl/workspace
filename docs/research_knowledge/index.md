# Research Knowledge Index

This directory is the task-direction router for SOXL and related leveraged-ETF strategy work. Start here before opening raw memory notes, runtime artifacts, or experiment packages.

## First Decision: What Kind Of Task Is This?

| Direction | Knowledge | Skills | Use when |
|---|---|---|---|
| Factor | `factors.md` | `leveraged-etf-factor-research`, `leveraged-etf-factor-mining`, `leveraged-etf-signal-backtest` | feature hypotheses, IC, rulebase-derived signals, independent signal validation |
| Data | `data.md` | `qlib-rl-trading` | raw CSVs, qlib stores, processed panels, training caches, external volatility inputs |
| Model selection | `model_selection.md` | `post-train-backtest-review` | checkpoints, promotion gates, fixed splits, early stopping, actor retention |
| Network | `networks.md` | `qlib-rl-trading`, `post-train-backtest-review` | GRU, attention, MoE, mode classifiers, temporal context, asset-specific heads |
| Reward | `rewards.md` | `rl-kelly-integration`, `post-train-backtest-review` | reward penalties, Kelly, imitation, activity, opportunity cost, trade quality |
| Execution | `execution_overlays.md` | `post-train-backtest-review`, `dashboard` | fixed-actor overlays, position budgets, mode routing, dashboards, runtime mapping |
| Multi-asset | `multi_asset.md` | `qlib-rl-trading`, `soxl-signal-backtest` | TQQQ/UPRO/SPXL/TECL transfer tests or shared leveraged-ETF training |

Machine-readable route: `task_router.json`.

## Retrieval Rule

1. Pick exactly one primary direction from the table above.
2. Read that direction's knowledge file.
3. Invoke the listed skill before changing code or running a new experiment.
4. Query historical context from the relevant `runtime/<direction>/knowledge.md`.
5. Open runtime artifacts only from the route's `runtime` paths.
6. Record the chosen direction in `codex_exec_plans/active/current.md` before execution.

## Memory Role

`runtime/<direction>/knowledge.md` is not a starting point. It is evidence retrieval behind the selected direction. Do not start by reading broad runtime knowledge files; start from `task_router.json`, then query the relevant `runtime/<direction>/knowledge.md` for the chosen direction.

## Canonical Anchors

- `balanced` remains the control baseline.
- `balanced_rule_residual` is the successful residual family.
- `configs/soxl/profile_registry.yaml` remains the profile lifecycle source of truth.
- `runtime/` remains generated research evidence; do not move runtime logs into experiment cards.
