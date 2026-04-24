# EXECUTION Knowledge

_Auto-generated from memory/ integration on 2026-04-24._
_Total entries: 15._

##  — Memory (`dropped`)
`memory/` is the continuity layer for durable decisions, failures, reviews, and reusable session context that should survive chat history. Do not scan all notes by hand first. Read in this order: 1. `memory/index.json` 2. `qlib_venv/bin/pyt

## 2026-04-16 — 2026-04-16: MoE Regime Analysis & Gap to Rule-Based Backtest (`dropped`)
Legacy memory note normalized into the repo-wide schema. Evaluated the multi-asset v2 actor (`soxl_balanced_moe_actor.pt`) on the SOXL test set. Extracted gate softmax probabilities for every trading day. Performed an **ablation study** by

## 2026-04-17 — 2026-04-17: Dashboard window clarified, regime-routed overlay is directional but not best (`dropped`)
Legacy memory note normalized into the repo-wide schema. `runtime/execution/generalization/dashboard.html` only renders the fixed `test` window: `2020-01-02 .. 2026-03-31` The page does **not** show: `train = 2010-2018` `valid = 2019` The earlier imp

## 2026-04-19 — App And Policy Consolidation (`reference`)
Consolidated real train/evaluate/promote ownership under app/, moved execution/observation/positioning implementations under policy/, deleted top-level rl_* helper modules, and tightened runtime/memory/experiments index guidance.

## 2026-04-19 — CLI Shell Shrink (`reference`)
Split runtime bootstrap and end-of-run reporting out of app/cli.py so the CLI entrypoint is a thin orchestration shell rather than a mixed setup/execution/reporting file.

## 2026-04-19 — 2026-04-19 residual-over-rule smoke loop (`kept`)
Implemented and smoke-tested a residual-over-rule path. The actor now learns a bounded delta around the rule-based base position, with rule-deviation and micro-churn penalties. The first smoke candidate materially improved the previously negative `3-10d` trade bucket to positive Kelly/expectancy and reduced total trade actions from the prior canonical diagnosis baseline, but `micro_<5%` rebalances remain weak and too frequent.

## 2026-04-20 — v4 Results: Trade Compression Fails Again — Trades Increase to 327 (`kept`)
Legacy memory note normalized into the repo-wide schema. v4 training completed. Despite aggressive trade compression design (flat penalty 1.20, longer lock 8 days, reduced alpha 2.0), **trades increased from 314 to 327**. Strategy is highly

## 2026-04-21 — 2026-04-21 dashboard slimming and next-step overlay order (`active`)
Slimmed `runtime/execution/generalization/dashboard.html` by changing historical `versioned` payloads to carry only compact trading-series rows and reusing current per-ETF market metadata client-side by date. Dashboard size fell from 50.54 MB to 20.41 MB. Mainline optimization order remains fixed-actor execution/risk overlay before any new retraining branch.

## 2026-04-21 — 2026-04-21 existing overlay gap review (`active`)
Reviewed the existing fixed-actor execution overlay sweep before starting any new overlay work. The repo already validated the generic overlay direction with `regime_route_v1`, but on SOXL it did not materially improve the `2022` bear window and also gave up recovery return. Any next overlay loop that only retunes stress/bull thresholds would be duplicate work; the next valid overlay mechanism must introduce an explicit release/re-entry state.

## 2026-04-21 — 2026-04-21 v19 dwell overlay diagnostic (`reference`)
Base actor: `balanced_rule_residual_v19` Motivation: `v24` showed that lower-frequency coarse retraining did not reduce realized churn, so the next cheap question was whether the missing ingredient was execution-side persistence. `dwell_hol

## 2026-04-21 — 2026-04-21 v19 Dwell-Stress Merge Review (`active`)
Follow-up to the `v19` fixed-actor overlay frontier review. The goal was to test one narrow merge only: keep the low-frequency persistence behavior of `dwell_hold_v2`, then add only the stress-side asymmetry that had helped `regime_route_v1

## 2026-04-21 — 2026-04-21 v19 overlay promotion review (`kept`)
Base actor: `balanced_rule_residual_v19` Goal: compare the new dwell overlays against the existing SOXL overlay set on the same fixed-split promotion surface 1. `dwell_hold_v2` 2. `dwell_hold_v1` 3. `regime_route_v1` 4. `stress_release_rout

## 2026-04-22 — 2026-04-22 balanced_rule_residual_mode_v1 training result (`reference`)
First full train of the mode-hierarchy scaffold over the retained `balanced_rule_residual_v19` family. This was the first attempt to add persistent rule-routed modes directly into train/eval/backtest/runtime instead of doing another overlay

## 2026-04-22 — 2026-04-22 Mode Hierarchy Scaffold Start (`active`)
The repo hit a clear optimization plateau on threshold, cooldown, and overlay retuning. The next step moved to a larger mechanism change: persistent high-level trading modes constraining the existing residual actor, rather than further sing

## 2026-04-23 — Position Budget Productionization Smoke (`reference`)
Config-owned regime-conditional position budget plumbing was added across evaluation, qlib backtest, runtime, and training action interpretation. Smoke seed=12345 ran end-to-end but failed strict drawdown promotion; keep the infrastructure and candidate scaffold, do not promote the smoke actor.
