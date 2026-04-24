---
date: '2026-04-17'
title: '2026-04-17: Post-train backtest review skill'
kind: skill
profiles:
- balanced
- balanced_floor40_sym
- balanced_floor40_sym_bear30
- soxl
status: active
tags:
- active
- balanced
- balanced_floor40_sym
- balanced_floor40_sym_bear30
- checkpoint-selection
- eval-2022
- skill
- soxl
- stress-regime
- walk-forward
artifacts:
- runtime/model/reviews/floor40_sym_vs_balanced_moe.md
workpad: memory/index.json
summary: 'Legacy memory note normalized into the repo-wide schema. Added repo-local
  skill `.codex/skills/post-train-backtest-review/`. The skill standardizes how to
  compare a new training result against a prior backtest using: `artifacts/models/*_sum'
---

# 2026-04-17: Post-train backtest review skill

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What changed

- Added repo-local skill `.codex/skills/post-train-backtest-review/`.
- The skill standardizes how to compare a new training result against a prior backtest using:
  - `artifacts/models/*_summary.json`
  - optional feature summaries
  - promotion checks and slice-level evidence
- Added `scripts/compare_backtests.py` to generate a Markdown post-train review from two summary JSON files.

## Why it matters

- The repo was repeatedly doing the same qualitative comparison in chat and memory files:
  - what got better
  - what got worse
  - which windows improved or regressed
  - whether the change came from higher exposure / weaker de-risking / earlier checkpoint peak
- This is now a reusable workflow instead of session-specific narration.

## Validation evidence

- Skill validation passed via:
  - `python3 /Users/rziiiii/.codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/post-train-backtest-review`
- Sample report generated at:
  - `runtime/model/reviews/floor40_sym_vs_balanced_moe.md`

## External references baked into the skill

- Bailey et al. - Probability of Backtest Overfitting
- Lopez de Prado - Deflated Sharpe Ratio
- AQR - out-of-sample-oriented model selection
- Man Group - discounting backtest Sharpe due to data mining
- AAMDRL - walk-forward analysis for DRL

## Follow-up

- If feature analysis becomes checkpoint-aware rather than only "latest model", wire those outputs into the same review loop using `--latest-features` and `--previous-features`.
- The skill now also contains a repo-specific improvement roadmap:
  - optimize canonical `balanced` first
  - treat `balanced_floor40_sym` as an aggressive branch
  - prefer stress-window de-risking, trade compression, and better checkpoint selection before longer training
- The skill was made more generic after review:
  - added a universal quant optimization framework
  - added a `counterfactual backtest before retrain` rule
  - kept SOXL-specific priority as a secondary snapshot, not the main body
- Verified the generic rule on this repo:
  - fixed the latest `balanced_floor40_sym` actor and swept `bull_min_cap` / `bear_min_cap` / `bear_regime_threshold`
  - found that bull-floor changes barely move metrics, while tighter `bear_min_cap` modestly improves `2022` drawdown and `avg_pos`
- Patched the skill again after review:
  - added a `Quant Reality Audit` section:
    - leak / future-data checks
    - training-vs-backtest cost alignment
    - transaction-cost sensitivity
    - execution realism and impact/capacity notes
  - added an `Auto-Optimization Rule`:
    - default order is counterfactual -> realism audit -> real retrain
    - next experiment is chosen by evidence, not by a generic suggestion list
  - added `Continuity Discipline`:
    - default to updating the active workpad
    - only write `memory/` when the result is cross-session durable
- Added reusable harness support for the skill:
  - `scripts/cost_sensitivity_backtest.py`
  - `compare_backtests.py --latest-cost-audit ...`
- Immediate evidence from the new audit:
  - `balanced_floor40_sym_bear30` is more friction-robust than `balanced_floor40_sym`
  - canonical `balanced` currently has a replayability gap:
    - saved actor bundle is `56` dims
    - current feature set expects `64`
    - so the baseline summary remains useful, but the actor artifact itself needs repair before full realism audits

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
