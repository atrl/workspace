---
date: 2026-04-19
title: 2026-04-19 post-train Kelly skill patch
kind: skill
profiles: []
status: kept
tags:
- execution-overlay
- kept
- skill
artifacts:
- runtime/diagnosis/phase1_phase2_combined.json
- runtime/diagnosis/soxl_trade_attribution.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Patched `.codex/skills/post-train-backtest-review/` so post-train review
  now covers negative Kelly / Expectancy diagnosis, reward-alignment interpretation,
  fixed-actor counterfactual order, and optional external analysis stack boundaries.
  Validation used the existing trade-attribution artifacts plus skill quick-validate.
---

# 2026-04-19 post-train Kelly skill patch

## Context

The post-train review skill already handled summary / slice / feature comparisons, but it did not standardize what to do when users asked why a model still produced negative-expectancy T trades.

## What changed

- Extended `.codex/skills/post-train-backtest-review/SKILL.md`.
- Added a dedicated `Trade-Quality And Negative-Kelly Audit` section.
- Added repo-specific guidance for mapping negative trade buckets back to:
  - reward design
  - execution thresholds
  - trade compression
  - fixed-actor counterfactual order
- Added `Optional External Analysis Stack` guidance:
  - `vectorbt`
  - `QuantStats`
  - `pyfolio`
  - `Qlib`

## Validation evidence

- Verified the new workflow against existing repo artifacts:
  - `runtime/diagnosis/phase1_phase2_combined.json`
  - `runtime/diagnosis/soxl_trade_attribution.md`
- Skill structure validation:
  - `python3 /Users/rziiiii/.codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/post-train-backtest-review`

## Why it matters

- The repo now has a standard answer for:
  - what negative `Kelly` means
  - what reward the model likely learned instead
  - when to blame signal vs execution vs reward alignment
  - when to do fixed-actor overlays before retraining
