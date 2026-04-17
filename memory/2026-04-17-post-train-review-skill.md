# 2026-04-17: Post-train backtest review skill

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
  - `runtime/post_train_review/floor40_sym_vs_balanced_moe.md`

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
