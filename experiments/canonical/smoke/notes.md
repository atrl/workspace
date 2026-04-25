---
title: smoke
direction: model
status: candidate
---

# Smoke

Canonical harness smoke profile.

- Purpose: cheap end-to-end sanity for train/eval/backtest pipeline
- Baseline: `smoke`
- Retention: keep actor because this is the fastest reproducibility anchor

## Expectations

# Expectations

- Must stay cheap and stable
- Used for harness and replay sanity, not for model-quality claims
- Any added complexity here should only support faster validation of system integrity
