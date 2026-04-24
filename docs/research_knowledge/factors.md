# Factor Knowledge

## Retained Insight

Rulebase-derived timing signals can be useful as independent factors when judged by time-series IC, directional accuracy, profit factor, and correlation budget rather than strategy Sharpe alone.

## Failed Pattern

Do not promote a feature because it improves a local validation window if it increases fixed-split `test_2022` exposure or duplicates existing core features above the correlation budget.

## Evidence Path

- `docs/factor_research/`
- `data/soxl_factor_library.db`
- `memory/2026-04-23-soxl-factor-mining-skill.md`

## Reusable Method

Use the leveraged ETF factor workflow: hypothesize from rulebase or literature, generate the feature, run IC / decay / stability / correlation checks, then run independent signal backtests before touching RL profiles.

## Next Valid Experiment Type

Generalize accepted SOXL factor checks to another 3x ETF with explicit asset metadata and compare whether signal direction, horizon, and correlation budget survive outside semiconductors.
