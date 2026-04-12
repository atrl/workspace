# HEARTBEAT.md

## Current focus

- Project: SOXL-only qlib-native research harness
- Mode: agent-first experiment loop
- Requirement: at least one idea validation per hour while work is active

## Latest validated result (2026-04-11 17:48)

**Balanced walk-forward run completed:**
- Full test (2020-2026): `368.7% / Sharpe 0.80 / MaxDD 51.6% / 894 trades`
- test_2022 slice: `-26.5% / Sharpe -0.84 / MaxDD 34.6%` (improved from -39.8%)
- WF_2022: `53.3% / Sharpe 0.38 / MaxDD 65.4%` (positive return in walk-forward 2022)
- WF_2023_2024: `1.1% / Sharpe 0.10` (flat — needs investigation)

## First checks on heartbeat

1. Read `codex_exec_plans/active/2026-04-10-soxl-agent-loop.md`
2. Check whether any training run is still active
3. If idle: run the next planned validation from the active plan

## Next hypotheses to validate

1. WF_2023_2024 flat at 1.1% — investigate whether reward shaping is too conservative in sideways markets
2. Consider volatility-targeting position sizing to reduce MaxDD in walk-forward folds

## Guardrails

- No future leakage
- No undocumented experiment
- No stopping at "I have an idea" without logging the validation plan
