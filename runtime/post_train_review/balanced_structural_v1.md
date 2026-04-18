# Checkpoint Review: soxl_balanced_structural_v1_summary

## Summary
- profile: `balanced_structural_v1`
- seed: `20260411`
- staged training: `True`
- stage iters: `20`
- best checkpoint iter: `80`
- best score: `75.67`
- selection checkpoints captured: `5`

## Valid
- return: `47.28%`
- sharpe: `1.575`
- max_dd: `25.93%`
- avg_pos: `0.512`

## Fixed-Split
- test_2020: `101.26% / Sharpe 2.348 / MaxDD 19.98% / avg_pos 0.472`
- test_2022: `-31.27% / Sharpe -0.903 / MaxDD 39.86% / avg_pos 0.338`
- test_2023_2024: `45.53% / Sharpe 0.708 / MaxDD 33.47% / avg_pos 0.477`
- test_proxy: `306.97% / Sharpe 0.739 / MaxDD 53.66% / avg_pos 0.469`

## Gate Checks
- `test_2022_max_dd`: fail (actual=39.858, limit=35.000)
- `test_2022_avg_pos`: pass (actual=0.338, limit=0.350)
- `test_proxy_sharpe`: fail (actual=0.739, limit=0.750)

## Selection History
- iter `0`: score `63.82`, return `39.46%`, sharpe `1.369`, max_dd `25.25%`, avg_pos `0.500`
- iter `20`: score `54.62`, return `33.70%`, sharpe `1.197`, max_dd `25.18%`, avg_pos `0.502`
- iter `40`: score `55.92`, return `34.55%`, sharpe `1.226`, max_dd `26.25%`, avg_pos `0.497`
- iter `60`: score `64.94`, return `40.86%`, sharpe `1.366`, max_dd `26.95%`, avg_pos `0.517`
- iter `80`: score `75.67`, return `47.28%`, sharpe `1.575`, max_dd `25.93%`, avg_pos `0.512`

## Decision
- selection evidence is useful only if the best iter is earlier than max_iters and gate quality is preserved.
- promotion eligible: `False`
