# Checkpoint Review: soxl_balanced_reward_guard_summary

## Summary
- profile: `balanced_reward_guard`
- seed: `20260411`
- staged training: `False`
- stage iters: `20`
- best checkpoint iter: `0`
- best score: `83.32`
- selection checkpoints captured: `2`

## Valid
- return: `55.81%`
- sharpe: `1.552`
- max_dd: `29.36%`
- avg_pos: `0.604`

## Fixed-Split
- test_2020: `137.91% / Sharpe 2.315 / MaxDD 21.10% / avg_pos 0.609`
- test_2022: `-36.00% / Sharpe -0.719 / MaxDD 47.36% / avg_pos 0.412`
- test_2023_2024: `50.23% / Sharpe 0.709 / MaxDD 39.85% / avg_pos 0.572`
- test_proxy: `523.36% / Sharpe 0.830 / MaxDD 62.27% / avg_pos 0.559`

## Gate Checks
- `test_2022_max_dd`: fail (actual=47.361, limit=35.000)
- `test_2022_avg_pos`: fail (actual=0.412, limit=0.350)
- `test_proxy_sharpe`: pass (actual=0.830, limit=0.750)

## Selection History
- iter `0`: score `83.32`, return `55.81%`, sharpe `1.552`, max_dd `29.36%`, avg_pos `0.604`
- iter `80`: score `83.32`, return `55.81%`, sharpe `1.552`, max_dd `29.36%`, avg_pos `0.604`

## Decision
- selection evidence is useful only if the best iter is earlier than max_iters and gate quality is preserved.
- promotion eligible: `False`
