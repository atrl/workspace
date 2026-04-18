# Checkpoint Review: soxl_balanced_phase_sampling_v1_summary

## Summary
- profile: `balanced_phase_sampling_v1`
- seed: `20260411`
- staged training: `True`
- stage iters: `20`
- best checkpoint iter: `20`
- best score: `80.84`
- selection checkpoints captured: `5`

## Valid
- return: `54.78%`
- sharpe: `1.482`
- max_dd: `29.80%`
- avg_pos: `0.636`

## Fixed-Split
- test_2020: `121.83% / Sharpe 2.193 / MaxDD 22.21% / avg_pos 0.619`
- test_2022: `-40.67% / Sharpe -1.221 / MaxDD 47.50% / avg_pos 0.364`
- test_2023_2024: `41.67% / Sharpe 0.644 / MaxDD 41.66% / avg_pos 0.572`
- test_proxy: `306.89% / Sharpe 0.705 / MaxDD 63.94% / avg_pos 0.560`

## Gate Checks
- `test_2022_max_dd`: fail (actual=47.503, limit=35.000)
- `test_2022_avg_pos`: fail (actual=0.364, limit=0.350)
- `test_proxy_sharpe`: fail (actual=0.705, limit=0.750)

## Selection History
- iter `0`: score `65.92`, return `39.52%`, sharpe `1.460`, max_dd `23.45%`, avg_pos `0.458`
- iter `20`: score `80.84`, return `54.78%`, sharpe `1.482`, max_dd `29.80%`, avg_pos `0.636`
- iter `40`: score `72.67`, return `47.77%`, sharpe `1.416`, max_dd `28.48%`, avg_pos `0.604`
- iter `60`: score `74.24`, return `49.69%`, sharpe `1.407`, max_dd `29.83%`, avg_pos `0.619`
- iter `80`: score `77.92`, return `52.03%`, sharpe `1.466`, max_dd `28.55%`, avg_pos `0.617`

## Decision
- selection evidence is useful only if the best iter is earlier than max_iters and gate quality is preserved.
- promotion eligible: `False`
