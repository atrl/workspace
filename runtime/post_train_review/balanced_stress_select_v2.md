# Checkpoint Review: soxl_balanced_stress_select_v2_summary

## Summary
- profile: `balanced_stress_select_v2`
- seed: `20260411`
- staged training: `True`
- stage iters: `20`
- best checkpoint iter: `80`
- best score: `71.11`
- selection checkpoints captured: `5`

## Valid
- return: `53.79%`
- sharpe: `1.534`
- max_dd: `27.47%`
- avg_pos: `0.588`

## Fixed-Split
- test_2020: `128.18% / Sharpe 2.369 / MaxDD 22.27% / avg_pos 0.568`
- test_2022: `-34.68% / Sharpe -0.772 / MaxDD 46.28% / avg_pos 0.398`
- test_2023_2024: `52.02% / Sharpe 0.734 / MaxDD 37.21% / avg_pos 0.548`
- test_proxy: `462.28% / Sharpe 0.817 / MaxDD 63.28% / avg_pos 0.534`

## Gate Checks
- `test_2022_max_dd`: fail (actual=46.283, limit=35.000)
- `test_2022_avg_pos`: fail (actual=0.398, limit=0.350)
- `test_proxy_sharpe`: pass (actual=0.817, limit=0.750)

## Selection History
- iter `0`: score `59.03`, return `41.07%`, sharpe `1.494`, max_dd `23.84%`, avg_pos `0.470`
- iter `20`: score `65.20`, return `49.01%`, sharpe `1.446`, max_dd `28.11%`, avg_pos `0.588`
- iter `40`: score `67.03`, return `49.56%`, sharpe `1.504`, max_dd `27.52%`, avg_pos `0.568`
- iter `60`: score `56.78`, return `42.26%`, sharpe `1.358`, max_dd `28.22%`, avg_pos `0.556`
- iter `80`: score `71.11`, return `53.79%`, sharpe `1.534`, max_dd `27.47%`, avg_pos `0.588`

## Decision
- selection evidence is useful only if the best iter is earlier than max_iters and gate quality is preserved.
- promotion eligible: `False`
