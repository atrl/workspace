# Checkpoint Review: soxl_balanced_checkpoint_review_summary

## Summary
- profile: `balanced_checkpoint_review`
- seed: `20260411`
- staged training: `True`
- stage iters: `20`
- best checkpoint iter: `40`
- best score: `80.60`
- selection checkpoints captured: `5`

## Valid
- return: `52.64%`
- sharpe: `1.558`
- max_dd: `26.71%`
- avg_pos: `0.570`

## Fixed-Split
- test_2020: `114.65% / Sharpe 2.266 / MaxDD 21.29% / avg_pos 0.560`
- test_2022: `-39.54% / Sharpe -1.095 / MaxDD 46.30% / avg_pos 0.372`
- test_2023_2024: `39.59% / Sharpe 0.638 / MaxDD 38.65% / avg_pos 0.535`
- test_proxy: `371.50% / Sharpe 0.765 / MaxDD 60.26% / avg_pos 0.521`

## Gate Checks
- `test_2022_max_dd`: fail (actual=46.297, limit=35.000)
- `test_2022_avg_pos`: fail (actual=0.372, limit=0.350)
- `test_proxy_sharpe`: pass (actual=0.765, limit=0.750)

## Selection History
- iter `0`: score `68.08`, return `41.07%`, sharpe `1.494`, max_dd `23.84%`, avg_pos `0.470`
- iter `20`: score `76.99`, return `50.42%`, sharpe `1.496`, max_dd `27.84%`, avg_pos `0.582`
- iter `40`: score `80.60`, return `52.64%`, sharpe `1.558`, max_dd `26.71%`, avg_pos `0.570`
- iter `60`: score `79.02`, return `51.35%`, sharpe `1.546`, max_dd `27.23%`, avg_pos `0.560`
- iter `80`: score `74.86`, return `48.87%`, sharpe `1.463`, max_dd `27.27%`, avg_pos `0.576`

## Decision
- selection evidence is useful only if the best iter is earlier than max_iters and gate quality is preserved.
- promotion eligible: `False`
