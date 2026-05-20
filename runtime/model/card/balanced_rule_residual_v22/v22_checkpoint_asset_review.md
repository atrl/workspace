# Checkpoint Review: soxl_balanced_rule_residual_v22_summary

## Summary
- profile: `balanced_rule_residual_v22`
- seed: `12345`
- staged training: `True`
- stage iters: `20`
- best checkpoint iter: `80`
- best score: `91.94`
- selection checkpoints captured: `2`

## Valid
- return: `62.93%`
- sharpe: `1.634`
- max_dd: `30.56%`
- avg_pos: `0.632`

## Fixed-Split
- test_2020: `115.60% / Sharpe 2.743 / MaxDD 16.22% / avg_pos 0.415`
- test_2022: `-24.52% / Sharpe -0.911 / MaxDD 32.71% / avg_pos 0.269`
- test_2023_2024: `37.85% / Sharpe 0.660 / MaxDD 32.43% / avg_pos 0.447`
- test_proxy: `695.45% / Sharpe 0.941 / MaxDD 54.52% / avg_pos 0.509`

## Gate Checks
- `test_2022_max_dd`: pass (actual=32.711, limit=55.000)
- `test_2022_avg_pos`: pass (actual=0.269, limit=0.350)
- `test_proxy_sharpe`: pass (actual=0.941, limit=0.500)

## Selection History
- iter `0`: score `89.54`, return `60.10%`, sharpe `1.639`, max_dd `27.93%`, avg_pos `0.614`
- iter `80`: score `91.94`, return `62.93%`, sharpe `1.634`, max_dd `30.56%`, avg_pos `0.632`

## Asset Availability

- retained actor: `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt`
- intermediate actor checkpoints: not present in the current repo
- captured selection rows: only iter `0` and iter `80`
- missing selection rows: expected staged boundaries at iter `20`, `40`, and
  `60` are not available for replay

This blocks direct checkpoint re-selection as a no-retrain optimization path.
The current retained actor is the final/best-scored iter `80` actor.

## Decision
- selection evidence is useful only if the best iter is earlier than max_iters and gate quality is preserved.
- promotion eligible: `True`
- no checkpoint switch is available without retraining or changing the training
  artifact-retention policy for future runs
