# balanced_rule_residual_v23

Date reviewed: 2026-05-21

Role:

- Historical summary-only model-selection artifact.
- Direct comparison: canonical `balanced_rule_residual_v22`.

Available artifacts:

- `soxl_balanced_rule_residual_v23_summary.json`
- `v23_vs_v22_review.md`

Missing artifacts:

- No retained actor exists at the summary `model_path`.
- No `profile.yaml` exists in this card directory.
- No walk-forward summaries exist for this profile.

Readout:

- Proxy return fell from `695.45%` on v22 to `493.23%`.
- Proxy MaxDD improved from `54.52%` to `51.20%`, indicating a more defensive
  style rather than a better frontier.
- `test_2022` improved modestly, but `test_2020` and `test_2023_2024`
  deteriorated.
- Selection history peaked at iter `0` and fell by iter `80`, which is
  checkpoint-instability evidence, not a reason to retrain this branch.

Decision:

- Dropped as an active candidate.
- Keep summary and review for retrieval; do not attempt walk-forward because the
  actor is missing.
- Do not use this as a reason for another broad PPO run.
