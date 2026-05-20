# balanced_rule_residual_v24

Date reviewed: 2026-05-21

Role:

- Historical summary-only model-selection artifact.
- Direct comparison: canonical `balanced_rule_residual_v22`.

Available artifacts:

- `soxl_balanced_rule_residual_v24_summary.json`
- `v24_vs_v22_review.md`

Missing artifacts:

- No retained actor exists at the summary `model_path`.
- No `profile.yaml` exists in this card directory.
- No walk-forward summaries exist for this profile.

Readout:

- Proxy return improved from `695.45%` on v22 to `758.59%`, but proxy MaxDD was
  essentially unchanged and slightly worse.
- `test_2022` improved modestly and `test_2023_2024` was near-flat, but
  `test_2020` deteriorated materially.
- Fixed-split promotion alone is insufficient because the branch has no retained
  actor and no walk-forward evidence.

Decision:

- Dropped as an active candidate.
- Keep summary and review for retrieval; do not attempt walk-forward because the
  actor is missing.
- Do not revive this branch unless a future mechanism can reproduce the fixed
  split behavior with retained actors and complete walk-forward evidence.
