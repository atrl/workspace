# balanced_rule_residual_v19_reg_v22 Artifact Gap Review

Date reviewed: 2026-05-21

## Purpose

`balanced_rule_residual_v19_reg_v22` combined the `v19_reg` regularization
setup with the `v22` VIX term-structure feature pack. It was meant to test
whether the walk-forward bear-market improvement from regularization could be
combined with the fixed-split VIX feature gain.

## Artifact Audit

Available:

- `runtime/model/card/balanced_rule_residual_v19_reg_v22/profile.yaml`

Missing:

- No retained actor checkpoint under `artifacts/models/`.
- No `soxl_balanced_rule_residual_v19_reg_v22_summary.json`.
- No walk-forward summaries for `wf_2020`, `wf_2022`, or `wf_2023_2024`.
- No completed comparison row before this audit.

Older `runtime/model/knowledge.md` text claimed a fixed-split pass and said the
walk-forward task had started, but no durable output is present in the repo.
Treat that old entry as an unverified historical note, not as promotion or
candidate evidence.

## Historical Comparison

Closest success:

- `balanced_rule_residual_v19_reg` produced a real walk-forward frontier:
  stronger bear-market defense in `wf_2022`, but weaker recovery in
  `wf_2023_2024`.
- `balanced_rule_residual_v22` is now the canonical scheduled runtime model
  after corrected walk-forward selection scoring and post-release-cap overlay
  approval.

Closest failure:

- `balanced_rule_residual_v22_release_v1` and
  `balanced_rule_residual_v22_soxx` showed fixed-window gains that did not
  survive later validation or walk-forward execution absorption.
- `balanced_rule_residual_v23` / `v24` were summary-only branches with missing
  actors and are now dropped.

Why this is not an active experiment:

- The retained repo state is profile-only.
- The old fixed-split claim cannot be reproduced from durable artifacts.
- Running it now would require a new governed experiment card, not a continuation
  of the stale task id.

## Decision

Drop `balanced_rule_residual_v19_reg_v22` from active-candidate interpretation.
Keep the profile and this note as a searchable historical idea only.

If revived, create a new governed experiment with:

- canonical baseline: `balanced_rule_residual_v22`
- direct comparison: `balanced_rule_residual_v19_reg`
- role: `ablate`
- changed layer: `feature` + `network`
- external reference: VIX term-structure feature family plus regularization
- required outputs: retained actor, fixed-split summary, complete walk-forward
  summaries, and a fresh row in `docs/experiment-landscape-review.md`
