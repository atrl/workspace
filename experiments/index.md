# Experiment Card Index

`experiments/` stores runnable intent cards only. Generated evidence and result narratives live in `runtime/` and `docs/research_knowledge/`.

## Canonical Baselines

| Profile | Package | Family | Evidence |
|---|---|---|---|
| `balanced` | `experiments/canonical/balanced/` | `balanced` | `experiments/canonical/balanced/notes.md` |
| `smoke` | `experiments/canonical/smoke/` | `smoke` | `experiments/canonical/smoke/notes.md` |

## Active Candidates

| Profile | Package | Family | Baseline | Evidence |
|---|---|---|---|---|
| `balanced_conservative_ft` | `experiments/candidate/balanced_conservative_ft/` | `balanced` | `balanced` | `notes.md` |
| `balanced_kelly` | `experiments/candidate/balanced_kelly/` | `balanced` | `balanced_rule_residual_imitation_v4` | `notes.md` (merged v9/v9b) |
| `balanced_rule_residual_imitation` | `experiments/candidate/balanced_rule_residual_imitation/` | `balanced` | `balanced_rule_residual_v16` | `notes.md` (merged v1-v7) |
| `balanced_rule_residual_mode_v1` | `experiments/candidate/balanced_rule_residual_mode_v1/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_moderate_v1` | `experiments/candidate/balanced_rule_residual_moderate_v1/` | `balanced` | `balanced_rule_residual_v16` | `notes.md` |
| `balanced_rule_residual_refined_v2` | `experiments/candidate/balanced_rule_residual_refined_v2/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_refined_v3` | `experiments/candidate/balanced_rule_residual_refined_v3/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_v19` | `experiments/candidate/balanced_rule_residual_v19/` | `balanced` | `balanced_rule_residual_v16` | `notes.md` |
| `balanced_rule_residual_v19_academic` | `experiments/candidate/balanced_rule_residual_v19_academic/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_v22` | `experiments/candidate/balanced_rule_residual_v22/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `risk_budget_usage_v1` | `experiments/candidate/risk_budget_usage_v1/` | `balanced` | `balanced_rule_residual_imitation_v4` | `notes.md` |
| `smoke_rule_residual_v1` | `experiments/candidate/smoke_rule_residual_v1/` | `smoke` | `smoke` | `notes.md` |

## Archive (by family)

- **balanced_rule_residual** evolution: v1 → v2 → v3,4,6,8,9 → v12,14,15,16,17,18,20,21
  - `experiments/archive/balanced_rule_residual_v{1,2,3,4,6,8,9,12,14,15,16,17,18,20,21}/`
  - Most retain `profile.yaml` only; notes trimmed to ≤50 lines.
- **checkpoint_review**: `experiments/archive/balanced_checkpoint_review/`
