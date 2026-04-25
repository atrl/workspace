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
| `aggressive` | `experiments/candidate/aggressive/` | `aggressive` | `balanced` | `notes.md` |
| `balanced_conservative_ft` | `experiments/candidate/balanced_conservative_ft/` | `balanced` | `balanced` | `notes.md` |
| `balanced_rule_residual_mode_v1` | `experiments/candidate/balanced_rule_residual_mode_v1/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_moderate_v1` | `experiments/candidate/balanced_rule_residual_moderate_v1/` | `balanced` | `balanced_rule_residual_v16` | `notes.md` |
| `balanced_rule_residual_refined_v2` | `experiments/candidate/balanced_rule_residual_refined_v2/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_refined_v3` | `experiments/candidate/balanced_rule_residual_refined_v3/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_v19` | `experiments/candidate/balanced_rule_residual_v19/` | `balanced` | `balanced_rule_residual_v16` | `notes.md` |
| `balanced_rule_residual_v19_academic` | `experiments/candidate/balanced_rule_residual_v19_academic/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `balanced_rule_residual_v22` | `experiments/candidate/balanced_rule_residual_v22/` | `balanced` | `balanced_rule_residual_v19` | `notes.md` |
| `risk_budget_usage_v1` | `experiments/candidate/risk_budget_usage_v1/` | `balanced` | `balanced_rule_residual_imitation_v4` | `notes.md` |
| `risk_budget_usage_v3` | `experiments/candidate/risk_budget_usage_v3/` | `balanced` | `risk_budget_usage_v1` | `notes.md` |
| `smoke_rule_residual_v1` | `experiments/candidate/smoke_rule_residual_v1/` | `smoke` | `smoke` | `notes.md` |

## Archive

| Profile | Package | Family | Baseline | Evidence |
|---|---|---|---|---|
| `balanced_kelly` | `experiments/archive/balanced_kelly/` | `balanced` | `balanced_rule_residual_imitation_v4` | `notes.md` (merged v5-v9b) |
| `balanced_rule_residual_imitation` | `experiments/archive/balanced_rule_residual_imitation/` | `balanced` | `balanced_rule_residual_v16` | `notes.md` (merged v1-v7) |
| `risk_budget_usage_v2` | `experiments/archive/risk_budget_usage_v2/` | `balanced` | `risk_budget_usage_v1` | `notes.md` |

### Historical Families (yaml only in `configs/soxl/`)

- **balanced_rule_residual** evolution: v1 → v2 → v3,4,6,8,9 → v12,14,15,16,17,18,20,21
  - `configs/soxl/balanced_rule_residual_v{1,2,3,4,6,8,9,12,14,15,16,17,18,20,21}.yaml`
- **balanced_floor40** family: `configs/soxl/balanced_floor40*.yaml`
- **smoke variants**: `configs/soxl/smoke_*.yaml`
