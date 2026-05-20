# Runtime Selection Review

- Profile: `balanced_rule_residual_v22`
- Decision: `blocked`
- Runtime auto-select allowed: `False`
- Current auto-selected profile: `balanced_rule_residual_v19`

## Checks

| check | passed | detail |
|---|---:|---|
| `model_fixed_split_eligible` | `True` | base model fixed-split promotion gate is eligible |
| `model_walk_forward_evaluated` | `False` | base model walk-forward promotion gate is evaluated |
| `model_promotion_eligible` | `False` | base model promotion sidecar is eligible |
| `overlay_review_candidate` | `True` | overlay promotion review remains a candidate |
| `overlay_runtime_profile_bound` | `True` | overlay is bound in the runtime profile |
| `profile_overlay_auto_select_allowed` | `False` | profile overlay is explicitly approved for runtime auto-selection |

## Blocking Reasons

- model walk-forward promotion gate not evaluated
- model promotion sidecar is not eligible
- profile overlay auto_select_allowed=false
