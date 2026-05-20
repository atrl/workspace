# Selective Release Post-Release Cap Promotion Review

- Variant: `selective_release_recovery_post_release_cap_v1`
- Decision: `promotion_review_candidate`
- Runtime auto-select allowed: `False`
- Runtime auto-select reason: profile binds overlay but keeps auto_select_allowed=false and the base actor has not passed the model promotion gate

## Checks

| check | passed | detail |
|---|---:|---|
| `wf_2022_unchanged` | `True` | defensive walk-forward window remains unchanged |
| `wf_2023_2024_return_improved` | `True` | recovery walk-forward return improves versus baseline |
| `wf_2023_2024_maxdd_improved` | `True` | recovery walk-forward MaxDD improves versus baseline |
| `fixed_actor_noop` | `True` | fixed full-window actor overlay grid is unchanged |
| `runtime_profile_bound` | `True` | profile binds the overlay to runtime execution config |

## Readout

- This card packages the walk-forward candidate for promotion review only.
- It must not be used by scheduler auto-selection until the profile flag and model-promotion sidecar are explicitly approved.
