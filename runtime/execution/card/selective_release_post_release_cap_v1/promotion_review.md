# Selective Release Post-Release Cap Promotion Review

- Variant: `selective_release_recovery_post_release_cap_v1`
- Decision: `promotion_review_candidate`
- Runtime auto-select allowed: `False`
- Runtime auto-select reason: overlay is not profile-bound production runtime behavior and the base v22 actor sidecar has not passed the model promotion gate

## Checks

| check | passed | detail |
|---|---:|---|
| `wf_2022_unchanged` | `True` | defensive walk-forward window remains unchanged |
| `wf_2023_2024_return_improved` | `True` | recovery walk-forward return improves versus baseline |
| `wf_2023_2024_maxdd_improved` | `True` | recovery walk-forward MaxDD improves versus baseline |
| `fixed_actor_noop` | `True` | fixed full-window actor overlay grid is unchanged |

## Readout

- This card packages the walk-forward candidate for promotion review only.
- It must not be used by scheduler auto-selection until a profile-bound runtime overlay and model-promotion sidecar are explicitly approved.
