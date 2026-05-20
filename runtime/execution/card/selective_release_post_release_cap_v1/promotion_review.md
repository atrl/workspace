# Selective Release Post-Release Cap Promotion Review

- Variant: `selective_release_recovery_post_release_cap_v1`
- Decision: `promotion_review_candidate`
- Runtime auto-select allowed: `True`
- Runtime auto-select reason: promotion review, profile flag, and model gate all allow auto-selection

## Checks

| check | passed | detail |
|---|---:|---|
| `wf_2022_unchanged` | `True` | defensive walk-forward window remains unchanged |
| `wf_2023_2024_return_improved` | `True` | recovery walk-forward return improves versus baseline |
| `wf_2023_2024_maxdd_improved` | `True` | recovery walk-forward MaxDD improves versus baseline |
| `fixed_actor_noop` | `True` | fixed full-window actor overlay grid is unchanged |
| `runtime_profile_bound` | `True` | profile binds the overlay to runtime execution config |

## Readout

- This card is approved for scheduler auto-selection.
- Keep the profile flag and model-promotion gate passing; either one should block auto-selection again if revoked.
