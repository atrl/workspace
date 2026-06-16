# Mode-Conditioned Drawdown-Drift Short Training Screen

## Decision

- decision: `primary_fold_short_training_failed_no_promotion`
- primary_fold: `wf_2020`
- promotion_allowed: `False`
- final_retained_count: `0`

## Primary Fold Comparison

| candidate | kind | return | buyhold | relative | MaxDD | DD improvement | return delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mode_conditioned_rebound_reward_train_v2 | source_replay | 50.08% | 58.09% | -8.01pp | 23.09% | 0.00pp | 0.00pp |
| mode_conditioned_drawdown_drift_short_train_v1 | short_training_screen | 50.08% | 58.09% | -8.01pp | 23.09% | 0.00pp | 0.00pp |

## Readout

- This is one primary-fold short training loop, not a retained model.
- Relative return is recomputed from the review daily CSV to keep the source and short-screen rows on the same buyhold basis.
- Full retention still requires all required walk-forward windows plus the strict retention contract.
