# wf2020_low_exposure_floor_representation_drift_v1 conclusion

Decision: `low_exposure_floor_representation_drift_confirmed_no_promotion`.

Key evidence:
- Low-exposure replay relative return delta vs prior: `-3.32pp`; MaxDD delta: `3.11pp`.
- Rebound slice samples expanded by `358`; dominant label shifted from action `4` to action `2`.
- September strategy return delta: `-3.33pp`; November delta: `-3.42pp`; December delta: `-3.13pp`.
- Late-rally mean position delta after 2020-10-01: `0.000182`; action-changed rows: `14`.

Interpretation:
- The training floor broadened the rebound representation instead of selectively fixing the late-2020 low-exposure rally state.
- More direct floor targets are likely to repeat the same drift unless a narrower state feature or reward gate separates late rally from mid-year drawdown risk.

Next action: `discover_narrow_late_rally_state_feature_or_reward_before_more_target_revision`.
