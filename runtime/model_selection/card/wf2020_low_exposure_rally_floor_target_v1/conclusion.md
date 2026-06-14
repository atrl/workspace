# wf2020_low_exposure_rally_floor_target_v1 conclusion

Decision: `wf2020_low_exposure_rally_floor_target_ready_for_official_replay_no_promotion`.

- Changed rows: `452` in `wf_2020_train` only.
- Floor: `0.30` when `soxl_mom5_prev >= -0.02`, source target is below floor, and row is not downtrend.
- Eval leakage rows: `0`.
- Consumed-target mismatch rows: `0`.

Promotion remains blocked. Next action: official three-fold checkpoint replay with this target contract.
