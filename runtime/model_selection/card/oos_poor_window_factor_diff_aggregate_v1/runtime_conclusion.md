## Runtime Conclusion

Decision: `next_training_target_reentry_participation_or_behavior_clone`.

Top shared bad-window cluster: `2025-04-04` .. `2025-07-22`, profiles `5`, windows `30`. Dominant failure type is `underparticipation` with mean excess `-160.50pp`, worst excess `-199.39pp`, max drawdown `16.32%`, and mean position `0.520`.

Stable feature-delta hypotheses: vol_of_vol_21, tracking_error_vol_21, decay_63, realized_vol_63, tracking_error_21, realized_vol_21.

Readout:

- This is a shared OOS failure cluster, not an isolated artifact of the newest reward run.
- If the top cluster stays underparticipation-heavy, the next training branch should not be another reward coefficient tweak. It should refresh participation behavior with supervised/BC targets or a fixed-actor re-entry overlay and then retest `2022`, `2023-2024`, and walk-forward.
- Feature deltas remain hypotheses only; they need IC / single-signal / walk-forward gates before becoming promoted factors.
