## Runtime Conclusion

Decision: `blocked_no_feature_promotion`. This pack should not be used for direct model training now.

Evidence:

- Full-pass feature count: `0` out of `4`.
- Correlation-redundant features: `SOXL_SOXX_SPREAD_VOL_Z_21, SOXL_SOXX_RATIO_Z_252, SOXL_SOXX_RATIO_WIDEN_5`. These mainly re-express existing trend/momentum/volatility dimensions.
- Weak direction/execution features: `SOXL_SOXX_CHOP_DIVERGENCE_21`.
- `SOXL_SOXX_CHOP_DIVERGENCE_21` remains the only orthogonal spread/oscillation candidate (`max rho=0.081`), but its best IC is `-0.0016` with IC+ `0.425`, so it lacks directional stability.

Implication for the active overfit objective:

- Do not launch a SOXX/SOXL spread feature retrain from this evidence. It would likely add redundant inputs and increase overfit risk.
- If this family is revisited, use it only as a regime-gated execution diagnostic or as one term inside a walk-forward-weighted multi-objective selector, not as a standalone core feature.
- The next factor loop should prefer orthogonal candidates from the gate-gap context (`vol_of_vol_21`, `tracking_error_vol_21`, `decay_63`) or an explicitly new external mechanism with IC and walk-forward validation.
