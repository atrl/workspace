# 2026-04-17: historical-best `balanced` feature gap

## Conclusion

- Canonical `balanced` is already competent at:
  - persistent stress defense
  - mature trend capture
- Its main blind spot is:
  - transition states
  - especially stress-release, bear-market rebounds, and early trend restarts

## Why this matters

- This explains why recent loops that only retuned reward weights failed:
  - the actor is seeing the wrong state abstraction, not just using the wrong penalty scalar

## Next feature direction

- Prioritize transition / persistence features before more reward tuning:
  - `regime_risk_delta_*`
  - `vol_stress_score_delta_*`
  - days-in-stress / days-below-anchor
  - rebound-from-low / days-since-low

## Evidence

- `runtime/post_train_review/historical_balanced_feature_recall.md`
