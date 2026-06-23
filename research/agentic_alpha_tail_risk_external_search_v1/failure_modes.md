# Failure Modes

- tail_risk_term_structure_ensemble: VIX9D/VIX3M, VVIX/VIX, and SKEW gates can cut risk, but high-return settings still accept deep SOXL drawdowns and drawdown-gated settings compound far below the production baseline.
- vol_target_tail_risk_throttle: volatility targeting reduces MaxDD only by shrinking exposure so aggressively that return_gate fails.
- coverage: VIX9D/VIX3M/VVIX/SKEW local files end on 2026-04-17, so this diagnostic cannot claim full 2026-06-09 production-window completion.
- next_action: use a longer independent options-risk history, live flow features, or a validated model family instead of another threshold-only tail-risk throttle.
