# Failure modes

- SOXS is a daily-reset inverse leveraged ETF; even five-day exposure is path dependent and can decay in chop.
- Yahoo corporate-action history was inconsistent in the recent SOXS window; the provider uses Nasdaq as authoritative after the stitch date, records the scale audit, and applies Yahoo adjusted-close ratios for distributions.
- The defense label is rare and regime-dependent, so class balancing does not create new crash regimes.
- Cash has zero return because a PIT Treasury series is not part of this provider.
- The 2023-2026 outer interval was already observed before this experiment and is diagnostic only.
