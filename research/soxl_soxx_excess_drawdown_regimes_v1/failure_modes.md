# Failure modes

- 2023–2026 has already been inspected by prior SOXL research, so it is a
  reused diagnostic segment rather than a clean holdout.
- Daily forward labels overlap, especially at 21 and 63 days. The card reports
  year-level bootstrap and horizon-spaced observations, but neither creates new
  independent market regimes.
- Equal-capital SOXL versus SOXX mixes timing skill with roughly three-times
  leverage. The beta-matched comparison is required before calling a regime
  risk-efficient.
- Drawdown bands are based on the contemporaneous SOXX all-time high and use
  only prior-close information, but historical ETF structure, financing costs,
  and tracking behavior can change.
- This card is descriptive. Turning the best historical cell into a trading
  rule without a preregistered clean-forward test would be overfit.
