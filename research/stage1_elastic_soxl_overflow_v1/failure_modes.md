# Failure modes

- Development selection is negative if `return_delta_pct_points <= 0`; outer
  gains cannot override this failure.
- Cap-binding events are sparse and clustered. The minimum expanding-fold
  events-per-feature ratio is 6.67.
- The outer/development cap-event-rate ratio is
  3.61; this is a material regime shift.
- 2023–2026 has already been inspected and is not a clean holdout.
- The overflow sleeve is modeled as a separate open-to-next-open SOXL-for-SOXX
  swap. It does not claim venue-level execution or intraday fill precision.
