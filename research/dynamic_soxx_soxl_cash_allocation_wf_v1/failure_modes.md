# Failure Modes

- The model can beat SOXX in SOXL validation and still fail if TQQQ/UPRO/SPXL/TECL proxy generalization is negative.
- The annual target is intentionally stricter than SOXX benchmark reporting: excess must be at least 20 percentage points and drawdown must stay above -10%.
- Cash lowers drawdown but can also make the strategy miss SOXX trend participation.
- SOXX constituent factors are current-holdings backfilled and remain survivorship-biased until point-in-time holdings are available.
- 5m/15m intraday data is used only as a next-day risk state; it does not prove intraday trading alpha.
- Current promotion decision: not_promoted_dynamic_allocation_goal_or_generalization_gate_failed.
