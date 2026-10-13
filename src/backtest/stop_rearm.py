"""Opt-in diagnostic: a completed stop cycle rearms after drawdown recovers."""
from src.workflow.intraday_recovery_direct_allocation import _advance_state


def advance_rearmed_state(state, clock, drawdown, spec):
    # Saturating clock retains the existing state 3 feature category. All
    # emergency/trailing-loss caps are still applied by the caller afterward.
    if state == 3 and clock + 1 >= spec.hold_days and drawdown >= spec.stop_drawdown:
        return 3, spec.hold_days - 1, 1.0
    return _advance_state(state, clock, drawdown, spec)
