"""Read-only H5 source compatibility audit (no parent modification, no whitewashing).

Scope discipline: this is a SINGLE clock-contradiction check, not a whole-model causal
certification. It reports three separate things:

1. ``clock_contradiction_detected`` — does the recorded metadata claim a decision clock
   earlier than the actual observation of the consumed pretrade state?
2. ``actual_future_dependency`` — does the strategy actually depend on future prices?
   This is ``unknown`` unless proven per source; benchmark_terminal's mostly pure
   SOXX/cash pretrade weights may not depend on a future price.
3. ``financial_target_impact`` — whether the clock issue changes targets/fees/weights,
   ``unknown`` until a correct quote-clock replay is compared.

It never asserts a lookahead advantage and never modifies a parent or old promotion flag.
"""
from __future__ import annotations

from src.data.generic_horizon import allocation_clock, predictor_clock

CLAIM_INVALID = "legacy_clock_claim_invalid_for_promotion"
NO_CONTRADICTION = "no_clock_contradiction_detected"
PREDICTOR_CLOCK = "signal_day_16:15"
ALLOCATION_CLOCK = "next_open_after_09:31"


def audit_h5_source_compatibility(*, held_is_current_opening_state: bool,
                                  metadata_clock: str = PREDICTOR_CLOCK,
                                  quote_clock: str = ALLOCATION_CLOCK) -> dict:
    """Single clock-contradiction verdict; not a whole-model causal pass."""
    contradiction = bool(held_is_current_opening_state and metadata_clock == PREDICTOR_CLOCK)
    return {
        "check_scope": "single_clock_contradiction_only",
        "held_is_current_opening_state": bool(held_is_current_opening_state),
        "metadata_clock": metadata_clock,
        "correct_quote_clock": quote_clock,
        "clock_contradiction_detected": contradiction,
        "legacy_clock_claim_verdict": CLAIM_INVALID if contradiction else NO_CONTRADICTION,
        "whole_model_causal_certified": False,
        "actual_future_dependency": "unknown",   # do NOT assert a lookahead advantage
        "financial_target_impact": "unknown",     # requires a correct quote-clock comparison
        "financial_reference_status": "not_comparable_until_correct_quote_clock_replay",
        "financial_reference_comparable_only_if": "new correct quote-clock H5 reproduces old targets/fees/weights (no refit)",
        "soxx_primary_dependency": False,
        "parent_modified": False,
        "old_results_relabeled": False,
        "old_promotion_status_modified": False,
    }


def audit_legacy_clock_from_frames(*, signal_dates, trade_dates, held_is_current_opening_state: bool = True,
                                   metadata_clock: str = PREDICTOR_CLOCK) -> dict:
    """Read-only check on recorded frames; rejects NaT clocks instead of silently passing."""
    prediction = predictor_clock(signal_dates)
    allocation = allocation_clock(trade_dates)
    if len(prediction) != len(allocation):
        raise ValueError("blocked: signal/trade frame length mismatch")
    if prediction.hasnans or allocation.hasnans:
        raise ValueError("blocked: clock contains NaT")
    verdict = audit_h5_source_compatibility(held_is_current_opening_state=held_is_current_opening_state,
                                            metadata_clock=metadata_clock)
    verdict["prediction_at_examples"] = [str(stamp) for stamp in prediction[:2]]
    verdict["allocation_at_examples"] = [str(stamp) for stamp in allocation[:2]]
    verdict["allocation_after_prediction"] = bool((allocation > prediction).all())
    return verdict
