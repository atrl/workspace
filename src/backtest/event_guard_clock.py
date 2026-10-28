"""Completed-minute decision and next-bar execution clock for Stage1 research."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GuardSlot:
    bar_end_at: pd.Timestamp
    decision_at: pd.Timestamp
    fill_at: pd.Timestamp


def guard_slot(bar_end_at, *, session_open, session_close,
               original_open_fill=None) -> GuardSlot | None:
    """Return a causal event slot; bar timestamps are right-edge local ET."""
    end = pd.Timestamp(bar_end_at)
    opened, closed = pd.Timestamp(session_open), pd.Timestamp(session_close)
    if any(value.tzinfo is not None for value in (end, opened, closed)):
        raise ValueError("blocked: guard clock requires local naive ET timestamps")
    if closed <= opened or end.normalize() != opened.normalize() or end.normalize() != closed.normalize():
        raise ValueError("blocked: event and session calendar differ")
    first_fill = (pd.Timestamp(original_open_fill) if original_open_fill is not None
                  else opened + pd.Timedelta(minutes=17))
    if first_fill.normalize() != opened.normalize():
        raise ValueError("blocked: original opening fill belongs to another session")
    decision = end + pd.Timedelta(minutes=1)
    fill = end + pd.Timedelta(minutes=2)
    if end < first_fill + pd.Timedelta(minutes=1) or fill >= closed:
        return None
    return GuardSlot(bar_end_at=end, decision_at=decision, fill_at=fill)
