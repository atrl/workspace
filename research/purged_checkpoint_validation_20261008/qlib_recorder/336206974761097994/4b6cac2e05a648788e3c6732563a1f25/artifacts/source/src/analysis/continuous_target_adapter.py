"""Bounded float64 boundary correction before the sealed native Qlib helper.

The original accounting path and selection are unchanged. At an exact zero-cash
target, subtraction of the fee reserve can report cash a fraction of one ULP
below zero. Only affected target rows are normalized for the strict native
long-only interface; material negatives remain errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.convex_qlib import native_open_reconciliation as _sealed_native_open_reconciliation

TARGET_COLUMNS = tuple(f"open_{asset}_weight" for asset in ("soxl", "soxx", "cash"))
ROUNDOFF_BOUND = 8 * np.finfo(np.float64).eps


def canonicalize_target_roundoff(schedule: pd.DataFrame):
    """Return a copied schedule and audit; never mutate the research replay."""
    if not set(TARGET_COLUMNS).issubset(schedule):
        raise ValueError("blocked: native roundoff adapter requires explicit open net targets")
    target = schedule.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if not len(target) or not np.isfinite(target).all():
        raise ValueError("blocked: native roundoff adapter requires nonempty finite targets")
    affected = (target < 0).any(axis=1)
    corrected = target.copy()
    row_error = np.abs(target.sum(axis=1) - 1.)
    if affected.any():
        values = target[affected]
        if values.min() < -ROUNDOFF_BOUND or row_error[affected].max() > ROUNDOFF_BOUND:
            raise ValueError("blocked: native target error exceeds eight float64 epsilons")
        values = np.maximum(values, 0.)
        values /= values.sum(axis=1, keepdims=True)
        corrected[affected] = values
        if np.max(np.abs(corrected - target)) > ROUNDOFF_BOUND:
            raise ValueError("blocked: native target correction exceeds eight float64 epsilons")
    result = schedule.copy(deep=True)
    if affected.any():
        result.loc[affected, list(TARGET_COLUMNS)] = corrected[affected]
    return result, {
        "scope": "native adapter only; original accounting replay and model selection are unchanged",
        "affected_rows": int(affected.sum()), "negative_components": int((target < 0).sum()),
        "original_minimum_weight": float(target.min()),
        "maximum_affected_row_sum_error": float(row_error[affected].max()) if affected.any() else 0.,
        "maximum_absolute_weight_change": float(np.max(np.abs(corrected - target))),
        "absolute_bound": float(ROUNDOFF_BOUND), "float64_epsilon_multiple": 8,
        "normalization": "clip negative components then normalize affected rows only",
        "unchanged_nonnegative_rows": True, "original_replay_mutated": False,
    }


def native_open_reconciliation(frame, replay, *, provider_uri, cost_bps=10., account=100_000.):
    """Delegate the bounded corrected copy to the unchanged Qlib reconciler."""
    schedule, audit = canonicalize_target_roundoff(replay)
    result = _sealed_native_open_reconciliation(frame, schedule, provider_uri=provider_uri,
                                               cost_bps=cost_bps, account=account)
    result["audit"]["target_roundoff_adapter"] = audit
    return result
