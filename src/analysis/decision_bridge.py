"""Small decision-chain reports from an already validated native account.

This is descriptive attribution, not a causal estimator or a promotion gate.
Callers supply the explicit trace schema; no historical card is discovered or
loaded here. Large row/window outputs belong in the caller's Qlib Recorder.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

import numpy as np
import pandas as pd


def _dates(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["trade_date"]))
    if (dates.tz is not None or dates.hasnans or dates.has_duplicates
            or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())):
        raise ValueError(f"{name} requires unique chronological native session dates")
    return dates


def _digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def build_decision_bridge(
    trace: pd.DataFrame,
    native_replay: pd.DataFrame,
    *,
    native_audit: Mapping,
    source_identity: Mapping,
    fields: Mapping[str, str | Sequence[str]],
    native_calendar: pd.DatetimeIndex,
) -> dict:
    """Preserve every account day and expose gaps in the decision chain.

    ``fields`` optionally maps source_target/projected_target/postfill_weights to
    three ordered SOXL/SOXX/CASH columns, and order_count/fees_usd to one column.
    Missing stages remain unknown. They never become zero exposure or no trade.
    Native net returns are read from ``native_replay.strategy_return``.
    """
    if native_audit.get("status") != "passed":
        raise ValueError("decision bridge requires passed native account evidence")
    if not source_identity:
        raise ValueError("decision bridge requires caller-bound source identity")
    dates = _dates(trace, "trace")
    if not len(dates) or not dates.equals(_dates(native_replay, "native replay")):
        raise ValueError("trace and native account must retain the same nonempty date grid")
    calendar = pd.DatetimeIndex(native_calendar)
    if (calendar.tz is not None or calendar.has_duplicates or calendar.hasnans
            or not calendar.is_monotonic_increasing):
        raise ValueError("invalid native calendar")
    expected = calendar[(calendar >= dates[0]) & (calendar <= dates[-1])]
    if not dates.equals(expected):
        raise ValueError("decision bridge cannot omit native sessions")
    returns = pd.to_numeric(native_replay["strategy_return"], errors="raise").to_numpy(float)
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("invalid native net returns")
    daily = pd.DataFrame({"trade_date": dates, "strategy_return": returns})
    unknowns: list[str] = []
    stage_values = {}
    for stage in ("source_target", "projected_target", "postfill_weights"):
        columns = fields.get(stage)
        if not columns or isinstance(columns, str) or len(columns) != 3 or not set(columns).issubset(trace):
            unknowns.append(stage)
            continue
        values = trace[list(columns)].to_numpy(float)
        finite = np.isfinite(values).all(axis=1)
        daily[stage + "_known"] = finite
        # A bound missingness mask is part of the report, not a row filter.
        stage_values[stage] = np.where(finite[:, None], values, np.nan)
        for i, asset in enumerate(("soxl", "soxx", "cash")):
            daily[f"{stage}_{asset}"] = stage_values[stage][:, i]
    for left, right in (("source_target", "projected_target"),
                        ("projected_target", "postfill_weights")):
        name = f"{left}_to_{right}_l1"
        daily[name] = np.nan
        if left in stage_values and right in stage_values:
            # L1 target distance is not actual turnover and not causal P&L.
            daily[name] = np.abs(stage_values[left] - stage_values[right]).sum(axis=1)
    for field in ("order_count", "fees_usd"):
        column = fields.get(field)
        if not isinstance(column, str) or column not in trace:
            daily[field] = np.nan
            unknowns.append(field)
        else:
            values = pd.to_numeric(trace[column], errors="raise").to_numpy(float)
            if ((np.isfinite(values) & (values < 0)).any()
                    or (field == "order_count" and
                        (np.isfinite(values) & (values != np.floor(values))).any())):
                raise ValueError(f"invalid {field}")
            daily[field] = values
    # Days since an actual filled order; cash weights drifting with prices and
    # split-driven quantity changes must not manufacture trades.
    duration, last_trade = [], None
    for i, count in enumerate(daily.order_count):
        if not np.isfinite(count):
            last_trade = None
            duration.append(np.nan)
        elif count > 0:
            last_trade = i
            duration.append(0.0)
        else:
            duration.append(np.nan if last_trade is None else float(i - last_trade))
    daily["sessions_since_last_filled_order"] = duration

    def summarize(mask) -> dict:
        part = daily.loc[mask]
        result = {"rows": len(part),
                  "net_return": float(np.expm1(np.log1p(part.strategy_return).sum()))}
        for name in daily.columns:
            if name.endswith("_l1") or name.endswith("_known"):
                result[name + "_observed"] = int(part[name].notna().sum())
                result[name + "_mean"] = None if part[name].notna().sum() == 0 else float(part[name].mean())
        for name in ("order_count", "fees_usd"):
            result[name + "_observed"] = int(part[name].notna().sum())
            result[name + "_sum"] = None if part[name].isna().any() else float(part[name].sum())
        return result

    periods = []
    for frequency, kind in (("Y", "year"), ("Q", "quarter")):
        labels = dates.to_period(frequency)
        for label in labels.unique():
            expected_period = calendar[calendar.to_period(frequency) == label]
            observed = dates[labels == label]
            complete = (calendar[0] < label.start_time and calendar[-1] > label.end_time
                        and observed.equals(expected_period))
            periods.append({"kind": kind, "period": str(label),
                            "complete_period": bool(complete), **summarize(labels == label)})
    windows = [{"first_trade": dates[i - 125], "last_trade": dates[i],
                **summarize((np.arange(len(daily)) >= i - 125) & (np.arange(len(daily)) <= i))}
               for i in range(125, len(daily))]
    return {
        "summary": {"status": "descriptive_only", "rows": len(daily),
                    "trace_sha256": _digest(trace), "source_identity": dict(source_identity),
                    "unknown_stages": unknowns, "causal_effect_identified": False,
                    "SOXX_comparison_performed": False, "promotion_eligible": False,
                    "rolling_windows_overlap": True, **summarize(np.ones(len(daily), dtype=bool))},
        "daily": daily, "periods": pd.DataFrame(periods), "rolling_126": pd.DataFrame(windows),
    }
