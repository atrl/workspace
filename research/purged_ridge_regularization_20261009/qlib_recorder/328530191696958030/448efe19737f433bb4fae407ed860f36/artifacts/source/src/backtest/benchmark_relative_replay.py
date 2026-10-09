"""Funded schedules for native Qlib replay, with strictly delayed expert feedback.

These paths are independent reconciliation ledgers, not portfolio authority.
The workflow must replay each path through Qlib before using its evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import ASSETS, DATES, _dates, validate_replay
from src.model.post_fee_settlement import scalar_settle


def validate_market(frame):
    result = _dates(frame)
    required = {"decision_at", "return_available_at", "soxl_return", "soxx_return"}
    if required - set(result):
        raise ValueError("blocked: missing market timing or return fields")
    for column in ("decision_at", "return_available_at"):
        if any(pd.isna(x) or pd.Timestamp(x).tzinfo is None for x in result[column]):
            raise ValueError("blocked: market times require explicit timezones")
        result[column] = pd.to_datetime(result[column], utc=True)
    decision = result.decision_at
    available = result.return_available_at
    if (not decision.is_monotonic_increasing or decision.duplicated().any()
            or not available.is_monotonic_increasing or available.duplicated().any()
            or not decision.lt(available).all()):
        raise ValueError("blocked: invalid market outcome chronology")
    # An open-to-open outcome cannot mature before its exit session's open.
    exit_open = pd.DatetimeIndex(result.exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    if not (available.array > exit_open.tz_convert("UTC")).all():
        raise ValueError("blocked: outcome is available before exit-open observation")
    signal_close = pd.DatetimeIndex(result.signal_date).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    trade_open = pd.DatetimeIndex(result.trade_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    if not ((decision.array >= signal_close.tz_convert("UTC"))
            & (decision.array < trade_open.tz_convert("UTC"))).all():
        raise ValueError("blocked: decision must follow signal close and precede trade open")
    vals = result[["soxl_return", "soxx_return"]].to_numpy(float)
    if not np.isfinite(vals).all() or (vals <= -1).any():
        raise ValueError("blocked: invalid complete market returns")
    return result


def _target(value):
    arr = np.asarray(value, float)
    if (arr.shape != (3,) or not np.isfinite(arr).all() or (arr < 0).any()
            or not np.isclose(arr.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("blocked: target must be a funded three-asset simplex")
    return arr


def _step(row, held, target, cost_bps, equity, execute):
    desired = _target(target if execute else held)
    ending, multiple, fee, turnover, _, _ = scalar_settle(
        *held, *desired, row.soxl_return, row.soxx_return, cost_bps / 10000.)
    equity *= multiple
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("blocked: nonfinite portfolio NAV")
    record = {key: getattr(row, key) for key in DATES}
    record.update(strategy_return=multiple - 1., fees=fee, turnover=turnover,
                  cost_bps=cost_bps, equity=equity, execute_open_target=bool(execute))
    for i, asset in enumerate(ASSETS):
        record[f"pretrade_{asset}_weight"] = float(held[i])
        record[f"open_{asset}_weight"] = float(desired[i])
        record[f"ending_{asset}_weight"] = float(ending[i])
    return record, np.asarray(ending), equity


def _settings(cost_bps, rebalance_sessions):
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("blocked: invalid trading cost")
    if isinstance(rebalance_sessions, bool) or not isinstance(rebalance_sessions, int) or rebalance_sessions < 1:
        raise ValueError("blocked: rebalance_sessions must be a positive integer")


def replay_fixed_target(frame, *, target, cost_bps, rebalance_sessions, buy_and_hold=False):
    market = validate_market(frame)
    _settings(cost_bps, rebalance_sessions)
    desired = _target(target)
    held, equity, rows = np.array([0., 0., 1.]), 1., []
    for i, row in enumerate(market.itertuples()):
        execute = i == 0 or (not buy_and_hold and i % rebalance_sessions == 0)
        record, held, equity = _step(row, held, desired, cost_bps, equity, execute)
        rows.append(record)
    return validate_replay(pd.DataFrame(rows))


def replay_online_allocator(frame, expert_paths, *, allocator, cost_bps, rebalance_sessions):
    """Consume each expert's own mature net outcome; settle a separate account.

    Future paths may be supplied for batch replay but are never consumed until
    their explicit availability time is strictly before the current decision.
    """
    market = validate_market(frame)
    _settings(cost_bps, rebalance_sessions)
    names = allocator.config.expert_names
    if set(expert_paths) != set(names):
        raise ValueError("blocked: expert inventory differs from registered model")
    checked = {name: validate_replay(expert_paths[name]) for name in names}
    for path in checked.values():
        if ("cost_bps" not in path or not path[list(DATES)].equals(market[list(DATES)])
                or not path.cost_bps.eq(cost_bps).all()):
            raise ValueError("blocked: expert dates or costs differ")
    feedback = np.column_stack([checked[name].strategy_return for name in names])
    if "soxx" not in checked:
        raise ValueError("blocked: explicit soxx benchmark expert required")
    benchmark = checked["soxx"].strategy_return.to_numpy(float)
    held, equity, cursor = np.array([0., 0., 1.]), 1., 0
    rows, audits = [], []
    for i, row in enumerate(market.itertuples()):
        while cursor < len(market) and market.return_available_at.iloc[cursor] < row.decision_at:
            allocator.update(observation_id=str(market.signal_date.iloc[cursor].date()),
                available_at=market.return_available_at.iloc[cursor], decision_at=row.decision_at,
                expert_net_returns=feedback[cursor], benchmark_net_return=benchmark[cursor])
            cursor += 1
        desired = allocator.target(row.decision_at)
        execute = i % rebalance_sessions == 0
        record, held, equity = _step(row, held, desired, cost_bps, equity, execute)
        rows.append(record)
        audits.append({"signal_date": row.signal_date, "decision_at": row.decision_at,
            "feedback_count": cursor, "last_feedback_available_at": allocator.last_available_at,
            "execute_open_target": execute,
            **{f"posterior_{name}": float(allocator.posterior[j]) for j, name in enumerate(names)}})
    return validate_replay(pd.DataFrame(rows)), pd.DataFrame(audits)
