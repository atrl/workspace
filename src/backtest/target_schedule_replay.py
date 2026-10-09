"""Dated daily targets with owned positions and independent cash-fee checks.

This reconciliation ledger must still pass a native Qlib portfolio replay.
Targets are desired post-fee NAV weights, not pre-fee asset dollar allocations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import ASSETS, validate_replay
from src.backtest.benchmark_relative_replay import _settings, _step, validate_market
from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import fee_affordable_target


def replay_target_schedule(frame, targets: pd.DataFrame, *, cost_bps):
    """Replay every native signal session; never silently align or drop targets."""
    market = validate_market(frame)
    _settings(cost_bps, 1)
    if (not isinstance(targets, pd.DataFrame) or targets.columns.has_duplicates
            or tuple(targets.columns) != ASSETS or targets.index.has_duplicates
            or not isinstance(targets.index, pd.DatetimeIndex)
            or not targets.index.equals(pd.DatetimeIndex(market.signal_date))):
        raise ValueError("blocked: targets must have exact native signal dates and SOXL/SOXX/cash columns")
    values = targets.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError("blocked: daily targets must be complete funded simplexes")
    held, equity, rows, maximum = np.array([0., 0., 1.]), 1., [], 0.
    for number, row in enumerate(market.itertuples()):
        target = values[number]
        record, ending, next_equity = _step(row, held, target, cost_bps, equity, True)
        executable = fee_affordable_target(target, pretrade_weights=held, cost_bps=cost_bps)
        independent = settle_target_session(held, open_target_weights=executable,
            asset_returns=[row.soxl_return, row.soxx_return, 0.], cost_bps=cost_bps)
        error = max(float(np.max(np.abs(independent["open_net_weights"] - target))),
            float(np.max(np.abs(independent["ending_weights"] - ending))),
            abs(independent["net_return"] - record["strategy_return"]),
            abs(independent["fees"] - record["fees"]), abs(independent["turnover"] - record["turnover"]))
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError("blocked: daily target independent funding or settlement differs")
        maximum = max(maximum, error)
        record.update(decision_at=row.decision_at, return_available_at=row.return_available_at,
                      execute_open_trade=bool(record["turnover"] > 1e-12))
        for asset, weight in zip(ASSETS, independent["open_net_weights"]):
            record[f"settled_open_{asset}_weight"] = float(weight)
        rows.append(record)
        held, equity = ending.copy(), next_equity
    return validate_replay(pd.DataFrame(rows)), {"sessions": len(rows), "cost_bps": float(cost_bps),
        "rebalance_sessions": 1, "maximum_independent_accounting_error": maximum,
        "own_positions": True, "current_fees_only": True, "terminal_liquidation": False,
        "native_qlib_verified": False}
