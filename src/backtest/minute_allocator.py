"""Independent per-cost settlement of the existing allocator's EMA/HOLD policy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.continuous_target_adapter import native_open_reconciliation
from src.analysis.integrated_allocation import validate_replay
from src.backtest.benchmark_relative_replay import _step, replay_fixed_target, validate_market
from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import fee_affordable_target


WEIGHTS = ("soxl_weight", "soxx_weight", "cash_weight")


def settle_allocator_targets(market: pd.DataFrame, predictions: pd.DataFrame, *, cost_bps: float,
                              smoothing_span: int, rebalance_threshold: float) -> tuple[pd.DataFrame, dict]:
    """Each cost owns its quantities and threshold decisions; HOLD sends no order."""
    market = validate_market(market)
    dates = pd.DatetimeIndex(market.signal_date)
    if predictions.index.has_duplicates or not dates.isin(predictions.index).all():
        raise ValueError("needs_data: raw allocator predictions do not cover every native signal session")
    targets = predictions.reindex(dates).loc[:, list(WEIGHTS)].to_numpy(float)
    if (not np.isfinite(targets).all() or (targets < 0).any()
            or not np.allclose(targets.sum(axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError("blocked: raw allocator predictions are not funded simplexes")
    if smoothing_span < 1 or not 0 <= rebalance_threshold < 2:
        raise ValueError("blocked: invalid fixed allocator execution settings")
    alpha = 2. / (1. + smoothing_span)
    smoothed = None
    held, equity, rows, maximum = np.array([0., 0., 1.]), 1., [], 0.
    for position, row in enumerate(market.itertuples()):
        raw = targets[position]
        smoothed = raw.copy() if smoothed is None else alpha * raw + (1. - alpha) * smoothed
        smoothed /= smoothed.sum()
        execute = bool(np.abs(smoothed[:2] - held[:2]).sum() >= rebalance_threshold)
        desired = smoothed.copy() if execute else held.copy()
        record, ending, next_equity = _step(row, held, desired, cost_bps, equity, execute)
        funded = fee_affordable_target(desired, pretrade_weights=held, cost_bps=cost_bps)
        independent = settle_target_session(held, open_target_weights=funded,
            asset_returns=[row.soxl_return, row.soxx_return, 0.], cost_bps=cost_bps)
        error = max(abs(independent["net_return"] - record["strategy_return"]),
                    abs(independent["fees"] - record["fees"]),
                    float(np.max(np.abs(independent["ending_weights"] - ending))))
        if error > 1e-10 or (not execute and record["turnover"] > 1e-10):
            raise ValueError("blocked: allocator funding or exact HOLD accounting differs")
        maximum = max(maximum, error)
        record.update(decision_at=row.decision_at, return_available_at=row.return_available_at)
        for number, asset in enumerate(("soxl", "soxx", "cash")):
            record["ema_" + asset + "_weight"] = float(smoothed[number])
        rows.append(record)
        held, equity = ending.copy(), next_equity
    return validate_replay(pd.DataFrame(rows)), {"maximum_independent_accounting_error": maximum,
        "own_cost_cash_positions_and_threshold_decisions": True, "smoothing_span": smoothing_span,
        "rebalance_threshold": rebalance_threshold, "source_unknown_rows_retained": True}


def native_allocator_book(market: pd.DataFrame, predictions: pd.DataFrame | None, *, provider_uri,
                           cost_bps: float, smoothing_span: int, rebalance_threshold: float,
                           benchmark: bool = False) -> dict:
    if benchmark:
        replay = replay_fixed_target(market, target=[0., 1., 0.], cost_bps=cost_bps,
                                      rebalance_sessions=1, buy_and_hold=True)
        audit = {"own_cost_cash_positions_and_threshold_decisions": True, "buy_and_hold": True}
    else:
        replay, audit = settle_allocator_targets(market, predictions, cost_bps=cost_bps,
            smoothing_span=smoothing_span, rebalance_threshold=rebalance_threshold)
    native = native_open_reconciliation(market, replay, provider_uri=provider_uri, cost_bps=cost_bps)
    if native["audit"]["status"] != "passed":
        raise ValueError("blocked: allocator native Qlib next-open reconciliation failed")
    authority = replay.copy()
    authority["strategy_return"] = native["reconciliation"].native_next_open_return.to_numpy(float)
    authority["equity"] = native["reconciliation"].native_next_open_nav.to_numpy(float) / native["audit"]["initial_cash"]
    return {"authority": validate_replay(authority), "independent_replay": replay,
            "native": native, "settlement_audit": audit}
