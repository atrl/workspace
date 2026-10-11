"""Generic matched-horizon replay entry (default 5) with the quote-after-9:31 clock.

The solver is injected and must return BOTH ``target_net_weights`` (post-fee normalized
target used for the risk check) and ``execution_target_weights`` (fee-affordable pre-fee
target actually settled) — a bare net vector is rejected. The target is applied at the
trade open (the allocation quote clock), never at the prior signal 16:15.

Accounting basis: this uses the verified NORMALIZED-account settlement helper
(``settle_target_session``); it reports weight-level cash/value, NOT actual broker cash or
integer quantities. It must not be described as an actual cash/quantity engine; real
cash/quantity reconciliation is deferred to the native actual-accounting /
target-schedule replay and will be reported as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.generic_horizon_replay_contract import check_covariance, validate_unit_tags
from src.backtest.position_accounting import settle_target_session
from src.data.generic_horizon import allocation_clock, strict_positive_int

REQUIRED = ("signal_date", "trade_date", "exit_date", "decision_at", "soxl_return", "soxx_return")
ACCOUNTING_BASIS = "normalized_research_weights_not_broker_qty"


def _simplex(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all() or (array < 0).any() or not np.isclose(array.sum(), 1.0):
        raise ValueError(f"blocked: {name} must be a funded three-asset simplex")
    return array


def _validate_frame(frame: pd.DataFrame, *, calendar=None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or not set(REQUIRED).issubset(frame.columns):
        raise ValueError(f"blocked: replay frame must carry {REQUIRED}")
    market = frame.copy()
    for column in ("signal_date", "trade_date", "exit_date"):
        market[column] = pd.to_datetime(market[column])
        if market[column].isna().any() or market[column].duplicated().any() or not market[column].is_monotonic_increasing:
            raise ValueError(f"blocked: {column} must be unique and ordered")
    if not (market["trade_date"] < market["exit_date"]).all():
        raise ValueError("blocked: trade_date must precede exit_date")
    if not (market["trade_date"].to_numpy()[:-1] == market["signal_date"].to_numpy()[1:]).all():
        raise ValueError("blocked: replay requires consecutive signal/trade sessions")
    if not (market["exit_date"].to_numpy()[:-1] == market["trade_date"].to_numpy()[1:]).all():
        raise ValueError("blocked: exit[i] must equal trade[i+1] (no compressed middle session)")
    if calendar is None:
        raise ValueError("blocked: an explicit authoritative calendar is required (missing middle days cannot be compressed)")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar))
    if sessions.hasnans or sessions.duplicated().any() or not sessions.is_monotonic_increasing:
        raise ValueError("blocked: authoritative calendar must be unique and ordered")
    covered = pd.DatetimeIndex(market["signal_date"]).union(market["trade_date"]).union(market["exit_date"])
    span = sessions[(sessions >= market["signal_date"].iloc[0]) & (sessions <= market["exit_date"].iloc[-1])]
    if not pd.DatetimeIndex(span).equals(pd.DatetimeIndex(covered).sort_values()):
        raise ValueError("blocked: replay rows do not cover every authoritative calendar session in span")
    returns = market[["soxl_return", "soxx_return"]].to_numpy(float)
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("blocked: invalid complete market returns")
    decision = pd.to_datetime(market["decision_at"], utc=True)
    expected = allocation_clock(market["trade_date"])
    if not decision.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError("blocked: decision_at must equal the trade-open quote clock (after 09:31), not prior 16:15")
    return market


def _validate_solution(solution) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(solution, dict) or "target_net_weights" not in solution or "execution_target_weights" not in solution:
        raise ValueError("blocked: solver must return target_net_weights and execution_target_weights")
    net = _simplex(solution["target_net_weights"], "target_net_weights")
    execution = _simplex(solution["execution_target_weights"], "execution_target_weights")
    return net, execution


def replay_matched_horizon(frame: pd.DataFrame, expected_returns, mean_variances, daily_covariances, *,
                           solver, calendar, horizon_sessions: int = 5, cost_bps: float = 10.0,
                           unit_tags: dict = None, buy_and_hold: bool = False) -> tuple[pd.DataFrame, dict]:
    """Replay a matched-horizon path with the injected two-target ``solver``."""
    horizon_sessions = strict_positive_int(horizon_sessions)
    if type(buy_and_hold) is not bool:
        raise ValueError("blocked: buy_and_hold must be a boolean")
    if not callable(solver):
        raise ValueError("blocked: solver must be callable")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("blocked: invalid trading cost")
    if unit_tags is None:
        raise ValueError("blocked: unit tags are mandatory for a matched-horizon replay")
    validate_unit_tags(unit_tags, horizon_sessions=horizon_sessions)
    market = _validate_frame(frame, calendar=calendar)
    means = np.asarray(expected_returns, dtype=float)
    variances = np.asarray(mean_variances, dtype=float)
    dailies = np.asarray(daily_covariances, dtype=float)
    rows, risk_exceeded = [], 0
    hold_risk_exceeded = 0
    if means.shape != (len(market), 2) or not np.isfinite(means).all() or (means <= -1).any():
        raise ValueError("blocked: expected returns must be finite, > -1, for all rows including HOLD")
    # reuse the same symmetric+PSD validator for ALL rows (including HOLD)
    check_covariance(variances, name="mean error covariance", sessions=len(market), atol=1e-10,
                     require_positive_diagonal=False)
    check_covariance(dailies, name="daily covariance", sessions=len(market), atol=1e-10,
                     require_positive_diagonal=False)

    held, equity = np.array([0.0, 0.0, 1.0]), 1.0
    for index, row in enumerate(market.itertuples()):
        execute = index == 0 if buy_and_hold else index % horizon_sessions == 0
        sigma, budget = dailies[index][:2, :2], float(dailies[index][1, 1])
        if execute:
            net_target, execution_target = _validate_solution(
                solver(held.copy(), means[index].copy(), variances[index].copy(), dailies[index].copy()))
            held_variance = float(net_target[:2] @ sigma @ net_target[:2])
            exceeds = held_variance > budget * (1 + 1e-8) if budget > 0 else held_variance > 1e-20
            if exceeds:
                raise ValueError("blocked: rebalance target exceeds the SOXX variance budget")
            risk_exceeded += int(exceeds)
        else:
            net_target, execution_target = held.copy(), None
            hold_variance = float(held[:2] @ sigma @ held[:2])
            hold_exceeds = hold_variance > budget * (1 + 1e-8) if budget > 0 else hold_variance > 1e-20
            hold_risk_exceeded += int(hold_exceeds)
            held_variance, exceeds = hold_variance, hold_exceeds
        settled = settle_target_session(held, open_target_weights=execution_target,
                                        asset_returns=[row.soxl_return, row.soxx_return, 0.0], cost_bps=cost_bps)
        net_check = np.max(np.abs(settled["open_net_weights"] - net_target)) if execute else 0.0
        if execute and net_check > 1e-10:
            raise ValueError("blocked: execution_target_weights do not settle to target_net_weights (fee affordability)")
        if not execute and (settled["fees"] != 0 or settled["turnover"] != 0):
            raise ValueError("blocked: exact HOLD unexpectedly traded")
        ending = settled["ending_weights"]
        fee_currency = equity * settled["fees"]
        equity = equity * settled["net_equity"]
        record = {"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
                     "execute": bool(execute), "net_return": float(settled["net_return"]), "fees": float(settled["fees"]),
                     "turnover": float(settled["turnover"]), "ending_soxl_weight": float(ending[0]),
                     "ending_soxx_weight": float(ending[1]), "ending_cash_weight": float(ending[2]),
                     "ending_values": settled["ending_values"], "session_net_equity": float(settled["net_equity"]),
                     "cumulative_equity": float(equity), "equity": float(equity),
                     "strategy_return": float(settled["net_return"]), "cost_bps": float(cost_bps),
                     "decision_at": row.decision_at, "execute_open_target": bool(execute),
                     "execute_open_trade": bool(settled["turnover"] > 1e-12),
                     "fee_currency_initial_nav_one": float(fee_currency),
                     "portfolio_variance": held_variance, "benchmark_variance": budget,
                     "estimated_risk_exceeds_benchmark": bool(exceeds),
                     "risk_constraint_enforced": bool(execute)}
        for asset, pretrade, opening in zip(("soxl", "soxx", "cash"), held, net_target):
            record[f"pretrade_{asset}_weight"] = float(pretrade)
            record[f"open_{asset}_weight"] = float(opening)
        rows.append(record)
        held = ending.copy()
    path = pd.DataFrame(rows)
    values_ok = bool(np.allclose([v.sum() for v in path["ending_values"]], path["session_net_equity"], atol=1e-12))
    return path, {
        "kind": "replay_entry",
        "accounting_basis": ACCOUNTING_BASIS,
        "broker_quantity_reconciled": False,
        "horizon_sessions": horizon_sessions,
        "buy_and_hold": buy_and_hold,
        "cost_bps": float(cost_bps),
        "sessions": int(len(path)),
        "target_decision_calls": int(path["execute"].sum()),
        "trades": int(path["execute_open_trade"].sum()),
        "hold_sessions": int((~path["execute"]).sum()),
        "total_fees": float(path["fees"].sum()),
        "total_fees_unit": "sum_of_session_fee_fractions_not_currency",
        "final_cumulative_equity": float(equity),
        "replay_day_completeness": bool(len(path) == len(market)),
        "cash_plus_stocks_equals_equity": values_ok,
        "hold_requires_zero_fee": bool((path.loc[~path["execute"], "fees"] == 0).all()),
        "rebalance_risk_exceedance_sessions": int(risk_exceeded),
        "mid_hold_risk_drift_exceedance_sessions": int(hold_risk_exceeded),
        "allocation_clock": "next_open_after_09:31_America/New_York",
        "unit_tags_validated": True,
    }
