"""Fixed-path fee vs drawdown audit (0 fit, 0 strategy, 0 orders).

Re-settles the SAME recorded target/flag path at 0/10/25bp (controller NOT re-run) and
reports the global true drawdown (running peak includes the initial 1.0), the
drawdown_before series, the per-row running-peak capital date, and the actual emergency
days whose fixed-0 path still has drawdown >= the frozen emergency threshold.

Descriptive only: no additive DD causal decomposition and no unique market-vs-fee
attribution. Fees are deducted once. Step B (fee-neutral guard) is not implemented.
The emergency threshold is an explicit parameter (from the frozen plan/source spec), not
a magic default presented as a risk budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_CONVENTION = "post_fee_desired"
LEGACY_CONVENTION = "pre_fee_account"
CONVENTIONS = (RAW_CONVENTION, LEGACY_CONVENTION)
REQUIRED = ("signal_date", "trade_date", "desired_soxl", "desired_soxx", "desired_cash",
            "recovery_soxl", "recovery_soxx", "recovery_cash", "execute", "recovery_action",
            "soxl_return", "soxx_return", "early_ret_soxl", "early_ret_soxx")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _bool_ok(series: pd.Series) -> bool:
    if series.dtype == object or series.isna().any():
        return False
    return set(pd.unique(series)) <= {True, False, 0, 1, 0.0, 1.0}


def _validate_base(base: pd.DataFrame, *, convention: str) -> None:
    missing = [c for c in REQUIRED if c not in base.columns]
    _require(not missing, f"missing columns {missing}")
    _require(convention in CONVENTIONS, f"unsupported convention {convention}")
    for column in ("execute", "recovery_action"):
        _require(_bool_ok(base[column]), f"non-bool flag {column}")
    for column in ("soxl_return", "soxx_return"):
        values = pd.to_numeric(base[column], errors="coerce").to_numpy(float)
        _require(np.isfinite(values).all(), f"nonfinite {column}")
        _require((values > -1.0).all(), f"{column} <= -1")
    execute = base["execute"].astype(bool).to_numpy()
    desired = base[["desired_soxl", "desired_soxx", "desired_cash"]].to_numpy(float)
    _require(np.isfinite(desired[execute]).all(), "missing desired target while execute=True")
    trade = pd.to_datetime(base["trade_date"], errors="coerce")
    _require(trade.notna().all() and not trade.duplicated().any() and trade.is_monotonic_increasing,
             "trade_date not unique/ordered")
    if "exit_date" in base.columns:
        exit_ = pd.to_datetime(base["exit_date"], errors="coerce")
        _require(exit_.notna().all() and not exit_.duplicated().any() and exit_.is_monotonic_increasing,
                 "exit_date not unique/ordered")
        _require(bool((exit_.iloc[:-1].reset_index(drop=True)
                       == trade.iloc[1:].reset_index(drop=True)).all()), "exit_date not linked to next trade_date")


def resettle_preset_path(base: pd.DataFrame, *, settlement_cost: float,
                         convention: str = LEGACY_CONVENTION) -> pd.DataFrame:
    """Re-settle the recorded path from fresh cash at ``settlement_cost`` (same targets/flags)."""
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    _validate_base(base, convention=convention)
    previous = np.array([0.0, 0.0, 1.0])
    rows = []
    for _, r in base.iterrows():
        desired = (np.array([r["desired_soxl"], r["desired_soxx"], r["desired_cash"]])
                   if np.isfinite(r["desired_soxl"]) else None)
        target = desired
        if target is not None and convention == RAW_CONVENTION:
            target = fee_affordable_target(target, pretrade_weights=previous, cost_bps=settlement_cost)
        rec = (np.array([r["recovery_soxl"], r["recovery_soxx"], r["recovery_cash"]])
               if bool(r["recovery_action"]) else None)
        settled = settle_target_session(
            previous, open_target_weights=(target if bool(r["execute"]) else None),
            asset_returns=[r["soxl_return"], r["soxx_return"], 0.0], cost_bps=settlement_cost,
            recovery_target_weights=rec,
            early_returns=[r["early_ret_soxl"], r["early_ret_soxx"], 0.0] if rec is not None else None)
        previous = settled["ending_weights"]
        rows.append({"signal_date": r["signal_date"], "trade_date": r["trade_date"],
                     "exit_date": r.get("exit_date"), "strategy_return": settled["net_return"],
                     "fees": settled["fees"], "turnover": settled["turnover"],
                     "ending_soxl_weight": previous[0], "ending_soxx_weight": previous[1],
                     "ending_cash_weight": previous[2], "settlement_cost_bps": settlement_cost})
    return pd.DataFrame(rows)


def base_parity(base: pd.DataFrame, resettled: pd.DataFrame, *, tol: float = 1e-12) -> dict:
    """Day-by-day parity of a base re-settlement vs the recorded base path."""
    if len(base) != len(resettled):
        return {"status": "failed", "reason": "length"}
    a = base.reset_index(drop=True)
    b = resettled.reset_index(drop=True)
    date_checks = {}
    for column in ("signal_date", "trade_date", "exit_date"):
        if column in a.columns and column in b.columns and a[column].notna().all() and b[column].notna().all():
            date_checks[column] = bool(np.array_equal(pd.to_datetime(a[column]).to_numpy(),
                                                      pd.to_datetime(b[column]).to_numpy()))
    checks = {**date_checks,
              "strategy_return": bool(np.allclose(a["strategy_return"], b["strategy_return"], rtol=0, atol=tol)),
              "fees": bool(np.allclose(a["fees"], b["fees"], rtol=0, atol=tol)),
              "ending_weights": bool(all(np.allclose(a[c], b[c], rtol=0, atol=tol) for c in
                                         ("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight")))}
    return {"status": "passed" if all(checks.values()) else "failed", **checks}


def drawdown_stats(daily: pd.DataFrame, *, emergency_threshold: float) -> dict:
    """Global true DD + drawdown_before + per-row running-peak capital date + emergency episodes.

    Running peak includes the initial 1.0. ``peak_capital_date[i]`` is the capital date of
    the running peak used to evaluate session ``i``: None while it is still the initial
    capital, otherwise the ``exit_date`` of the last session that set a new peak (that
    return matures at the next ``trade_date`` and is NEVER brought forward to its own
    trade_date). Raises (blocks) on empty / nonfinite / <= -1 input rather than returning NaN.
    """
    _require(isinstance(daily, pd.DataFrame) and not daily.empty, "empty daily path")
    r = pd.to_numeric(daily["strategy_return"], errors="coerce").to_numpy(float)
    _require(np.isfinite(r).all(), "nonfinite strategy_return")
    _require((r > -1.0).all(), "strategy_return <= -1")
    dates = pd.to_datetime(daily["trade_date"]).reset_index(drop=True)
    exit_dates = (pd.to_datetime(daily["exit_date"]).reset_index(drop=True) if "exit_date" in daily.columns
                  else dates)
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    prior_peak = np.maximum.accumulate(wealth[:-1])
    dd_before = 1.0 - wealth[:-1] / prior_peak

    # per-row peak capital date: last session j<i whose post-session wealth set a new peak
    peak_capital_date, updates = [], []
    last_peak = 1.0
    last_peak_date = None
    running = [1.0]
    for i in range(len(r)):
        peak_capital_date.append(last_peak_date)
        if wealth[i + 1] > last_peak + 1e-15:
            last_peak = wealth[i + 1]
            last_peak_date = str(exit_dates.iloc[i].date())
            updates.append({"date": last_peak_date, "amount": float(last_peak)})

    emergency_mask = dd_before >= emergency_threshold
    episodes, start = [], None
    for i, flag in enumerate(emergency_mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append({"start": str(dates.iloc[start].date()), "end": str(dates.iloc[i - 1].date()),
                             "days": i - start})
            start = None
    if start is not None:
        episodes.append({"start": str(dates.iloc[start].date()), "end": str(dates.iloc[-1].date()),
                         "days": len(emergency_mask) - start})
    return {"emergency_threshold": float(emergency_threshold),
            "true_max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
            "sessions": int(len(r)),
            "drawdown_before": dd_before,
            "peak_capital_date": peak_capital_date,
            "peak_updates": updates,
            "emergency_days": int(emergency_mask.sum()),
            "emergency_share": float(emergency_mask.mean()),
            "emergency_longest_days": int(max((e["days"] for e in episodes), default=0)),
            "emergency_episodes": episodes}
