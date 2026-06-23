from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.workflow.agentic_alpha import AGENTIC_ALPHA_STAGE_ORDER


SESSION_EFFECT_SOURCE_FIELDS = [
    "SOXL_OVERNIGHT_RETURN",
    "SOXL_INTRADAY_RETURN",
    "SOXL_TUG_OF_WAR",
    "SOXX_TREND",
    "SOXL_VIX",
    "SOXL_RANGE",
]
SESSION_EFFECT_BRANCHES = [
    "overnight_momentum",
    "intraday_gap_reversal",
    "tug_of_war_close_to_close",
]


def _wide_qlib_panel(frame: pd.DataFrame, *, symbols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", "$open", "$high", "$low", "$close"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"session-effect qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()

    pieces: list[pd.DataFrame] = []
    for symbol in [item.upper() for item in symbols]:
        symbol_frame = normalized.loc[normalized["instrument"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"session-effect qlib frame missing instrument: {symbol}")
        columns = {
            "$open": f"{symbol.lower()}_open",
            "$high": f"{symbol.lower()}_high",
            "$low": f"{symbol.lower()}_low",
            "$close": f"{symbol.lower()}_close",
        }
        if "$volume" in symbol_frame:
            columns["$volume"] = f"{symbol.lower()}_volume"
        if "$vix" in symbol_frame:
            columns["$vix"] = f"{symbol.lower()}_vix"
        piece = symbol_frame[["datetime", *columns]].rename(columns=columns).groupby("datetime", as_index=False).last()
        pieces.append(piece)

    panel = pieces[0]
    for piece in pieces[1:]:
        panel = panel.merge(piece, on="datetime", how="inner")
    panel = panel.sort_values("datetime").set_index("datetime")
    if "soxl_vix" in panel:
        panel["vix"] = panel["soxl_vix"].ffill()
    return panel


def build_session_effect_frame(
    qlib_frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    min_history: int = 20,
) -> pd.DataFrame:
    """Build close-to-open, open-to-close, and tug-of-war session features."""

    panel = _wide_qlib_panel(qlib_frame, symbols=["SOXL", "SOXX"])
    soxl_open = panel["soxl_open"].astype("float64")
    soxl_high = panel["soxl_high"].astype("float64")
    soxl_low = panel["soxl_low"].astype("float64")
    soxl_close = panel["soxl_close"].astype("float64")
    soxx_close = panel["soxx_close"].astype("float64")
    vix = panel["vix"].astype("float64") if "vix" in panel else pd.Series(np.nan, index=panel.index)

    overnight = soxl_open / soxl_close.shift(1) - 1.0
    intraday = soxl_close / soxl_open - 1.0
    close_return = soxl_close.pct_change()
    features = pd.DataFrame(
        {
            "datetime": panel.index,
            "overnight_return": overnight,
            "intraday_return": intraday,
            "next_overnight_return": soxl_open.shift(-1) / soxl_close - 1.0,
            "next_intraday_return": soxl_close.shift(-1) / soxl_open.shift(-1) - 1.0,
            "next_close_return": soxl_close.shift(-1) / soxl_close - 1.0,
            "vix": vix,
            "range20": ((soxl_high - soxl_low) / soxl_close.shift(1)).rolling(
                20,
                min_periods=min(20, min_history),
            ).mean(),
        }
    )
    for window in [3, 5, 10, 20, 40, 60]:
        min_periods = min(window, min_history)
        features[f"overnight_mom{window}"] = overnight.rolling(window, min_periods=min_periods).mean()
        features[f"intraday_mom{window}"] = intraday.rolling(window, min_periods=min_periods).mean()
        features[f"tug{window}"] = (overnight - intraday).rolling(window, min_periods=min_periods).mean()
    for window in [20, 60, 120]:
        features[f"soxx_trend{window}"] = soxx_close / soxx_close.shift(window) - 1.0
        features[f"soxl_trend{window}"] = soxl_close / soxl_close.shift(window) - 1.0
    features["hv20"] = close_return.rolling(20, min_periods=min(20, min_history)).std() * np.sqrt(252)

    required = [
        "next_overnight_return",
        "next_intraday_return",
        "next_close_return",
        "overnight_mom3",
        "tug5",
        "soxx_trend20",
        "vix",
    ]
    result = features.replace([np.inf, -np.inf], np.nan)
    result = result.loc[(result["datetime"] >= pd.Timestamp(start)) & (result["datetime"] <= pd.Timestamp(end))]
    result = result.dropna(subset=required).reset_index(drop=True)
    result.attrs["source_fields"] = SESSION_EFFECT_SOURCE_FIELDS
    result.attrs["source_coverage_start"] = panel.index.min().date().isoformat()
    result.attrs["source_coverage_end"] = panel.index.max().date().isoformat()
    return result


def _strategy_stats(
    returns: np.ndarray,
    exposure: np.ndarray,
    *,
    cost_per_turnover: float,
) -> dict[str, float]:
    exposure = np.nan_to_num(np.asarray(exposure, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    daily_turnover = np.abs(np.diff(np.r_[0.0, exposure]))
    strategy_returns = np.nan_to_num(np.asarray(returns, dtype="float64"), nan=0.0) * exposure
    strategy_returns = strategy_returns - daily_turnover * float(cost_per_turnover)
    if np.any(strategy_returns <= -0.99):
        return {
            "return_pct": -100.0,
            "max_drawdown_pct": -100.0,
            "turnover": float(daily_turnover.sum()),
            "avg_abs_exposure": float(np.abs(exposure).mean()) if len(exposure) else 0.0,
            "max_exposure": float(exposure.max()) if len(exposure) else 0.0,
            "trade_count": int(np.count_nonzero(exposure)),
        }
    equity = np.cumprod(1.0 + strategy_returns)
    if len(equity) == 0:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "turnover": 0.0,
            "avg_abs_exposure": 0.0,
            "max_exposure": 0.0,
            "trade_count": 0,
        }
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "return_pct": float((equity[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "turnover": float(daily_turnover.sum()),
        "avg_abs_exposure": float(np.abs(exposure).mean()),
        "max_exposure": float(exposure.max()),
        "trade_count": int(np.count_nonzero(exposure)),
    }


def _decision(return_pct: float, drawdown_pct: float, baseline_return_pct: float, max_drawdown_floor: float) -> tuple[str, str]:
    beats = float(return_pct) > float(baseline_return_pct)
    passes_drawdown = float(drawdown_pct) >= float(max_drawdown_floor)
    if not passes_drawdown:
        return "Drop", "drawdown_gate"
    if not beats:
        return "Iterate", "return_gate"
    return "Promote", "beats_baseline_and_drawdown_gate"


def evaluate_session_effect_candidates(
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    overnight_windows: list[int] | None = None,
    overnight_thresholds: list[float] | None = None,
    trend_filters: list[str] | None = None,
    vix_maxes: list[float] | None = None,
    gap_thresholds: list[float] | None = None,
    gap_modes: list[str] | None = None,
    tug_windows: list[int] | None = None,
    tug_thresholds: list[float] | None = None,
    tug_directions: list[str] | None = None,
    scales: list[float] | None = None,
    cost_per_turnover: float = 0.0002,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("session-effect frame is empty")
    data = frame.sort_values("datetime").reset_index(drop=True).copy()
    overnight_windows = overnight_windows or [3, 5, 10, 20, 40, 60]
    overnight_thresholds = overnight_thresholds or [-0.01, -0.005, 0.0, 0.002, 0.005, 0.01]
    trend_filters = trend_filters or ["none", "soxx20", "soxx60", "soxl20", "soxl60"]
    vix_maxes = vix_maxes or [22.0, 26.0, 30.0, 35.0, 45.0, 100.0]
    gap_thresholds = gap_thresholds or [-0.12, -0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08]
    gap_modes = gap_modes or ["buy_gap_down", "buy_gap_up", "dip_in_trend"]
    tug_windows = tug_windows or [5, 10, 20, 40, 60]
    tug_thresholds = tug_thresholds or [-0.01, -0.005, 0.0, 0.005, 0.01, 0.02]
    tug_directions = tug_directions or ["positive", "negative"]
    scales = scales or [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 40.0, 60.0, 100.0]

    returns = {
        "close_to_next_open": data["next_overnight_return"].astype("float64").to_numpy(),
        "open_to_close": data["next_intraday_return"].astype("float64").to_numpy(),
        "close_to_next_close": data["next_close_return"].astype("float64").to_numpy(),
    }
    rows: list[dict[str, Any]] = []

    def apply_trend_filter(signal: pd.Series, trend_filter: str) -> pd.Series:
        if trend_filter == "none":
            return signal
        if trend_filter == "soxx20":
            return signal & (data["soxx_trend20"] > 0.0)
        if trend_filter == "soxx60":
            return signal & (data["soxx_trend60"] > 0.0)
        if trend_filter == "soxl20":
            return signal & (data["soxl_trend20"] > 0.0)
        if trend_filter == "soxl60":
            return signal & (data["soxl_trend60"] > 0.0)
        raise ValueError(f"unknown trend_filter: {trend_filter}")

    def add_row(branch: str, session: str, signal: pd.Series, scale: float, params: dict[str, Any]) -> None:
        exposure = signal.astype("float64").to_numpy() * float(scale)
        stats = _strategy_stats(returns[session], exposure, cost_per_turnover=cost_per_turnover)
        decision, decision_basis = _decision(
            stats["return_pct"],
            stats["max_drawdown_pct"],
            baseline_return_pct,
            max_drawdown_floor,
        )
        rows.append(
            {
                "branch": branch,
                "candidate": f"{branch}_{session}_scale{float(scale):g}_{len(rows)}",
                "session": session,
                "return_pct": float(stats["return_pct"]),
                "baseline_return_pct": float(baseline_return_pct),
                "return_improvement_pct": float(stats["return_pct"] - baseline_return_pct),
                "max_drawdown_pct": float(stats["max_drawdown_pct"]),
                "max_drawdown_floor": float(max_drawdown_floor),
                "turnover": float(stats["turnover"]),
                "avg_abs_exposure": float(stats["avg_abs_exposure"]),
                "max_exposure": float(stats["max_exposure"]),
                "trade_count": int(stats["trade_count"]),
                "beats_baseline_return": bool(stats["return_pct"] > float(baseline_return_pct)),
                "passes_drawdown_gate": bool(stats["max_drawdown_pct"] >= float(max_drawdown_floor)),
                "decision": decision,
                "decision_basis": decision_basis,
                "params": json.dumps(params, sort_keys=True),
                "coverage_start": pd.Timestamp(data["datetime"].iloc[0]).date().isoformat(),
                "coverage_end": pd.Timestamp(data["datetime"].iloc[-1]).date().isoformat(),
            }
        )

    for window in overnight_windows:
        column = f"overnight_mom{window}"
        if column not in data:
            continue
        for threshold in overnight_thresholds:
            base_signal = data[column] > float(threshold)
            for trend_filter in trend_filters:
                trend_signal = apply_trend_filter(base_signal, trend_filter)
                for vix_max in vix_maxes:
                    signal = trend_signal & (data["vix"] < float(vix_max))
                    for scale in scales:
                        add_row(
                            "overnight_momentum",
                            "close_to_next_open",
                            signal,
                            scale,
                            {
                                "window": window,
                                "threshold": float(threshold),
                                "trend_filter": trend_filter,
                                "vix_max": float(vix_max),
                                "scale": float(scale),
                            },
                        )

    for gap_threshold in gap_thresholds:
        for mode in gap_modes:
            if mode == "buy_gap_down":
                base_signal = data["overnight_return"] < float(gap_threshold)
            elif mode == "buy_gap_up":
                base_signal = data["overnight_return"] > float(gap_threshold)
            elif mode == "dip_in_trend":
                base_signal = (data["overnight_return"] < float(gap_threshold)) & (data["soxx_trend20"] > 0.0)
            else:
                raise ValueError(f"unknown gap_mode: {mode}")
            for vix_max in vix_maxes:
                signal = base_signal & (data["vix"] < float(vix_max))
                for scale in scales:
                    add_row(
                        "intraday_gap_reversal",
                        "open_to_close",
                        signal,
                        scale,
                        {
                            "gap_threshold": float(gap_threshold),
                            "mode": mode,
                            "vix_max": float(vix_max),
                            "scale": float(scale),
                        },
                    )

    for window in tug_windows:
        column = f"tug{window}"
        if column not in data:
            continue
        for threshold in tug_thresholds:
            for direction in tug_directions:
                if direction == "positive":
                    base_signal = data[column] > float(threshold)
                elif direction == "negative":
                    base_signal = data[column] < float(threshold)
                else:
                    raise ValueError(f"unknown tug_direction: {direction}")
                for vix_max in vix_maxes:
                    signal = base_signal & (data["vix"] < float(vix_max))
                    for scale in scales:
                        add_row(
                            "tug_of_war_close_to_close",
                            "close_to_next_close",
                            signal,
                            scale,
                            {
                                "window": window,
                                "threshold": float(threshold),
                                "direction": direction,
                                "vix_max": float(vix_max),
                                "scale": float(scale),
                            },
                        )

    return pd.DataFrame(rows)


def summarize_session_effect_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"session-effect branch has no candidate rows: {branch}")
    best_return = rows.sort_values("return_pct", ascending=False).iloc[0]
    drawdown_gate = rows.loc[rows["passes_drawdown_gate"] == True].sort_values(  # noqa: E712
        "return_pct",
        ascending=False,
    )
    best_drawdown_gate = drawdown_gate.iloc[0] if not drawdown_gate.empty else None
    passers = rows.loc[rows["beats_baseline_return"] & rows["passes_drawdown_gate"]]
    return {
        "grid_rows": int(len(rows)),
        "coverage_start": str(best_return.get("coverage_start")),
        "coverage_end": str(best_return.get("coverage_end")),
        "goal_baseline_return_pct": float(baseline_return_pct),
        "max_drawdown_floor": float(max_drawdown_floor),
        "passes_both_count": int(len(passers)),
        "best_return_pct": float(best_return["return_pct"]),
        "best_return_max_drawdown_pct": float(best_return["max_drawdown_pct"]),
        "best_return_candidate": str(best_return["candidate"]),
        "best_return_trade_count": int(best_return["trade_count"]),
        "best_drawdown_gate_available": bool(best_drawdown_gate is not None),
        "best_drawdown_gate_return_pct": (
            float(best_drawdown_gate["return_pct"]) if best_drawdown_gate is not None else None
        ),
        "best_drawdown_gate_max_drawdown_pct": (
            float(best_drawdown_gate["max_drawdown_pct"]) if best_drawdown_gate is not None else None
        ),
        "best_drawdown_gate_candidate": str(best_drawdown_gate["candidate"]) if best_drawdown_gate is not None else None,
        "best_drawdown_gate_trade_count": int(best_drawdown_gate["trade_count"]) if best_drawdown_gate is not None else None,
    }


def compact_session_effect_summary(
    metrics: pd.DataFrame,
    *,
    branches: list[str],
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("session-effect metrics are empty")
    branch_results = {
        branch: summarize_session_effect_branch(
            metrics,
            branch=branch,
            baseline_return_pct=baseline_return_pct,
            max_drawdown_floor=max_drawdown_floor,
        )
        for branch in branches
    }
    passers = metrics.loc[metrics["beats_baseline_return"] & metrics["passes_drawdown_gate"]]
    best_candidate = None
    if not passers.empty:
        best_candidate = str(passers.sort_values("return_pct", ascending=False).iloc[0]["candidate"])
    return {
        "workflow": "agentic_alpha_session_effect_search",
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "goal_achieved": bool(best_candidate is not None),
        "best_candidate": best_candidate,
        "passes_both_count": int(len(passers)),
        "branch_results": branch_results,
        "selected_features": SESSION_EFFECT_SOURCE_FIELDS,
    }
