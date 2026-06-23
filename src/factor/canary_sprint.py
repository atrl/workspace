from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


CANARY_SPRINT_SOURCE_FIELDS = [
    "SOXX_TREND_20",
    "SOXX_TREND_60",
    "SOXL_DIP_2",
    "SOXL_VIX",
    "SOXL_AUTOCORR_10",
    "SOXL_HV20",
]


def _wide_qlib_panel(frame: pd.DataFrame, *, symbols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", "$close"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"canary sprint qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()

    pieces: list[pd.DataFrame] = []
    for symbol in [item.upper() for item in symbols]:
        symbol_frame = normalized.loc[normalized["instrument"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"canary sprint qlib frame missing instrument: {symbol}")
        columns = {"$close": f"{symbol.lower()}_close"}
        if "$volume" in symbol_frame.columns:
            columns["$volume"] = f"{symbol.lower()}_volume"
        if "$vix" in symbol_frame.columns:
            columns["$vix"] = f"{symbol.lower()}_vix"
        pieces.append(symbol_frame[["datetime", *columns]].rename(columns=columns).groupby("datetime", as_index=False).last())

    panel = pieces[0]
    for piece in pieces[1:]:
        panel = panel.merge(piece, on="datetime", how="inner")
    panel = panel.sort_values("datetime").set_index("datetime")
    if "soxl_vix" in panel.columns:
        panel["vix"] = panel["soxl_vix"].ffill()
    return panel


def build_canary_sprint_frame(
    qlib_frame: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Build short-hold leveraged ETF sprint features from qlib SOXL/SOXX/VIX data."""

    panel = _wide_qlib_panel(qlib_frame, symbols=["SOXL", "SOXX"])
    close = panel["soxl_close"].astype("float64")
    soxx_close = panel["soxx_close"].astype("float64")
    daily_return = close.pct_change()
    vix = panel["vix"] if "vix" in panel.columns else pd.Series(np.nan, index=panel.index)

    features = pd.DataFrame(
        {
            "datetime": panel.index,
            "next_return": daily_return.shift(-1),
            "vix": vix,
            "hv20": daily_return.rolling(20, min_periods=20).std() * np.sqrt(252),
            "autocorr10": daily_return.rolling(10, min_periods=10).corr(daily_return.shift(1)),
        }
    )
    for lookback in [2, 3, 5, 10]:
        features[f"soxl_dip{lookback}"] = close / close.shift(lookback) - 1.0
    for lookback in [10, 20, 40, 60, 120]:
        features[f"soxx_trend{lookback}"] = soxx_close / soxx_close.shift(lookback) - 1.0

    result = features.loc[
        (features["datetime"] >= pd.Timestamp(start)) & (features["datetime"] <= pd.Timestamp(end))
    ].dropna(subset=["next_return", "hv20", "autocorr10", "soxl_dip2", "soxx_trend20"])
    result = result.reset_index(drop=True)
    result.attrs["source_fields"] = CANARY_SPRINT_SOURCE_FIELDS
    result.attrs["source_coverage_start"] = panel.index.min().date().isoformat()
    result.attrs["source_coverage_end"] = panel.index.max().date().isoformat()
    return result


def _hold_signal(signal: np.ndarray, hold_days: int) -> np.ndarray:
    if hold_days <= 0:
        raise ValueError("hold_days must be positive")
    kernel = np.ones(int(hold_days), dtype="int64")
    return np.convolve(signal.astype("int64"), kernel, mode="full")[: len(signal)] > 0


def _strategy_stats(returns: np.ndarray, exposure: np.ndarray, *, cost_per_turnover: float) -> dict[str, float]:
    exposure = np.nan_to_num(np.asarray(exposure, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    strategy_returns = np.nan_to_num(np.asarray(returns, dtype="float64"), nan=0.0) * exposure
    turnover = float(np.abs(np.diff(np.r_[0.0, exposure])).sum())
    if len(strategy_returns):
        strategy_returns[0] -= turnover * float(cost_per_turnover)
    equity = np.cumprod(1.0 + strategy_returns)
    if len(equity) == 0:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "turnover": turnover,
            "avg_exposure": 0.0,
            "avg_abs_exposure": 0.0,
            "max_exposure": 0.0,
            "min_exposure": 0.0,
        }
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "return_pct": float((equity[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "turnover": turnover,
        "avg_exposure": float(exposure.mean()),
        "avg_abs_exposure": float(np.abs(exposure).mean()),
        "max_exposure": float(exposure.max()),
        "min_exposure": float(exposure.min()),
    }


def _decision(return_pct: float, drawdown_pct: float, baseline_return_pct: float, max_drawdown_floor: float) -> tuple[str, str]:
    beats = return_pct > baseline_return_pct
    passes_drawdown = drawdown_pct >= max_drawdown_floor
    if not passes_drawdown:
        return "Drop", "drawdown_gate"
    if not beats:
        return "Iterate", "return_gate"
    return "Promote", "beats_baseline_and_drawdown_gate"


def _guarded_exposure(
    base_exposure: np.ndarray,
    returns: np.ndarray,
    hv20: np.ndarray,
    worst20: np.ndarray,
    *,
    stop_drawdown: float,
    max_daily_loss: float | None,
    vol_target: float | None,
    restart_drawdown: float = -0.03,
    cooldown_days: int = 5,
) -> np.ndarray:
    guarded = np.zeros_like(base_exposure, dtype="float64")
    equity = 1.0
    peak = 1.0
    cooldown = 0
    risk_off = False
    for index, raw_exposure in enumerate(base_exposure):
        drawdown = equity / peak - 1.0
        if drawdown <= stop_drawdown:
            risk_off = True
            cooldown = cooldown_days
        exposure = float(raw_exposure)
        if risk_off:
            exposure = 0.0
            cooldown -= 1
            if drawdown >= restart_drawdown and cooldown <= 0:
                risk_off = False
        if max_daily_loss is not None:
            denom = float(worst20[index]) if np.isfinite(worst20[index]) and worst20[index] > 0 else 1.0
            exposure = min(exposure, float(max_daily_loss) / denom)
        if vol_target is not None:
            denom = float(hv20[index]) if np.isfinite(hv20[index]) and hv20[index] > 0 else 1.0
            exposure = min(exposure, float(vol_target) / denom)
        guarded[index] = max(exposure, 0.0)
        daily_return = float(returns[index]) if np.isfinite(returns[index]) else 0.0
        equity *= 1.0 + guarded[index] * daily_return
        peak = max(peak, equity)
    return guarded


def evaluate_canary_sprint_candidates(
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    trend_lookbacks: list[int] | None = None,
    trend_thresholds: list[float] | None = None,
    vix_maxes: list[float] | None = None,
    autocorr_filters: list[str] | None = None,
    hold_days: list[int] | None = None,
    scales: list[float] | None = None,
    dip_lookbacks: list[int] | None = None,
    dip_thresholds: list[float] | None = None,
    guard_scales: list[float] | None = None,
    guard_stop_drawdowns: list[float] | None = None,
    guard_max_daily_losses: list[float | None] | None = None,
    guard_vol_targets: list[float | None] | None = None,
    cost_per_turnover: float = 0.0002,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("canary sprint frame is empty")
    data = frame.reset_index(drop=True).copy()
    returns = data["next_return"].astype("float64").to_numpy()
    hv20 = data["hv20"].astype("float64").to_numpy()
    worst20 = pd.Series(returns).rolling(20, min_periods=5).min().abs().to_numpy()
    trend_lookbacks = trend_lookbacks or [20, 60, 120]
    trend_thresholds = trend_thresholds or [-0.05, -0.03, 0.0, 0.06, 0.10]
    vix_maxes = vix_maxes or [22.0, 26.0, 30.0, 45.0, 100.0]
    autocorr_filters = autocorr_filters or ["none", "positive"]
    hold_days = hold_days or [1, 2, 3, 5, 10]
    scales = scales or [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0]
    dip_lookbacks = dip_lookbacks or [2, 3, 5]
    dip_thresholds = dip_thresholds or [-0.05, -0.08, -0.12]
    guard_scales = guard_scales or [1.0, 2.0, 4.0, 6.0, 10.0, 20.0]
    guard_stop_drawdowns = guard_stop_drawdowns or [-0.10, -0.15, -0.18]
    guard_max_daily_losses = guard_max_daily_losses or [0.05, 0.10, 0.15, None]
    guard_vol_targets = guard_vol_targets or [0.8, 1.2, None]

    rows: list[dict[str, Any]] = []

    def add_row(branch: str, candidate: str, exposure: np.ndarray, params: dict[str, Any]) -> None:
        stats = _strategy_stats(returns, exposure, cost_per_turnover=cost_per_turnover)
        decision, decision_basis = _decision(
            stats["return_pct"],
            stats["max_drawdown_pct"],
            baseline_return_pct,
            max_drawdown_floor,
        )
        rows.append(
            {
                "branch": branch,
                "candidate": candidate,
                "return_pct": stats["return_pct"],
                "baseline_return_pct": float(baseline_return_pct),
                "return_improvement_pct": float(stats["return_pct"] - baseline_return_pct),
                "max_drawdown_pct": stats["max_drawdown_pct"],
                "max_drawdown_floor": float(max_drawdown_floor),
                "turnover": stats["turnover"],
                "cost_per_turnover": float(cost_per_turnover),
                "avg_exposure": stats["avg_exposure"],
                "avg_abs_exposure": stats["avg_abs_exposure"],
                "max_exposure": stats["max_exposure"],
                "min_exposure": stats["min_exposure"],
                "beats_baseline_return": bool(stats["return_pct"] > baseline_return_pct),
                "passes_drawdown_gate": bool(stats["max_drawdown_pct"] >= max_drawdown_floor),
                "decision": decision,
                "decision_basis": decision_basis,
                "params": json.dumps(params, sort_keys=True),
                "coverage_start": pd.Timestamp(data["datetime"].iloc[0]).date().isoformat(),
                "coverage_end": pd.Timestamp(data["datetime"].iloc[-1]).date().isoformat(),
            }
        )

    canary_bases: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for lookback in trend_lookbacks:
        trend_name = f"soxx_trend{lookback}"
        if trend_name not in data:
            continue
        trend = data[trend_name].astype("float64")
        for trend_threshold in trend_thresholds:
            for vix_max in vix_maxes:
                base_signal = (trend > trend_threshold) & (data["vix"] < vix_max)
                for autocorr_filter in autocorr_filters:
                    signal = base_signal if autocorr_filter == "none" else base_signal & (data["autocorr10"] > 0.0)
                    for hold in hold_days:
                        held = _hold_signal(signal.fillna(False).to_numpy(dtype=bool), hold)
                        params = {
                            "trend_lookback": lookback,
                            "trend_threshold": trend_threshold,
                            "vix_max": vix_max,
                            "autocorr_filter": autocorr_filter,
                            "hold_days": hold,
                        }
                        canary_bases.append(("canary", held.astype("float64"), params))
                        for scale in scales:
                            add_row(
                                "canary_sprint_short_hold",
                                f"trend{lookback}_thr{trend_threshold}_vix{vix_max}_{autocorr_filter}_h{hold}_s{scale}",
                                held.astype("float64") * float(scale),
                                params | {"scale": scale},
                            )

    dip_bases: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for dip_lookback in dip_lookbacks:
        dip_name = f"soxl_dip{dip_lookback}"
        if dip_name not in data:
            continue
        for dip_threshold in dip_thresholds:
            for trend_lookback in trend_lookbacks:
                trend_name = f"soxx_trend{trend_lookback}"
                if trend_name not in data:
                    continue
                for vix_max in vix_maxes:
                    signal = (
                        (data[dip_name] < dip_threshold)
                        & (data[trend_name] > 0.0)
                        & (data["vix"] < vix_max)
                    )
                    for hold in hold_days:
                        held = _hold_signal(signal.fillna(False).to_numpy(dtype=bool), hold)
                        params = {
                            "dip_lookback": dip_lookback,
                            "dip_threshold": dip_threshold,
                            "trend_lookback": trend_lookback,
                            "vix_max": vix_max,
                            "hold_days": hold,
                        }
                        dip_bases.append(("dip", held.astype("float64"), params))
                        for scale in scales:
                            add_row(
                                "dip_sprint_short_hold",
                                f"dip{dip_lookback}_thr{dip_threshold}_trend{trend_lookback}_vix{vix_max}_h{hold}_s{scale}",
                                held.astype("float64") * float(scale),
                                params | {"scale": scale},
                            )

    guard_seed_bases = canary_bases[:3] + dip_bases[:3]
    for seed_type, base, seed_params in guard_seed_bases:
        for scale in guard_scales:
            raw_exposure = base * float(scale)
            for stop_drawdown in guard_stop_drawdowns:
                for max_daily_loss in guard_max_daily_losses:
                    for vol_target in guard_vol_targets:
                        exposure = _guarded_exposure(
                            raw_exposure,
                            returns,
                            hv20,
                            worst20,
                            stop_drawdown=stop_drawdown,
                            max_daily_loss=max_daily_loss,
                            vol_target=vol_target,
                        )
                        add_row(
                            "guarded_sprint_budget",
                            f"{seed_type}_s{scale}_stop{stop_drawdown}_loss{max_daily_loss}_vt{vol_target}",
                            exposure,
                            seed_params
                            | {
                                "seed_type": seed_type,
                                "scale": scale,
                                "stop_drawdown": stop_drawdown,
                                "max_daily_loss": max_daily_loss,
                                "vol_target": vol_target,
                            },
                        )

    return pd.DataFrame(rows)


def summarize_canary_sprint_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"canary sprint branch has no candidate rows: {branch}")
    best_return = rows.sort_values("return_pct", ascending=False).iloc[0]
    drawdown_gate = rows.loc[rows["passes_drawdown_gate"] == True].sort_values(  # noqa: E712
        "return_pct",
        ascending=False,
    )
    has_drawdown_gate = not drawdown_gate.empty
    best_drawdown_gate = drawdown_gate.iloc[0] if has_drawdown_gate else best_return
    passes_both = rows.loc[rows["beats_baseline_return"] & rows["passes_drawdown_gate"]]
    return {
        "grid_rows": int(len(rows)),
        "coverage_start": str(best_return.get("coverage_start")),
        "coverage_end": str(best_return.get("coverage_end")),
        "goal_baseline_return_pct": float(baseline_return_pct),
        "max_drawdown_floor": float(max_drawdown_floor),
        "passes_both_count": int(len(passes_both)),
        "best_return_pct": float(best_return["return_pct"]),
        "best_return_max_drawdown_pct": float(best_return["max_drawdown_pct"]),
        "best_drawdown_gate_available": bool(has_drawdown_gate),
        "best_drawdown_gate_return_pct": float(best_drawdown_gate["return_pct"]),
        "best_drawdown_gate_max_drawdown_pct": float(best_drawdown_gate["max_drawdown_pct"]),
        "best_return_candidate": str(best_return["candidate"]),
        "best_drawdown_gate_candidate": str(best_drawdown_gate["candidate"]),
    }


def compact_canary_sprint_summary(
    metrics: pd.DataFrame,
    *,
    branches: Iterable[str],
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    branch_results = {
        branch: summarize_canary_sprint_branch(
            metrics,
            branch=branch,
            baseline_return_pct=baseline_return_pct,
            max_drawdown_floor=max_drawdown_floor,
        )
        for branch in branches
    }
    passes_both = metrics.loc[metrics["beats_baseline_return"] & metrics["passes_drawdown_gate"]]
    best_candidate = (
        str(passes_both.sort_values("return_pct", ascending=False).iloc[0]["candidate"])
        if not passes_both.empty
        else None
    )
    return {
        "branch_results": branch_results,
        "best_candidate": best_candidate,
        "goal_achieved": bool(best_candidate is not None),
        "passes_both_count": int(len(passes_both)),
    }
