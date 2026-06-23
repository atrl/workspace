from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


LONG_SHORT_CRISIS_ALPHA_SOURCE_FIELDS = [
    "SOXL_TREND_60",
    "SOXL_TREND_120",
    "SOXL_TREND_200",
    "SOXX_TREND_60",
    "SOXL_SOXX_RATIO_MOM_20",
    "SOXL_VIX",
    "SOXL_HV20",
]


def _wide_qlib_panel(frame: pd.DataFrame, *, symbols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", "$close"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"long-short qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()

    pieces: list[pd.DataFrame] = []
    for symbol in [item.upper() for item in symbols]:
        symbol_frame = normalized.loc[normalized["instrument"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"long-short qlib frame missing instrument: {symbol}")
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


def build_long_short_crisis_alpha_frame(
    qlib_frame: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Build time-series momentum and crisis-alpha state from qlib SOXL/SOXX features."""

    panel = _wide_qlib_panel(qlib_frame, symbols=["SOXL", "SOXX"])
    close = panel["soxl_close"].astype("float64")
    soxx_close = panel["soxx_close"].astype("float64")
    daily_return = close.pct_change()
    hv20 = daily_return.rolling(20, min_periods=20).std() * np.sqrt(252)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ma120 = close.rolling(120, min_periods=120).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    vix = panel["vix"] if "vix" in panel.columns else pd.Series(np.nan, index=panel.index)

    trend60 = close / close.shift(60) - 1.0
    trend120 = close / close.shift(120) - 1.0
    trend200 = close / close.shift(200) - 1.0
    soxx_trend60 = soxx_close / soxx_close.shift(60) - 1.0
    ratio_mom20 = (close / soxx_close).pct_change(20)
    vol_z = (hv20 - hv20.rolling(252, min_periods=126).mean()) / (
        hv20.rolling(252, min_periods=126).std() + 1e-12
    )
    vix_z = (vix - vix.rolling(252, min_periods=126).mean()) / (vix.rolling(252, min_periods=126).std() + 1e-12)

    features = pd.DataFrame(
        {
            "datetime": panel.index,
            "next_return": daily_return.shift(-1),
            "hv20": hv20,
            "trend60": trend60,
            "trend120": trend120,
            "trend200": trend200,
            "soxx_trend60": soxx_trend60,
            "ratio_mom20": ratio_mom20,
            "ma_stack_score": (
                (ma20 > ma60).astype("float64")
                + (ma60 > ma120).astype("float64")
                + (close > ma200).astype("float64")
            )
            / 3.0
            - 0.5,
            "vix": vix,
            "vol_z": vol_z,
            "vix_z": vix_z,
        }
    )
    features["dual60_120"] = 0.5 * features["trend60"] + 0.5 * features["trend120"]
    features["soxl_soxx_dual"] = (
        0.6 * features["trend60"] + 0.3 * features["soxx_trend60"] + 0.1 * features["ratio_mom20"]
    )
    result = features.loc[
        (features["datetime"] >= pd.Timestamp(start)) & (features["datetime"] <= pd.Timestamp(end))
    ].dropna(subset=["next_return", "hv20", "trend60", "trend120", "dual60_120", "soxl_soxx_dual"])
    result = result.reset_index(drop=True)
    result.attrs["source_fields"] = LONG_SHORT_CRISIS_ALPHA_SOURCE_FIELDS
    result.attrs["source_coverage_start"] = panel.index.min().date().isoformat()
    result.attrs["source_coverage_end"] = panel.index.max().date().isoformat()
    return result


def _strategy_stats(returns: np.ndarray, exposure: np.ndarray, *, cost_per_turnover: float) -> dict[str, float]:
    exposure = np.nan_to_num(np.asarray(exposure, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    strategy_returns = np.asarray(returns, dtype="float64") * exposure
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


def evaluate_long_short_crisis_alpha_candidates(
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    trend_names: list[str] | None = None,
    positive_thresholds: list[float] | None = None,
    negative_thresholds: list[float] | None = None,
    risk_modes: list[str] | None = None,
    long_scales: list[float] | None = None,
    short_scales: list[float] | None = None,
    target_vols: list[float | None] | None = None,
    crisis_base_thresholds: list[float] | None = None,
    crisis_base_scales: list[float] | None = None,
    crisis_hedge_scales: list[float] | None = None,
    cost_per_turnover: float = 0.0002,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("long-short crisis-alpha frame is empty")
    data = frame.reset_index(drop=True).copy()
    returns = data["next_return"].astype("float64").to_numpy()
    hv20 = data["hv20"].astype("float64").to_numpy()
    trend_names = trend_names or ["trend120", "dual60_120", "ma_stack_score", "soxl_soxx_dual"]
    positive_thresholds = positive_thresholds or [-0.05, 0.0, 0.05]
    negative_thresholds = negative_thresholds or [-0.12, -0.04, 0.0]
    risk_modes = risk_modes or ["none", "vix25", "vix30_or_volz1"]
    long_scales = long_scales or [1.0, 2.0]
    short_scales = short_scales or [0.0, 1.0, 4.0]
    target_vols = target_vols or [None, 0.40]
    crisis_base_thresholds = crisis_base_thresholds or [-0.05, 0.0, 0.05]
    crisis_base_scales = crisis_base_scales or [1.0, 2.0]
    crisis_hedge_scales = crisis_hedge_scales or [0.5, 2.0, 6.0]

    risk_lookup = {
        "none": np.ones(len(data), dtype=bool),
        "vix25": (data["vix"] > 25.0).to_numpy(dtype=bool),
        "vix30_or_volz1": ((data["vix"] > 30.0) | (data["vol_z"] > 1.0)).to_numpy(dtype=bool),
        "vixz0": (data["vix_z"] > 0.0).to_numpy(dtype=bool),
    }
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

    for trend_name in trend_names:
        score = data[trend_name].astype("float64").to_numpy()
        for pos_threshold in positive_thresholds:
            long_signal = (score > pos_threshold).astype("float64")
            for neg_threshold in negative_thresholds:
                short_mask = score < neg_threshold
                for risk_mode in risk_modes:
                    if risk_mode not in risk_lookup:
                        raise ValueError(f"unknown long-short risk mode: {risk_mode}")
                    short_signal = (short_mask & risk_lookup[risk_mode]).astype("float64")
                    for long_scale in long_scales:
                        for short_scale in short_scales:
                            raw = long_signal * float(long_scale) - short_signal * float(short_scale)
                            for target_vol in target_vols:
                                exposure = raw
                                if target_vol is not None:
                                    exposure = raw * np.minimum(1.0, float(target_vol) / (hv20 + 1e-12))
                                add_row(
                                    "long_short_time_series_momentum",
                                    (
                                        f"{trend_name}_p{pos_threshold}_n{neg_threshold}_{risk_mode}"
                                        f"_l{long_scale}_s{short_scale}_tv{target_vol}"
                                    ),
                                    exposure,
                                    {
                                        "trend": trend_name,
                                        "positive_threshold": pos_threshold,
                                        "negative_threshold": neg_threshold,
                                        "risk_mode": risk_mode,
                                        "long_scale": long_scale,
                                        "short_scale": short_scale,
                                        "target_vol": target_vol,
                                    },
                                )

    crisis_risk = ((data["vix"] > 25.0) | (data["vol_z"] > 1.0) | (data["vix_z"] > 0.0)).to_numpy(dtype=bool)
    for trend_name in trend_names:
        score = data[trend_name].astype("float64").to_numpy()
        for base_threshold in crisis_base_thresholds:
            base_signal = (score > base_threshold).astype("float64")
            crash_signal = ((score < 0.0) & crisis_risk).astype("float64")
            for base_scale in crisis_base_scales:
                for hedge_scale in crisis_hedge_scales:
                    for target_vol in target_vols:
                        raw = base_signal * float(base_scale) - crash_signal * float(hedge_scale)
                        exposure = raw
                        if target_vol is not None:
                            exposure = raw * np.minimum(1.0, float(target_vol) / (hv20 + 1e-12))
                        add_row(
                            "crisis_alpha_overlay",
                            f"{trend_name}_base{base_threshold}_b{base_scale}_h{hedge_scale}_tv{target_vol}",
                            exposure,
                            {
                                "trend": trend_name,
                                "base_threshold": base_threshold,
                                "base_scale": base_scale,
                                "hedge_scale": hedge_scale,
                                "target_vol": target_vol,
                            },
                        )

    return pd.DataFrame(rows)


def summarize_long_short_crisis_alpha_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"long-short crisis-alpha branch has no candidate rows: {branch}")
    best_return = rows.sort_values("return_pct", ascending=False).iloc[0]
    drawdown_gate = rows.loc[rows["passes_drawdown_gate"] == True].sort_values(  # noqa: E712
        "return_pct", ascending=False
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


def compact_long_short_crisis_alpha_summary(
    metrics: pd.DataFrame,
    *,
    branches: Iterable[str],
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    branch_results = {
        branch: summarize_long_short_crisis_alpha_branch(
            metrics,
            branch=branch,
            baseline_return_pct=baseline_return_pct,
            max_drawdown_floor=max_drawdown_floor,
        )
        for branch in branches
    }
    passes_both = metrics.loc[metrics["beats_baseline_return"] & metrics["passes_drawdown_gate"]]
    best_candidate = None
    if not passes_both.empty:
        best_candidate = str(passes_both.sort_values("return_pct", ascending=False).iloc[0]["candidate"])
    return {
        "branch_results": branch_results,
        "passes_both_count": int(len(passes_both)),
        "goal_achieved": best_candidate is not None,
        "best_candidate": best_candidate,
    }
