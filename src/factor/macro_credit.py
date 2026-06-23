from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.agentic_alpha import AGENTIC_ALPHA_STAGE_ORDER


MACRO_CREDIT_SOURCE_FIELDS = [
    "SOXL_CLOSE",
    "SOXX_TREND",
    "YIELD_CURVE",
    "CREDIT_SPREAD",
    "BAA_YIELD",
    "DXY",
    "OIL",
]
MACRO_CREDIT_BRANCHES = [
    "macro_credit_rate_gate",
    "macro_composite_throttle",
    "trend_drawdown_stop",
]
MACRO_CREDIT_REQUIRED_COLUMNS = {
    "$soxl_close",
    "$soxx_close",
    "$yc10y2y",
    "$yc10y3m",
    "$credit_baa",
    "$baa_yield",
    "$dxy",
    "$oil",
}
TREND_WINDOWS = [20, 42, 50, 63, 84, 100, 126, 168, 200, 252]
RISK_WINDOWS = [20, 42, 63, 126, 252, 504]
STOP_WINDOWS = [42, 63, 84, 126, 168, 200, 252]


def _normalize_qlib_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", *MACRO_CREDIT_REQUIRED_COLUMNS}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"macro-credit qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()
    keep = ["datetime", *sorted(MACRO_CREDIT_REQUIRED_COLUMNS), "$hy_spread"]
    keep = [column for column in keep if column in normalized.columns]
    normalized = normalized[keep].sort_values("datetime").groupby("datetime", as_index=False).last()
    rename = {column: column[1:] for column in keep if column.startswith("$")}
    normalized = normalized.rename(columns=rename)
    for column in rename.values():
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.sort_values("datetime").reset_index(drop=True)


def _zscore(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=min(window, min_periods))
    return (series - rolling.mean()) / (rolling.std() + 1e-12)


def build_macro_credit_frame(
    qlib_frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    min_history: int = 20,
) -> pd.DataFrame:
    """Build SOXL/SOXX trend features plus credit, curve, dollar, and oil state."""

    frame = _normalize_qlib_frame(qlib_frame)
    frame = frame.loc[(frame["datetime"] >= pd.Timestamp(start)) & (frame["datetime"] <= pd.Timestamp(end))].copy()
    frame = frame.sort_values("datetime").reset_index(drop=True)
    macro_columns = ["yc10y2y", "yc10y3m", "credit_baa", "baa_yield", "dxy", "oil"]
    if "hy_spread" in frame:
        macro_columns.append("hy_spread")
    for column in macro_columns:
        frame[column] = frame[column].ffill()

    derived: dict[str, pd.Series] = {}
    soxl_ret1 = frame["soxl_close"].pct_change()
    soxx_ret1 = frame["soxx_close"].pct_change()
    derived["soxl_ret1"] = soxl_ret1
    derived["soxx_ret1"] = soxx_ret1
    derived["next_return"] = frame["soxl_close"].pct_change().shift(-1)
    derived["soxl_hv20"] = soxl_ret1.rolling(20, min_periods=min(20, min_history)).std() * np.sqrt(252)

    for window in TREND_WINDOWS:
        min_periods = min(window, min_history)
        derived[f"soxl_trend{window}"] = frame["soxl_close"] / frame["soxl_close"].shift(window) - 1.0
        derived[f"soxx_trend{window}"] = frame["soxx_close"] / frame["soxx_close"].shift(window) - 1.0
        derived[f"soxl_trend_fast{window}"] = soxl_ret1.rolling(window, min_periods=min_periods).mean()

    for window in RISK_WINDOWS:
        min_periods = min(window, min_history)
        risk_features: dict[str, pd.Series] = {}
        for column in macro_columns:
            zscore = _zscore(frame[column], window=window, min_periods=min_periods)
            risk_features[column] = zscore
            derived[f"{column}_z{window}"] = zscore
            derived[f"{column}_chg{window}"] = frame[column] - frame[column].shift(window)
        score = (frame["yc10y2y"] < 0.0).astype("float64") + (frame["yc10y3m"] < 0.0).astype("float64")
        score = score + (risk_features["credit_baa"] > 0.5).fillna(False).astype("float64")
        score = score + 0.5 * (risk_features["baa_yield"] > 0.5).fillna(False).astype("float64")
        score = score + 0.5 * (risk_features["dxy"] > 0.75).fillna(False).astype("float64")
        score = score + 0.25 * (risk_features["oil"] > 1.0).fillna(False).astype("float64")
        if "hy_spread" in frame:
            score = score + (risk_features["hy_spread"] > 0.5).fillna(False).astype("float64")
        derived[f"macro_risk_score{window}"] = score

    for window in STOP_WINDOWS:
        derived[f"rolling_drawdown{window}"] = frame["soxl_close"] / frame["soxl_close"].rolling(
            window,
            min_periods=min(window, min_history),
        ).max() - 1.0

    frame = pd.concat([frame, pd.DataFrame(derived, index=frame.index)], axis=1)
    required = ["next_return", "soxx_trend20", "soxl_hv20", "macro_risk_score20", "rolling_drawdown42"]
    result = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    result.attrs["source_fields"] = MACRO_CREDIT_SOURCE_FIELDS
    result.attrs["source_coverage_start"] = frame["datetime"].min().date().isoformat() if len(frame) else None
    result.attrs["source_coverage_end"] = frame["datetime"].max().date().isoformat() if len(frame) else None
    return result


def _strategy_stats(
    returns: np.ndarray,
    exposure: np.ndarray,
    *,
    cost_per_turnover: float,
) -> dict[str, float]:
    exposure = np.nan_to_num(np.asarray(exposure, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    strategy_returns = np.nan_to_num(np.asarray(returns, dtype="float64"), nan=0.0) * exposure
    turnover = np.abs(np.diff(np.r_[0.0, exposure]))
    strategy_returns = strategy_returns - turnover * float(cost_per_turnover)
    if len(strategy_returns) == 0:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "turnover": 0.0,
            "avg_abs_exposure": 0.0,
            "max_exposure": 0.0,
            "trade_count": 0,
        }
    if np.any(strategy_returns <= -0.99):
        return {
            "return_pct": -100.0,
            "max_drawdown_pct": -100.0,
            "turnover": float(turnover.sum()),
            "avg_abs_exposure": float(np.abs(exposure).mean()),
            "max_exposure": float(exposure.max()),
            "trade_count": int(np.count_nonzero(exposure)),
        }
    equity = np.cumprod(1.0 + strategy_returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "return_pct": float((equity[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "turnover": float(turnover.sum()),
        "avg_abs_exposure": float(np.abs(exposure).mean()),
        "max_exposure": float(exposure.max()),
        "trade_count": int(np.count_nonzero(exposure)),
    }


def _decision(return_pct: float, drawdown_pct: float, baseline_return_pct: float, max_drawdown_floor: float) -> tuple[str, str]:
    beats_baseline = float(return_pct) > float(baseline_return_pct)
    passes_drawdown = float(drawdown_pct) >= float(max_drawdown_floor)
    if not passes_drawdown:
        return "Drop", "drawdown_gate"
    if not beats_baseline:
        return "Iterate", "return_gate"
    return "Promote", "beats_baseline_and_drawdown_gate"


def evaluate_macro_credit_candidates(
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    risk_windows: list[int] | None = None,
    trend_windows: list[int] | None = None,
    risk_thresholds: list[float] | None = None,
    scales: list[float] | None = None,
    composite_bases: list[float] | None = None,
    composite_trend_weights: list[float] | None = None,
    composite_risk_weights: list[float] | None = None,
    vol_targets: list[float | None] | None = None,
    stop_windows: list[int] | None = None,
    stop_levels: list[float] | None = None,
    reentry_thresholds: list[float] | None = None,
    cost_per_turnover: float = 0.0002,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("macro-credit frame is empty")
    data = frame.sort_values("datetime").reset_index(drop=True).copy()
    risk_windows = risk_windows or [20, 42, 63, 126, 252]
    trend_windows = trend_windows or [42, 50, 63, 84, 100, 126, 168, 200]
    risk_thresholds = risk_thresholds or [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    scales = scales or [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    composite_bases = composite_bases or [0.25, 0.5, 0.75, 1.0]
    composite_trend_weights = composite_trend_weights or [0.5, 0.75, 1.0, 1.25]
    composite_risk_weights = composite_risk_weights or [0.5, 0.75, 1.0, 1.25, 1.5]
    vol_targets = vol_targets or [0.2, 0.3, 0.4, 0.6, 0.8, None]
    stop_windows = stop_windows or [42, 63, 84, 126, 168, 200, 252]
    stop_levels = stop_levels or [-0.05, -0.1, -0.15, -0.2, -0.25, -0.3]
    reentry_thresholds = reentry_thresholds or [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    returns = data["next_return"].astype("float64").to_numpy()
    rows: list[dict[str, Any]] = []

    def add_row(branch: str, candidate: str, exposure: np.ndarray, params: dict[str, Any]) -> None:
        exposure = np.nan_to_num(np.asarray(exposure, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
        exposure = np.clip(exposure, -5.0, 5.0)
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

    for risk_window in risk_windows:
        risk_col = f"macro_risk_score{risk_window}"
        if risk_col not in data:
            continue
        risk = data[risk_col].astype("float64")
        for trend_window in trend_windows:
            soxx_col = f"soxx_trend{trend_window}"
            soxl_col = f"soxl_trend{trend_window}"
            if soxx_col not in data or soxl_col not in data:
                continue
            trend = (data[soxx_col] > 0.0) & (data[soxl_col] > -0.2)
            for risk_threshold in risk_thresholds:
                signal = trend & (risk <= float(risk_threshold))
                for scale in scales:
                    add_row(
                        "macro_credit_rate_gate",
                        f"risklb{risk_window}_trend{trend_window}_risk{float(risk_threshold):g}_scale{float(scale):g}",
                        signal.astype("float64").to_numpy() * float(scale),
                        {
                            "risk_window": risk_window,
                            "trend_window": trend_window,
                            "risk_threshold": float(risk_threshold),
                            "scale": float(scale),
                        },
                    )

    for risk_window in risk_windows:
        risk_col = f"macro_risk_score{risk_window}"
        if risk_col not in data:
            continue
        risk = data[risk_col].astype("float64").to_numpy()
        for trend_window in trend_windows:
            soxx_col = f"soxx_trend{trend_window}"
            soxl_col = f"soxl_trend{trend_window}"
            if soxx_col not in data or soxl_col not in data:
                continue
            trend_score = (data[soxx_col] > 0.0).astype("float64") + (data[soxl_col] > 0.0).astype("float64")
            trend_array = trend_score.to_numpy()
            for base in composite_bases:
                for trend_weight in composite_trend_weights:
                    for risk_weight in composite_risk_weights:
                        raw = np.clip(float(base) + float(trend_weight) * trend_array - float(risk_weight) * risk, 0.0, 3.0)
                        for vol_target in vol_targets:
                            exposure = raw
                            if vol_target is not None:
                                exposure = np.minimum(raw, float(vol_target) / (data["soxl_hv20"].to_numpy() + 1e-12))
                            add_row(
                                "macro_composite_throttle",
                                (
                                    f"risklb{risk_window}_trend{trend_window}_b{float(base):g}_"
                                    f"tw{float(trend_weight):g}_rw{float(risk_weight):g}_vol{vol_target}"
                                ),
                                exposure,
                                {
                                    "risk_window": risk_window,
                                    "trend_window": trend_window,
                                    "base": float(base),
                                    "trend_weight": float(trend_weight),
                                    "risk_weight": float(risk_weight),
                                    "vol_target": vol_target,
                                },
                            )

    for window in stop_windows:
        trend_col = f"soxx_trend{window}"
        drawdown_col = f"rolling_drawdown{window}"
        if trend_col not in data or drawdown_col not in data:
            continue
        for stop_level in stop_levels:
            for reentry_threshold in reentry_thresholds:
                signal = (data[trend_col] > float(reentry_threshold)) & (data[drawdown_col] > float(stop_level))
                for scale in scales:
                    add_row(
                        "trend_drawdown_stop",
                        f"lb{window}_stop{float(stop_level):g}_re{float(reentry_threshold):g}_scale{float(scale):g}",
                        signal.astype("float64").to_numpy() * float(scale),
                        {
                            "window": window,
                            "stop_level": float(stop_level),
                            "reentry_threshold": float(reentry_threshold),
                            "scale": float(scale),
                        },
                    )

    return pd.DataFrame(rows)


def summarize_macro_credit_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"macro-credit branch has no candidate rows: {branch}")
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


def compact_macro_credit_summary(
    metrics: pd.DataFrame,
    *,
    branches: list[str],
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("macro-credit metrics are empty")
    branch_results = {
        branch: summarize_macro_credit_branch(
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
        "workflow": "agentic_alpha_macro_credit_search",
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "goal_achieved": bool(best_candidate is not None),
        "best_candidate": best_candidate,
        "passes_both_count": int(len(passers)),
        "branch_results": branch_results,
        "selected_features": MACRO_CREDIT_SOURCE_FIELDS,
    }
