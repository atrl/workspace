from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.workflow.agentic_alpha import AGENTIC_ALPHA_STAGE_ORDER


AGENTIC_ENSEMBLE_SOURCE_FIELDS = [
    "TSMOM_STOP_SOXX_RET10",
    "TSMOM_STOP_SOXX_RET252",
    "CANARY_TREND60_AUTOCORR",
    "SIGNED_OVERNIGHT_MOM20",
    "MACRO_CREDIT_RATE_GATE",
]


def _wide_qlib_ohlc_panel(frame: pd.DataFrame, *, symbols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", "$open", "$high", "$low", "$close"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"agentic ensemble qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()

    pieces: list[pd.DataFrame] = []
    for symbol in [item.upper() for item in symbols]:
        symbol_frame = normalized.loc[normalized["instrument"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"agentic ensemble qlib frame missing instrument: {symbol}")
        columns = {
            "$open": f"{symbol.lower()}_open",
            "$high": f"{symbol.lower()}_high",
            "$low": f"{symbol.lower()}_low",
            "$close": f"{symbol.lower()}_close",
        }
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


def _ohlc_stop_return(panel: pd.DataFrame, *, stop_loss: float) -> pd.Series:
    if stop_loss <= 0:
        raise ValueError("stop_loss must be positive")
    close = panel["soxl_close"].astype("float64")
    next_open = panel["soxl_open"].shift(-1).astype("float64")
    next_low = panel["soxl_low"].shift(-1).astype("float64")
    next_close = panel["soxl_close"].shift(-1).astype("float64")
    open_return = next_open / close - 1.0
    low_return = next_low / close - 1.0
    close_return = next_close / close - 1.0
    stopped = np.where(
        open_return <= -float(stop_loss),
        open_return,
        np.where(low_return <= -float(stop_loss), -float(stop_loss), close_return),
    )
    return pd.Series(stopped, index=panel.index)


def _component_frame(
    *,
    datetimes: pd.Series | pd.DatetimeIndex,
    component: str,
    branch: str,
    component_return: np.ndarray | pd.Series,
    exposure: np.ndarray | pd.Series,
    params: dict[str, Any],
    start: str,
    end: str,
) -> pd.DataFrame:
    normalized_datetimes = pd.Series(pd.to_datetime(datetimes)).dt.normalize().to_numpy()
    result = pd.DataFrame(
        {
            "datetime": normalized_datetimes,
            "component": component,
            "branch": branch,
            "component_return": np.nan_to_num(np.asarray(component_return, dtype="float64"), nan=0.0),
            "exposure": np.nan_to_num(np.asarray(exposure, dtype="float64"), nan=0.0),
            "params": json.dumps(params, sort_keys=True),
        }
    )
    result = result.loc[
        (result["datetime"] >= pd.Timestamp(start)) & (result["datetime"] <= pd.Timestamp(end))
    ].reset_index(drop=True)
    result.attrs["source_fields"] = AGENTIC_ENSEMBLE_SOURCE_FIELDS
    return result


def build_tsmom_stop_component(
    qlib_frame: pd.DataFrame,
    *,
    component: str,
    start: str,
    end: str,
    score_name: str,
    threshold: float,
    regime_filter: str,
    scale: float,
    stop_loss: float,
    lowhv_quantile: float = 0.70,
) -> pd.DataFrame:
    """Build a close-to-next-close momentum leg with next-day OHLC stop logic."""

    panel = _wide_qlib_ohlc_panel(qlib_frame, symbols=["SOXL", "SOXX"])
    soxl_close = panel["soxl_close"].astype("float64")
    soxx_close = panel["soxx_close"].astype("float64")
    score_lookup = {
        "soxl_ret10": soxl_close / soxl_close.shift(10) - 1.0,
        "soxx_ret10": soxx_close / soxx_close.shift(10) - 1.0,
        "soxx_ret40": soxx_close / soxx_close.shift(40) - 1.0,
        "soxx_ret252": soxx_close / soxx_close.shift(252) - 1.0,
    }
    if score_name not in score_lookup:
        raise ValueError(f"unknown tsmom score_name: {score_name}")
    signal = score_lookup[score_name] > float(threshold)
    soxl_ret = soxl_close.pct_change()
    hv20 = soxl_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    if regime_filter == "lowhv":
        lowhv = hv20 < hv20.rolling(252, min_periods=126).quantile(float(lowhv_quantile))
        signal = signal & lowhv
    elif regime_filter == "vix30":
        signal = signal & (panel["vix"] < 30.0)
    elif regime_filter == "soxx_ret126":
        signal = signal & (soxx_close / soxx_close.shift(126) - 1.0 > 0.0)
    elif regime_filter != "none":
        raise ValueError(f"unknown tsmom regime_filter: {regime_filter}")

    exposure = signal.fillna(False).astype("float64") * float(scale)
    stopped_return = _ohlc_stop_return(panel, stop_loss=stop_loss)
    return _component_frame(
        datetimes=panel.index,
        component=component,
        branch="tsmom_stop",
        component_return=stopped_return * exposure,
        exposure=exposure,
        params={
            "score_name": score_name,
            "threshold": float(threshold),
            "regime_filter": regime_filter,
            "scale": float(scale),
            "stop_loss": float(stop_loss),
            "lowhv_quantile": float(lowhv_quantile),
        },
        start=start,
        end=end,
    )


def _hold_signal(signal: np.ndarray, hold_days: int) -> np.ndarray:
    if hold_days <= 0:
        raise ValueError("hold_days must be positive")
    return np.convolve(signal.astype("int64"), np.ones(int(hold_days), dtype="int64"), mode="full")[: len(signal)] > 0


def build_canary_trend_component(
    canary_frame: pd.DataFrame,
    *,
    component: str,
    start: str,
    end: str,
    scale: float,
    trend_threshold: float = 0.06,
    vix_max: float = 26.0,
    hold_days: int = 3,
    require_positive_autocorr: bool = True,
) -> pd.DataFrame:
    data = canary_frame.sort_values("datetime").reset_index(drop=True).copy()
    required = {"datetime", "next_return", "soxx_trend60", "vix", "autocorr10"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"canary trend frame missing columns: {missing}")
    signal = (data["soxx_trend60"] > float(trend_threshold)) & (data["vix"] < float(vix_max))
    if require_positive_autocorr:
        signal = signal & (data["autocorr10"] > 0.0)
    held = _hold_signal(signal.fillna(False).to_numpy(dtype=bool), hold_days)
    exposure = held.astype("float64") * float(scale)
    return _component_frame(
        datetimes=data["datetime"],
        component=component,
        branch="canary_sprint_short_hold",
        component_return=data["next_return"].astype("float64").to_numpy() * exposure,
        exposure=exposure,
        params={
            "trend_threshold": float(trend_threshold),
            "vix_max": float(vix_max),
            "hold_days": int(hold_days),
            "require_positive_autocorr": bool(require_positive_autocorr),
            "scale": float(scale),
        },
        start=start,
        end=end,
    )


def build_signed_overnight_component(
    session_frame: pd.DataFrame,
    *,
    component: str,
    start: str,
    end: str,
    neutral_band: float,
    scale: float,
    momentum_window: int = 20,
    trend_filter: str = "soxl60",
    vix_max: float = 45.0,
) -> pd.DataFrame:
    data = session_frame.sort_values("datetime").reset_index(drop=True).copy()
    score_col = f"overnight_mom{int(momentum_window)}"
    required = {"datetime", "next_overnight_return", score_col, "vix"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"signed overnight frame missing columns: {missing}")
    raw = np.select(
        [data[score_col].astype("float64") > float(neutral_band), data[score_col].astype("float64") < -float(neutral_band)],
        [1.0, -1.0],
        default=0.0,
    )
    if trend_filter == "soxl60":
        active = data["soxl_trend60"] > 0.0
    elif trend_filter == "soxx60":
        active = data["soxx_trend60"] > 0.0
    elif trend_filter == "none":
        active = pd.Series(True, index=data.index)
    else:
        raise ValueError(f"unknown signed overnight trend_filter: {trend_filter}")
    active = active & (data["vix"] < float(vix_max))
    exposure = raw * active.fillna(False).to_numpy(dtype=bool) * float(scale)
    return _component_frame(
        datetimes=data["datetime"],
        component=component,
        branch="signed_overnight_momentum",
        component_return=data["next_overnight_return"].astype("float64").to_numpy() * exposure,
        exposure=exposure,
        params={
            "momentum_window": int(momentum_window),
            "neutral_band": float(neutral_band),
            "trend_filter": trend_filter,
            "vix_max": float(vix_max),
            "scale": float(scale),
        },
        start=start,
        end=end,
    )


def build_macro_rate_gate_component(
    macro_frame: pd.DataFrame,
    *,
    component: str,
    start: str,
    end: str,
    risk_window: int = 20,
    trend_window: int = 50,
    risk_threshold: float = 0.0,
    scale: float = 0.5,
) -> pd.DataFrame:
    data = macro_frame.sort_values("datetime").reset_index(drop=True).copy()
    risk_col = f"macro_risk_score{int(risk_window)}"
    soxx_col = f"soxx_trend{int(trend_window)}"
    soxl_col = f"soxl_trend{int(trend_window)}"
    required = {"datetime", "next_return", risk_col, soxx_col, soxl_col}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"macro rate gate frame missing columns: {missing}")
    signal = (
        (data[risk_col].astype("float64") <= float(risk_threshold))
        & (data[soxx_col].astype("float64") > 0.0)
        & (data[soxl_col].astype("float64") > -0.2)
    )
    exposure = signal.fillna(False).astype("float64").to_numpy() * float(scale)
    return _component_frame(
        datetimes=data["datetime"],
        component=component,
        branch="macro_credit_rate_gate",
        component_return=data["next_return"].astype("float64").to_numpy() * exposure,
        exposure=exposure,
        params={
            "risk_window": int(risk_window),
            "trend_window": int(trend_window),
            "risk_threshold": float(risk_threshold),
            "scale": float(scale),
        },
        start=start,
        end=end,
    )


def _decision(return_pct: float, drawdown_pct: float, baseline_return_pct: float, max_drawdown_floor: float) -> tuple[str, str]:
    beats_baseline = float(return_pct) > float(baseline_return_pct)
    passes_drawdown = float(drawdown_pct) >= float(max_drawdown_floor)
    if not passes_drawdown:
        return "Drop", "drawdown_gate"
    if not beats_baseline:
        return "Iterate", "return_gate"
    return "Promote", "beats_baseline_and_drawdown_gate"


def _return_stats(strategy_returns: np.ndarray, exposure: np.ndarray) -> dict[str, float]:
    strategy_returns = np.nan_to_num(np.asarray(strategy_returns, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    exposure = np.nan_to_num(np.asarray(exposure, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    if len(strategy_returns) == 0:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_abs_exposure": 0.0,
            "max_exposure": 0.0,
            "min_exposure": 0.0,
        }
    if np.any(strategy_returns <= -0.99):
        return {
            "return_pct": -100.0,
            "max_drawdown_pct": -100.0,
            "avg_abs_exposure": float(np.abs(exposure).mean()),
            "max_exposure": float(exposure.max()),
            "min_exposure": float(exposure.min()),
        }
    equity = np.cumprod(1.0 + strategy_returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "return_pct": float((equity[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "avg_abs_exposure": float(np.abs(exposure).mean()),
        "max_exposure": float(exposure.max()),
        "min_exposure": float(exposure.min()),
    }


def evaluate_agentic_ensemble_candidates(
    component_frame: pd.DataFrame,
    *,
    candidates: list[dict[str, Any]],
    baseline_return_pct: float,
    max_drawdown_floor: float,
    cost_per_turnover: float = 0.0003,
) -> pd.DataFrame:
    if component_frame.empty:
        raise ValueError("agentic ensemble component_frame is empty")
    required = {"datetime", "component", "branch", "component_return", "exposure"}
    missing = sorted(required - set(component_frame.columns))
    if missing:
        raise ValueError(f"agentic ensemble component_frame missing columns: {missing}")
    if not candidates:
        raise ValueError("at least one ensemble candidate is required")

    data = component_frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"]).dt.normalize()
    data["component"] = data["component"].astype(str)
    data["branch"] = data["branch"].astype(str)
    branch_lookup = data.groupby("component")["branch"].first().to_dict()
    returns = data.pivot_table(index="datetime", columns="component", values="component_return", aggfunc="sum").sort_index()
    exposures = data.pivot_table(index="datetime", columns="component", values="exposure", aggfunc="sum").sort_index()
    returns = returns.fillna(0.0)
    exposures = exposures.reindex(returns.index).fillna(0.0)
    known_components = set(returns.columns)

    rows: list[dict[str, Any]] = []
    for spec in candidates:
        name = str(spec["candidate"])
        weights = {str(key): float(value) for key, value in dict(spec.get("weights", {})).items()}
        nonzero_weights = {key: value for key, value in weights.items() if abs(value) > 1e-12}
        unknown = sorted(set(nonzero_weights) - known_components)
        if unknown:
            raise ValueError(f"ensemble candidate {name!r} references unknown components: {unknown}")

        combined_return = pd.Series(0.0, index=returns.index)
        combined_exposure = pd.Series(0.0, index=returns.index)
        for component, weight in nonzero_weights.items():
            combined_return = combined_return + returns[component] * weight
            combined_exposure = combined_exposure + exposures[component] * weight
        turnover = np.abs(np.diff(np.r_[0.0, combined_exposure.to_numpy(dtype="float64")]))
        strategy_returns = combined_return.to_numpy(dtype="float64") - turnover * float(cost_per_turnover)
        stats = _return_stats(strategy_returns, combined_exposure.to_numpy(dtype="float64"))
        decision, decision_basis = _decision(
            stats["return_pct"],
            stats["max_drawdown_pct"],
            baseline_return_pct,
            max_drawdown_floor,
        )
        component_branches = sorted({branch_lookup[component] for component in nonzero_weights})
        rows.append(
            {
                "branch": "multi_branch_ensemble",
                "candidate": name,
                "return_pct": float(stats["return_pct"]),
                "baseline_return_pct": float(baseline_return_pct),
                "return_improvement_pct": float(stats["return_pct"] - float(baseline_return_pct)),
                "max_drawdown_pct": float(stats["max_drawdown_pct"]),
                "max_drawdown_floor": float(max_drawdown_floor),
                "turnover": float(turnover.sum()),
                "cost_per_turnover": float(cost_per_turnover),
                "avg_abs_exposure": float(stats["avg_abs_exposure"]),
                "max_exposure": float(stats["max_exposure"]),
                "min_exposure": float(stats["min_exposure"]),
                "component_count": int(len(nonzero_weights)),
                "component_branches": json.dumps(component_branches, sort_keys=True),
                "weights": json.dumps(nonzero_weights, sort_keys=True),
                "params": json.dumps(spec.get("params", {}), sort_keys=True),
                "beats_baseline_return": bool(stats["return_pct"] > float(baseline_return_pct)),
                "passes_drawdown_gate": bool(stats["max_drawdown_pct"] >= float(max_drawdown_floor)),
                "decision": decision,
                "decision_basis": decision_basis,
                "coverage_start": returns.index.min().date().isoformat(),
                "coverage_end": returns.index.max().date().isoformat(),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics["beats_baseline_return"] = metrics["beats_baseline_return"].astype(object)
    metrics["passes_drawdown_gate"] = metrics["passes_drawdown_gate"].astype(object)
    return metrics.sort_values(["decision", "return_pct"], ascending=[False, False]).reset_index(drop=True)


def summarize_agentic_ensemble_components(component_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, group in component_frame.groupby("component", sort=True):
        group = group.sort_values("datetime")
        stats = _return_stats(group["component_return"].to_numpy(dtype="float64"), group["exposure"].to_numpy(dtype="float64"))
        rows.append(
            {
                "component": str(component),
                "branch": str(group["branch"].iloc[0]),
                "return_pct": float(stats["return_pct"]),
                "max_drawdown_pct": float(stats["max_drawdown_pct"]),
                "avg_abs_exposure": float(stats["avg_abs_exposure"]),
                "max_exposure": float(stats["max_exposure"]),
                "min_exposure": float(stats["min_exposure"]),
                "coverage_start": pd.Timestamp(group["datetime"].min()).date().isoformat(),
                "coverage_end": pd.Timestamp(group["datetime"].max()).date().isoformat(),
                "params": str(group["params"].iloc[0]) if "params" in group else "{}",
            }
        )
    return pd.DataFrame(rows).sort_values("component").reset_index(drop=True)


def compact_agentic_ensemble_summary(
    metrics: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("agentic ensemble metrics are empty")
    passers = metrics.loc[metrics["beats_baseline_return"] & metrics["passes_drawdown_gate"]]
    best_candidate = None
    if not passers.empty:
        best_candidate = str(passers.sort_values("return_pct", ascending=False).iloc[0]["candidate"])
    best_return = metrics.sort_values("return_pct", ascending=False).iloc[0]
    drawdown_gate = metrics.loc[metrics["passes_drawdown_gate"] == True].sort_values(  # noqa: E712
        "return_pct",
        ascending=False,
    )
    best_drawdown_gate = drawdown_gate.iloc[0] if not drawdown_gate.empty else None
    return {
        "workflow": "agentic_alpha_ensemble_search",
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "goal_achieved": bool(best_candidate is not None),
        "best_candidate": best_candidate,
        "passes_both_count": int(len(passers)),
        "selected_features": AGENTIC_ENSEMBLE_SOURCE_FIELDS,
        "branch_results": {
            "multi_branch_ensemble": {
                "grid_rows": int(len(metrics)),
                "coverage_start": str(best_return["coverage_start"]),
                "coverage_end": str(best_return["coverage_end"]),
                "goal_baseline_return_pct": float(baseline_return_pct),
                "max_drawdown_floor": float(max_drawdown_floor),
                "passes_both_count": int(len(passers)),
                "best_return_pct": float(best_return["return_pct"]),
                "best_return_max_drawdown_pct": float(best_return["max_drawdown_pct"]),
                "best_return_candidate": str(best_return["candidate"]),
                "best_drawdown_gate_available": bool(best_drawdown_gate is not None),
                "best_drawdown_gate_return_pct": (
                    float(best_drawdown_gate["return_pct"]) if best_drawdown_gate is not None else None
                ),
                "best_drawdown_gate_max_drawdown_pct": (
                    float(best_drawdown_gate["max_drawdown_pct"]) if best_drawdown_gate is not None else None
                ),
                "best_drawdown_gate_candidate": (
                    str(best_drawdown_gate["candidate"]) if best_drawdown_gate is not None else None
                ),
            }
        },
    }
