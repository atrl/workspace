from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


TAIL_RISK_SOURCE_FIELDS = ["VIX9D", "VIX3M", "VVIX", "SKEW"]
TAIL_RISK_REQUIRED_PANEL_COLUMNS = {
    "soxl_close",
    "soxl_volume",
    "soxx_close",
    "soxx_volume",
    "vix",
}


def _normalize_datetime_frame(frame: pd.DataFrame, *, value_col: str, output_col: str) -> pd.DataFrame:
    date_cols = [col for col in frame.columns if col.lower() in {"date", "datetime", "observation_date"}]
    if not date_cols:
        raise ValueError("tail-risk source frame requires a date/datetime/observation_date column")
    if value_col not in frame.columns:
        raise ValueError(f"tail-risk source frame missing value column: {value_col}")
    date_col = date_cols[0]
    normalized = frame[[date_col, value_col]].copy()
    normalized["datetime"] = pd.to_datetime(normalized[date_col], errors="coerce").dt.normalize()
    normalized[output_col] = pd.to_numeric(normalized[value_col], errors="coerce")
    normalized = normalized.dropna(subset=["datetime"]).sort_values("datetime")
    return normalized[["datetime", output_col]].groupby("datetime", as_index=False).last()


def _normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    date_cols = [col for col in panel.columns if col.lower() in {"date", "datetime"}]
    if not date_cols:
        raise ValueError("tail-risk panel requires a date or datetime column")
    missing = sorted(TAIL_RISK_REQUIRED_PANEL_COLUMNS - set(panel.columns))
    if missing:
        raise ValueError(f"tail-risk panel missing columns: {missing}")
    date_col = date_cols[0]
    normalized = panel.copy()
    normalized["datetime"] = pd.to_datetime(normalized[date_col], errors="coerce").dt.normalize()
    normalized = normalized.dropna(subset=["datetime"]).sort_values("datetime")
    return normalized.groupby("datetime", as_index=False).last()


def _zscore(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std()
    return ((series - mean) / (std + 1e-12)).fillna(0.0)


def build_tail_risk_external_frame(
    *,
    panel: pd.DataFrame,
    vix9d: pd.DataFrame,
    vix3m: pd.DataFrame,
    vvix: pd.DataFrame,
    skew: pd.DataFrame,
    start: str,
    end: str,
    zscore_min_periods: int = 20,
) -> pd.DataFrame:
    """Merge SOXL/SOXX/VIX with VIX term-structure, VVIX, and SKEW features."""

    merged = _normalize_panel(panel)
    for source, value_col, output_col in [
        (vix9d, "CLOSE", "vix9d"),
        (vix3m, "CLOSE", "vix3m"),
        (vvix, "VVIX", "vvix"),
        (skew, "SKEW", "skew"),
    ]:
        merged = merged.merge(
            _normalize_datetime_frame(source, value_col=value_col, output_col=output_col),
            on="datetime",
            how="left",
        )

    merged = merged.sort_values("datetime")
    for col in ["vix9d", "vix3m", "vvix", "skew"]:
        merged[col] = merged[col].ffill()
    merged = merged.loc[
        (merged["datetime"] >= pd.Timestamp(start)) & (merged["datetime"] <= pd.Timestamp(end))
    ].copy()

    merged["next_return"] = merged["soxl_close"].pct_change().shift(-1)
    merged["same_day_return"] = merged["soxl_close"].pct_change()
    merged["soxx_return"] = merged["soxx_close"].pct_change()
    merged["trend_5"] = merged["soxl_close"] / merged["soxl_close"].shift(5) - 1
    merged["trend_20"] = merged["soxl_close"] / merged["soxl_close"].shift(20) - 1
    merged["trend_60"] = merged["soxl_close"] / merged["soxl_close"].shift(60) - 1
    merged["hv20"] = merged["same_day_return"].rolling(20, min_periods=20).std() * np.sqrt(252)
    merged["vix9d_vix3m"] = merged["vix9d"] / merged["vix3m"]
    merged["vix_vix3m"] = merged["vix"] / merged["vix3m"]
    merged["vvix_vix"] = merged["vvix"] / merged["vix"]

    zscore_specs = {
        "vix9d_vix3m": 126,
        "vix_vix3m": 126,
        "vvix_vix": 126,
        "skew": 252,
        "vvix": 126,
        "vix": 126,
    }
    for col, window in zscore_specs.items():
        merged[f"{col}_z"] = _zscore(merged[col], window=window, min_periods=zscore_min_periods)

    required = [
        "next_return",
        "trend_20",
        "trend_60",
        "hv20",
        "vix9d_vix3m_z",
        "vvix_vix_z",
        "skew_z",
    ]
    result = merged.dropna(subset=required).reset_index(drop=True)
    result.attrs["source_fields"] = TAIL_RISK_SOURCE_FIELDS
    result.attrs["source_coverage_end"] = min(
        _normalize_datetime_frame(vix9d, value_col="CLOSE", output_col="vix9d")["datetime"].max(),
        _normalize_datetime_frame(vix3m, value_col="CLOSE", output_col="vix3m")["datetime"].max(),
        _normalize_datetime_frame(vvix, value_col="VVIX", output_col="vvix")["datetime"].max(),
        _normalize_datetime_frame(skew, value_col="SKEW", output_col="skew")["datetime"].max(),
    ).date().isoformat()
    return result


def compound_return_pct(returns: Iterable[float]) -> float:
    values = pd.Series(list(returns), dtype="float64").fillna(0.0)
    if values.empty:
        return 0.0
    return float(((1.0 + values).prod() - 1.0) * 100.0)


def max_drawdown_pct(returns: Iterable[float]) -> float:
    values = pd.Series(list(returns), dtype="float64").fillna(0.0)
    if values.empty:
        return 0.0
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def evaluate_tail_risk_candidate(
    frame: pd.DataFrame,
    *,
    branch: str,
    candidate: str,
    exposure: Iterable[float],
    baseline_return_pct: float,
    max_drawdown_floor: float,
    params: dict[str, Any],
    cost_per_turnover: float = 0.0002,
) -> dict[str, Any]:
    exposure_series = pd.Series(list(exposure), dtype="float64")
    if len(exposure_series) != len(frame):
        raise ValueError("tail-risk exposure length must match feature frame length")
    strategy_returns = frame["next_return"].reset_index(drop=True).astype("float64") * exposure_series
    turnover = float(np.abs(np.diff(np.r_[0.0, exposure_series.to_numpy()])).sum())
    if not strategy_returns.empty:
        strategy_returns.iloc[0] = strategy_returns.iloc[0] - turnover * cost_per_turnover

    return_pct = compound_return_pct(strategy_returns)
    drawdown_pct = max_drawdown_pct(strategy_returns)
    beats_baseline = return_pct > float(baseline_return_pct)
    passes_drawdown = drawdown_pct >= float(max_drawdown_floor)
    if not passes_drawdown:
        decision = "Drop"
        decision_basis = "drawdown_gate"
    elif not beats_baseline:
        decision = "Iterate"
        decision_basis = "return_gate"
    else:
        decision = "Promote"
        decision_basis = "beats_baseline_and_drawdown_gate"

    return {
        "branch": branch,
        "candidate": candidate,
        "return_pct": float(return_pct),
        "baseline_return_pct": float(baseline_return_pct),
        "return_improvement_pct": float(return_pct - baseline_return_pct),
        "max_drawdown_pct": float(drawdown_pct),
        "max_drawdown_floor": float(max_drawdown_floor),
        "turnover": turnover,
        "cost_per_turnover": float(cost_per_turnover),
        "beats_baseline_return": bool(beats_baseline),
        "passes_drawdown_gate": bool(passes_drawdown),
        "decision": decision,
        "decision_basis": decision_basis,
        "params": json.dumps(params, sort_keys=True),
        "coverage_start": pd.Timestamp(frame["datetime"].iloc[0]).date().isoformat() if len(frame) else None,
        "coverage_end": pd.Timestamp(frame["datetime"].iloc[-1]).date().isoformat() if len(frame) else None,
    }


def summarize_tail_risk_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"tail-risk branch has no candidate rows: {branch}")
    best_return = rows.sort_values("return_pct", ascending=False).iloc[0]
    drawdown_gate = rows.loc[rows["passes_drawdown_gate"] == True].sort_values(  # noqa: E712
        "return_pct", ascending=False
    )
    best_drawdown_gate = drawdown_gate.iloc[0] if not drawdown_gate.empty else best_return
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
        "best_drawdown_gate_return_pct": float(best_drawdown_gate["return_pct"]),
        "best_drawdown_gate_max_drawdown_pct": float(best_drawdown_gate["max_drawdown_pct"]),
        "best_return_candidate": str(best_return["candidate"]),
        "best_drawdown_gate_candidate": str(best_drawdown_gate["candidate"]),
    }
