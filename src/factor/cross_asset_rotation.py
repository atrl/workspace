from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .tail_risk_external import compound_return_pct, max_drawdown_pct


def _normalize_symbol_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    date_cols = [col for col in frame.columns if col.lower() in {"date", "datetime"}]
    if not date_cols:
        raise ValueError(f"{symbol} frame requires a date or datetime column")
    required = {"close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{symbol} frame missing columns: {missing}")
    date_col = date_cols[0]
    normalized = frame[[date_col, "close", "volume"]].copy()
    normalized["datetime"] = pd.to_datetime(normalized[date_col], errors="coerce").dt.normalize()
    normalized[f"{symbol}_close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized[f"{symbol}_volume"] = pd.to_numeric(normalized["volume"], errors="coerce")
    normalized = normalized.dropna(subset=["datetime", f"{symbol}_close"]).sort_values("datetime")
    return normalized[["datetime", f"{symbol}_close", f"{symbol}_volume"]].groupby("datetime", as_index=False).last()


def build_cross_asset_price_frame(
    raw_frames: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    if not raw_frames:
        raise ValueError("at least one cross-asset frame is required")
    merged: pd.DataFrame | None = None
    for raw_symbol, frame in raw_frames.items():
        symbol = raw_symbol.upper()
        normalized = _normalize_symbol_frame(symbol, frame)
        merged = normalized if merged is None else merged.merge(normalized, on="datetime", how="inner")
    assert merged is not None
    result = merged.loc[
        (merged["datetime"] >= pd.Timestamp(start)) & (merged["datetime"] <= pd.Timestamp(end))
    ].reset_index(drop=True)
    if result.empty:
        raise ValueError("cross-asset frame has no rows after date filtering")
    result.attrs["symbols"] = [symbol.upper() for symbol in raw_frames]
    result.attrs["source_coverage_start"] = result["datetime"].min().date().isoformat()
    result.attrs["source_coverage_end"] = result["datetime"].max().date().isoformat()
    return result


def _candidate_decision(return_pct: float, drawdown_pct: float, baseline_return_pct: float, floor: float) -> tuple[str, str]:
    beats_baseline = return_pct > baseline_return_pct
    passes_drawdown = drawdown_pct >= floor
    if not passes_drawdown:
        return "Drop", "drawdown_gate"
    if not beats_baseline:
        return "Iterate", "return_gate"
    return "Promote", "beats_baseline_and_drawdown_gate"


def _strategy_row(
    *,
    branch: str,
    candidate: str,
    returns: pd.Series,
    exposure: pd.Series,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    strategy_returns = returns.astype("float64").fillna(0.0).copy()
    turnover = float(exposure.astype("float64").diff().abs().fillna(exposure.astype("float64").abs()).sum())
    if not strategy_returns.empty:
        strategy_returns.iloc[0] -= turnover * 0.0002
    return_pct = compound_return_pct(strategy_returns)
    drawdown_pct = max_drawdown_pct(strategy_returns)
    decision, decision_basis = _candidate_decision(
        return_pct,
        drawdown_pct,
        float(baseline_return_pct),
        float(max_drawdown_floor),
    )
    return {
        "branch": branch,
        "candidate": candidate,
        "return_pct": float(return_pct),
        "baseline_return_pct": float(baseline_return_pct),
        "return_improvement_pct": float(return_pct - baseline_return_pct),
        "max_drawdown_pct": float(drawdown_pct),
        "max_drawdown_floor": float(max_drawdown_floor),
        "turnover": turnover,
        "beats_baseline_return": bool(return_pct > baseline_return_pct),
        "passes_drawdown_gate": bool(drawdown_pct >= max_drawdown_floor),
        "decision": decision,
        "decision_basis": decision_basis,
        "params": json.dumps(params, sort_keys=True),
        "coverage_start": returns.index.min().date().isoformat(),
        "coverage_end": returns.index.max().date().isoformat(),
    }


def evaluate_cross_asset_rotation_candidates(
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    symbols: list[str],
    rotation_lookbacks: list[int] | None = None,
    breadth_lookbacks: list[int] | None = None,
    caps: list[float] | None = None,
    breadth_scales: list[float] | None = None,
) -> pd.DataFrame:
    symbols = [symbol.upper() for symbol in symbols]
    rotation_lookbacks = rotation_lookbacks or [20, 63, 126, 252]
    breadth_lookbacks = breadth_lookbacks or [20, 63, 126, 252]
    caps = caps or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    breadth_scales = breadth_scales or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    prices = frame.set_index("datetime")[[f"{symbol}_close" for symbol in symbols]].rename(
        columns={f"{symbol}_close": symbol for symbol in symbols}
    )
    next_returns = prices.pct_change().shift(-1).dropna(how="all")
    prices = prices.loc[next_returns.index]
    date_index = next_returns.index
    returns_array = next_returns.to_numpy(dtype="float64")
    rows: list[dict[str, Any]] = []

    for lookback in rotation_lookbacks:
        momentum = (prices / prices.shift(lookback) - 1.0).reindex(date_index).to_numpy(dtype="float64")
        valid_momentum = np.isfinite(momentum)
        vol20 = (
            prices.pct_change().rolling(20, min_periods=20).std().reindex(date_index).to_numpy(dtype="float64")
            * np.sqrt(252)
        )
        for topn in [1, 2, 3]:
            for abs_threshold in [-0.05, 0.0, 0.05]:
                for mode in ["equal", "vol10", "vol20", "vol30"]:
                    for cap in caps:
                        weights = np.zeros_like(returns_array)
                        for row_index in range(len(returns_array)):
                            values = np.where(valid_momentum[row_index], momentum[row_index], -np.inf)
                            order = np.argsort(values)[::-1]
                            picks = [int(idx) for idx in order if values[idx] > abs_threshold][:topn]
                            if len(picks) == 0:
                                continue
                            if mode == "equal":
                                weights[row_index, picks] = float(cap) / len(picks)
                                continue
                            target = {"vol10": 0.10, "vol20": 0.20, "vol30": 0.30}[mode]
                            raw = []
                            for symbol_index in picks:
                                vol = float(vol20[row_index, symbol_index])
                                raw.append(min(float(cap), target / (vol + 1e-12)) if np.isfinite(vol) and vol > 0 else 0.0)
                            raw_weights = np.asarray(raw, dtype="float64")
                            if raw_weights.sum() > cap and raw_weights.sum() > 0:
                                raw_weights = raw_weights / raw_weights.sum() * cap
                            weights[row_index, picks] = raw_weights
                        returns = pd.Series(np.nansum(weights * returns_array, axis=1), index=date_index)
                        exposure = pd.Series(np.abs(weights).sum(axis=1), index=date_index)
                        rows.append(
                            _strategy_row(
                                branch="relative_momentum_rotation",
                                candidate=f"lb{lookback}_top{topn}_abs{abs_threshold}_{mode}_cap{cap}",
                                returns=returns,
                                exposure=exposure,
                                baseline_return_pct=baseline_return_pct,
                                max_drawdown_floor=max_drawdown_floor,
                                params={
                                    "lookback": lookback,
                                    "topn": topn,
                                    "abs_threshold": abs_threshold,
                                    "mode": mode,
                                    "cap": cap,
                                },
                            )
                        )

    soxl_returns = next_returns["SOXL"]
    for lookback in breadth_lookbacks:
        momentum = prices / prices.shift(lookback) - 1.0
        soxl_vol20 = prices["SOXL"].pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
        for agree in [2, 3, 4, 5]:
            for scale in breadth_scales:
                for volcap in [None, 0.4, 0.6, 0.8, 1.0]:
                    breadth = (momentum > 0).sum(axis=1) >= agree
                    exposure = pd.Series(np.where(breadth, float(scale), 0.0), index=next_returns.index)
                    if volcap is not None:
                        exposure = np.minimum(exposure, float(volcap) / (soxl_vol20 + 1e-12))
                        exposure = pd.Series(np.where(np.isfinite(exposure), exposure, 0.0), index=next_returns.index)
                    rows.append(
                        _strategy_row(
                            branch="cross_asset_breadth_soxl",
                            candidate=f"lb{lookback}_agree{agree}_scale{scale}_volcap{volcap}",
                            returns=soxl_returns * exposure,
                            exposure=exposure,
                            baseline_return_pct=baseline_return_pct,
                            max_drawdown_floor=max_drawdown_floor,
                            params={
                                "lookback": lookback,
                                "agree": agree,
                                "scale": scale,
                                "volcap": volcap,
                            },
                        )
                    )
    return pd.DataFrame(rows)


def summarize_cross_asset_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"cross-asset branch has no candidate rows: {branch}")
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
