from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.agentic_alpha import AGENTIC_ALPHA_STAGE_ORDER


META_LABEL_SOURCE_FIELDS = [
    "SOXL_RETURNS",
    "SOXX_RETURNS",
    "SOXL_TREND",
    "SOXX_TREND",
    "SOXL_DRAWDOWN",
    "SOXL_VIX",
    "SOXL_VOLUME",
]
META_LABEL_BRANCH = "meta_label_probability_gate"
META_LABEL_MODEL_FAMILIES = ["LogisticRegression", "HistGradientBoosting"]


def _wide_qlib_panel(frame: pd.DataFrame, *, symbols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", "$close", "$volume"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"meta-label qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()

    pieces: list[pd.DataFrame] = []
    for symbol in [item.upper() for item in symbols]:
        symbol_frame = normalized.loc[normalized["instrument"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"meta-label qlib frame missing instrument: {symbol}")
        columns = {"$close": f"{symbol.lower()}_close", "$volume": f"{symbol.lower()}_volume"}
        if "$vix" in symbol_frame.columns:
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


def _rolling_mean(series: pd.Series, window: int, min_history: int) -> pd.Series:
    return series.rolling(window, min_periods=min(window, min_history)).mean()


def _rolling_std(series: pd.Series, window: int, min_history: int) -> pd.Series:
    return series.rolling(window, min_periods=min(window, min_history)).std()


def build_meta_label_probability_frame(
    qlib_frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    min_history: int = 60,
) -> pd.DataFrame:
    """Build SOXL/SOXX daily features and primary side labels for meta-label gates."""

    panel = _wide_qlib_panel(qlib_frame, symbols=["SOXL", "SOXX"])
    soxl_close = panel["soxl_close"].astype("float64")
    soxx_close = panel["soxx_close"].astype("float64")
    volume = panel["soxl_volume"].astype("float64")
    vix = panel["vix"].astype("float64") if "vix" in panel else pd.Series(np.nan, index=panel.index)
    soxl_return = soxl_close.pct_change()
    soxx_return = soxx_close.pct_change()

    payload: dict[str, pd.Series] = {
        "datetime": pd.Series(panel.index, index=panel.index),
        "next_return": soxl_return.shift(-1),
        "vix": vix,
        "vix_chg1": vix.pct_change(),
        "vix_chg5": vix.pct_change(5),
    }
    windows = [1, 2, 3, 5, 10, 21, 40, 63, 84, 126, 150, 200, 252]
    for window in windows:
        payload[f"ret{window}"] = soxl_close.pct_change(window)
        payload[f"soxx_ret{window}"] = soxx_close.pct_change(window)
        payload[f"ma{window}"] = soxl_close / _rolling_mean(soxl_close, window, min_history) - 1.0
        payload[f"soxx_ma{window}"] = soxx_close / _rolling_mean(soxx_close, window, min_history) - 1.0
        payload[f"dd{window}"] = soxl_close / soxl_close.rolling(window, min_periods=min(window, min_history)).max() - 1.0
        payload[f"volchg{window}"] = volume / _rolling_mean(volume, window, min_history) - 1.0
        if window >= 2:
            payload[f"vol{window}"] = _rolling_std(soxl_return, window, min_history) * np.sqrt(252)
            payload[f"soxx_vol{window}"] = _rolling_std(soxx_return, window, min_history) * np.sqrt(252)

    features = pd.DataFrame(payload, index=panel.index)
    for window in [21, 40, 63, 126, 252]:
        vix_std = _rolling_std(vix, window, min_history)
        features[f"vix_z{window}"] = (vix - _rolling_mean(vix, window, min_history)) / (vix_std + 1e-12)
        if f"vol{window}" in features:
            features[f"vol_vix_ratio{window}"] = features[f"vol{window}"] / (vix / 100.0)

    trend_ok = (
        (features["ma150"].fillna(features["ma40"]) > -0.08)
        & (features["soxx_ma150"].fillna(features["soxx_ma40"]) > -0.08)
        & (features["ma40"] > -0.10)
        & (features["vix"].fillna(20.0) < 45.0)
    )
    features["base_trend"] = trend_ok.astype(bool)
    features["base_dip"] = (features["base_trend"] & (features["ret2"] < -0.055)).astype(bool)
    features["base_calendar"] = (
        features["datetime"].dt.day.isin([1, 2, 3, 27, 28, 29, 30, 31])
        & (features["vix"].fillna(20.0) < 38.0)
    ).astype(bool)
    features["base_any"] = features[["base_trend", "base_dip", "base_calendar"]].any(axis=1).astype(bool)

    required = ["datetime", "next_return", "ret2", "ma40", "ma150", "soxx_ma150"]
    result = features.replace([np.inf, -np.inf], np.nan)
    result = result.loc[(result["datetime"] >= pd.Timestamp(start)) & (result["datetime"] <= pd.Timestamp(end))]
    result = result.dropna(subset=required).reset_index(drop=True)
    result.attrs["source_fields"] = META_LABEL_SOURCE_FIELDS
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
            "avg_exposure": float(exposure.mean()) if len(exposure) else 0.0,
            "avg_abs_exposure": float(np.abs(exposure).mean()) if len(exposure) else 0.0,
            "max_exposure": float(exposure.max()) if len(exposure) else 0.0,
            "min_exposure": float(exposure.min()) if len(exposure) else 0.0,
        }
    equity = np.cumprod(1.0 + strategy_returns)
    if len(equity) == 0:
        drawdown = np.array([0.0])
        return_pct = 0.0
    else:
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        return_pct = float((equity[-1] - 1.0) * 100.0)
    return {
        "return_pct": return_pct,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "turnover": float(daily_turnover.sum()),
        "avg_exposure": float(exposure.mean()) if len(exposure) else 0.0,
        "avg_abs_exposure": float(np.abs(exposure).mean()) if len(exposure) else 0.0,
        "max_exposure": float(exposure.max()) if len(exposure) else 0.0,
        "min_exposure": float(exposure.min()) if len(exposure) else 0.0,
    }


def _decision(
    *,
    beats_baseline: bool,
    passes_drawdown: bool,
    passes_exposure_sanity: bool,
    passes_trade_count_sanity: bool,
) -> tuple[str, str]:
    if not passes_drawdown:
        return "Drop", "drawdown_gate"
    if not beats_baseline:
        return "Iterate", "return_gate"
    if not passes_exposure_sanity:
        return "Drop", "exposure_sanity_gate"
    if not passes_trade_count_sanity:
        return "Drop", "trade_count_sanity_gate"
    return "Promote", "beats_baseline_drawdown_and_sanity_gates"


def evaluate_probability_gate_grid(
    frame: pd.DataFrame,
    *,
    branch: str,
    probability_col: str,
    event_col: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    thresholds: list[float] | None = None,
    scales: list[float] | None = None,
    size_modes: list[str] | None = None,
    max_exposure_sanity: float = 10.0,
    min_trade_count_sanity: int = 20,
    cost_per_turnover: float = 0.0002,
    model_name: str = "precomputed_probability",
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("meta-label probability frame is empty")
    required = {"datetime", "next_return", probability_col, event_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"meta-label probability frame missing columns: {missing}")
    thresholds = thresholds or [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    scales = scales or [1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    size_modes = size_modes or ["binary", "probability"]

    data = frame.sort_values("datetime").reset_index(drop=True).copy()
    probabilities = data[probability_col].astype("float64").to_numpy()
    events = data[event_col].astype(bool).to_numpy()
    returns = data["next_return"].astype("float64").to_numpy()
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        gate = events & (probabilities >= float(threshold))
        selected_returns = returns[gate]
        for size_mode in size_modes:
            if size_mode not in {"binary", "probability"}:
                raise ValueError("size_modes entries must be binary or probability")
            for scale in scales:
                if size_mode == "binary":
                    exposure = gate.astype("float64") * float(scale)
                else:
                    exposure = np.where(gate, probabilities * float(scale), 0.0)
                stats = _strategy_stats(returns, exposure, cost_per_turnover=cost_per_turnover)
                trade_count = int(gate.sum())
                beats_baseline = stats["return_pct"] > float(baseline_return_pct)
                passes_drawdown = stats["max_drawdown_pct"] >= float(max_drawdown_floor)
                passes_exposure_sanity = stats["max_exposure"] <= float(max_exposure_sanity)
                passes_trade_count_sanity = trade_count >= int(min_trade_count_sanity)
                decision, decision_basis = _decision(
                    beats_baseline=beats_baseline,
                    passes_drawdown=passes_drawdown,
                    passes_exposure_sanity=passes_exposure_sanity,
                    passes_trade_count_sanity=passes_trade_count_sanity,
                )
                rows.append(
                    {
                        "branch": branch,
                        "candidate": (
                            f"{event_col}_{model_name}_p{threshold:g}_{size_mode}_scale{float(scale):g}"
                        ),
                        "model": model_name,
                        "event_col": event_col,
                        "threshold": float(threshold),
                        "size_mode": size_mode,
                        "scale": float(scale),
                        "return_pct": float(stats["return_pct"]),
                        "baseline_return_pct": float(baseline_return_pct),
                        "return_improvement_pct": float(stats["return_pct"] - baseline_return_pct),
                        "max_drawdown_pct": float(stats["max_drawdown_pct"]),
                        "max_drawdown_floor": float(max_drawdown_floor),
                        "turnover": float(stats["turnover"]),
                        "avg_exposure": float(stats["avg_exposure"]),
                        "avg_abs_exposure": float(stats["avg_abs_exposure"]),
                        "max_exposure": float(stats["max_exposure"]),
                        "min_exposure": float(stats["min_exposure"]),
                        "trade_count": trade_count,
                        "win_rate": float((selected_returns > 0.0).mean()) if trade_count else 0.0,
                        "cost_per_turnover": float(cost_per_turnover),
                        "beats_baseline_return": bool(beats_baseline),
                        "passes_drawdown_gate": bool(passes_drawdown),
                        "passes_exposure_sanity_gate": bool(passes_exposure_sanity),
                        "passes_trade_count_sanity_gate": bool(passes_trade_count_sanity),
                        "passes_sanity_gate": bool(passes_exposure_sanity and passes_trade_count_sanity),
                        "decision": decision,
                        "decision_basis": decision_basis,
                        "params": json.dumps(
                            {
                                "event_col": event_col,
                                "model": model_name,
                                "threshold": float(threshold),
                                "size_mode": size_mode,
                                "scale": float(scale),
                                "max_exposure_sanity": float(max_exposure_sanity),
                                "min_trade_count_sanity": int(min_trade_count_sanity),
                            },
                            sort_keys=True,
                        ),
                        "coverage_start": pd.Timestamp(data["datetime"].iloc[0]).date().isoformat(),
                        "coverage_end": pd.Timestamp(data["datetime"].iloc[-1]).date().isoformat(),
                    }
                )
    return pd.DataFrame(rows)


def _model_factory(model: str, random_state: int):
    if model == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=2000, class_weight="balanced"))
    if model == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.03,
            random_state=random_state,
        )
    raise ValueError("model must be one of: logit, hgb")


def evaluate_meta_label_probability_candidates(
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    feature_cols: list[str] | None = None,
    event_cols: list[str] | None = None,
    models: list[str] | None = None,
    thresholds: list[float] | None = None,
    scales: list[float] | None = None,
    size_modes: list[str] | None = None,
    train_min_events: int = 80,
    max_exposure_sanity: float = 10.0,
    min_trade_count_sanity: int = 20,
    cost_per_turnover: float = 0.0002,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("meta-label feature frame is empty")
    data = frame.sort_values("datetime").reset_index(drop=True).copy()
    event_cols = event_cols or ["base_trend", "base_dip", "base_calendar", "base_any"]
    models = models or ["logit", "hgb"]
    excluded = {"datetime", "next_return", *event_cols}
    feature_cols = feature_cols or [
        col for col in data.select_dtypes(include=[np.number]).columns if col not in excluded
    ]
    missing = [col for col in ["datetime", "next_return", *event_cols, *feature_cols] if col not in data.columns]
    if missing:
        raise ValueError(f"meta-label feature frame missing columns: {missing}")

    rows: list[pd.DataFrame] = []
    years = range(pd.Timestamp(data["datetime"].min()).year, pd.Timestamp(data["datetime"].max()).year + 1)
    for event_col in event_cols:
        for model_name in models:
            prediction_pieces: list[pd.DataFrame] = []
            for year in years:
                test_start = pd.Timestamp(f"{year}-01-01")
                test_end = pd.Timestamp(f"{year}-12-31")
                train_all = data.loc[data["datetime"] < test_start].dropna(subset=[*feature_cols, "next_return"])
                train = train_all.loc[train_all[event_col].astype(bool)].copy()
                test = data.loc[(data["datetime"] >= test_start) & (data["datetime"] <= test_end)].dropna(
                    subset=[*feature_cols, "next_return"]
                )
                if len(train) < int(train_min_events) or test.empty:
                    continue
                y_train = train["next_return"].gt(0.0).astype(int)
                if y_train.nunique() < 2:
                    continue
                model = _model_factory(model_name, year)
                model.fit(train[feature_cols], y_train)
                probabilities = model.predict_proba(test[feature_cols])[:, 1]
                prediction_pieces.append(
                    pd.DataFrame(
                        {
                            "datetime": test["datetime"].to_numpy(),
                            "next_return": test["next_return"].to_numpy(dtype="float64"),
                            "probability": probabilities,
                            event_col: test[event_col].astype(bool).to_numpy(),
                        }
                    )
                )
            if not prediction_pieces:
                continue
            predictions = pd.concat(prediction_pieces, ignore_index=True).sort_values("datetime")
            model_label = "LogisticRegression" if model_name == "logit" else "HistGradientBoosting"
            rows.append(
                evaluate_probability_gate_grid(
                    predictions,
                    branch=META_LABEL_BRANCH,
                    probability_col="probability",
                    event_col=event_col,
                    baseline_return_pct=baseline_return_pct,
                    max_drawdown_floor=max_drawdown_floor,
                    thresholds=thresholds,
                    scales=scales,
                    size_modes=size_modes,
                    max_exposure_sanity=max_exposure_sanity,
                    min_trade_count_sanity=min_trade_count_sanity,
                    cost_per_turnover=cost_per_turnover,
                    model_name=model_label,
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def summarize_meta_label_probability_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"meta-label branch has no candidate rows: {branch}")
    best_return = rows.sort_values("return_pct", ascending=False).iloc[0]
    drawdown_gate = rows.loc[rows["passes_drawdown_gate"] == True].sort_values(  # noqa: E712
        "return_pct", ascending=False
    )
    best_drawdown_gate = drawdown_gate.iloc[0] if not drawdown_gate.empty else best_return
    raw_passes = rows.loc[rows["beats_baseline_return"] & rows["passes_drawdown_gate"]]
    passers = rows.loc[
        rows["beats_baseline_return"] & rows["passes_drawdown_gate"] & rows["passes_sanity_gate"]
    ]
    return {
        "grid_rows": int(len(rows)),
        "coverage_start": str(best_return.get("coverage_start")),
        "coverage_end": str(best_return.get("coverage_end")),
        "goal_baseline_return_pct": float(baseline_return_pct),
        "max_drawdown_floor": float(max_drawdown_floor),
        "raw_passes_both_count": int(len(raw_passes)),
        "passes_both_count": int(len(passers)),
        "best_return_pct": float(best_return["return_pct"]),
        "best_return_max_drawdown_pct": float(best_return["max_drawdown_pct"]),
        "best_return_max_exposure": float(best_return["max_exposure"]),
        "best_return_trade_count": int(best_return["trade_count"]),
        "best_drawdown_gate_available": bool(not drawdown_gate.empty),
        "best_drawdown_gate_return_pct": float(best_drawdown_gate["return_pct"]),
        "best_drawdown_gate_max_drawdown_pct": float(best_drawdown_gate["max_drawdown_pct"]),
        "best_drawdown_gate_max_exposure": float(best_drawdown_gate["max_exposure"]),
        "best_drawdown_gate_trade_count": int(best_drawdown_gate["trade_count"]),
        "best_return_candidate": str(best_return["candidate"]),
        "best_drawdown_gate_candidate": str(best_drawdown_gate["candidate"]),
    }


def compact_meta_label_probability_summary(
    metrics: pd.DataFrame,
    *,
    branches: list[str],
    baseline_return_pct: float,
    max_drawdown_floor: float,
    max_exposure_sanity: float = 10.0,
    min_trade_count_sanity: int = 20,
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("meta-label metrics are empty")
    branch_results = {
        branch: summarize_meta_label_probability_branch(
            metrics,
            branch=branch,
            baseline_return_pct=baseline_return_pct,
            max_drawdown_floor=max_drawdown_floor,
        )
        for branch in branches
    }
    raw_passes = metrics.loc[metrics["beats_baseline_return"] & metrics["passes_drawdown_gate"]]
    passers = metrics.loc[
        metrics["beats_baseline_return"] & metrics["passes_drawdown_gate"] & metrics["passes_sanity_gate"]
    ]
    best_candidate = None
    if not passers.empty:
        best_candidate = str(passers.sort_values("return_pct", ascending=False).iloc[0]["candidate"])
    return {
        "workflow": "agentic_alpha_meta_label_probability_gate_search",
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "goal_achieved": bool(best_candidate is not None),
        "best_candidate": best_candidate,
        "raw_passes_both_count": int(len(raw_passes)),
        "passes_both_count": int(len(passers)),
        "branch_results": branch_results,
        "risk_sanity": {
            "max_exposure_sanity": float(max_exposure_sanity),
            "min_trade_count_sanity": int(min_trade_count_sanity),
            "raw_numeric_passes_require_sanity": True,
        },
        "model_families": META_LABEL_MODEL_FAMILIES,
        "selected_features": META_LABEL_SOURCE_FIELDS,
    }
