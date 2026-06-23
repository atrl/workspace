from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .tail_risk_external import evaluate_tail_risk_candidate


FACTOR_LIBRARY_SOURCE_FIELDS = [
    "RB_RSI6_OVERSOLD",
    "RB_SELL_COMPOSITE",
    "BIAS_10",
    "BIAS_10_MOM",
    "REL_VOLUME_RATIO",
    "PRICE_RATIO_MOM",
    "SOXX_BIAS_10",
    "RB_VOL_REGIME",
]


def _wide_qlib_panel(frame: pd.DataFrame, *, symbols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", "$close", "$volume"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"factor-library qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()

    pieces: list[pd.DataFrame] = []
    for symbol in [item.upper() for item in symbols]:
        symbol_frame = normalized.loc[normalized["instrument"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"factor-library qlib frame missing instrument: {symbol}")
        columns = {
            "$close": f"{symbol.lower()}_close",
            "$volume": f"{symbol.lower()}_volume",
        }
        if "$vix" in symbol_frame.columns:
            columns["$vix"] = f"{symbol.lower()}_vix"
        selected = symbol_frame[["datetime", *columns]].rename(columns=columns)
        pieces.append(selected.groupby("datetime", as_index=False).last())

    panel = pieces[0]
    for piece in pieces[1:]:
        panel = panel.merge(piece, on="datetime", how="inner")
    panel = panel.sort_values("datetime").set_index("datetime")
    if "soxl_vix" in panel.columns:
        panel["vix"] = panel["soxl_vix"].ffill()
    return panel


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    losses = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    relative_strength = gains / (losses + 1e-12)
    return 100.0 - 100.0 / (1.0 + relative_strength)


def build_factor_library_reversal_frame(
    qlib_frame: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Rebuild selected local factor-library signals from qlib daily SOXL/SOXX features."""

    panel = _wide_qlib_panel(qlib_frame, symbols=["SOXL", "SOXX"])
    close = panel["soxl_close"].astype("float64")
    soxx_close = panel["soxx_close"].astype("float64")
    volume = panel["soxl_volume"].astype("float64")
    soxx_volume = panel["soxx_volume"].astype("float64")
    daily_return = close.pct_change()

    rsi6 = _rsi(close, 6)
    rsi14 = _rsi(close, 14)
    hv21 = daily_return.rolling(21, min_periods=21).std() * np.sqrt(252)
    vol_coef = hv21 / (hv21.rolling(252, min_periods=126).median() + 1e-12)
    bias10 = close / close.rolling(10, min_periods=10).mean() - 1.0
    soxx_bias10 = soxx_close / soxx_close.rolling(10, min_periods=10).mean() - 1.0
    ratio = close / soxx_close
    rel_volume = (volume / volume.rolling(21, min_periods=21).mean()) / (
        soxx_volume / soxx_volume.rolling(21, min_periods=21).mean() + 1e-12
    )
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ma200 = close.rolling(200, min_periods=200).mean()

    features = pd.DataFrame(
        {
            "datetime": panel.index,
            "next_return": daily_return.shift(-1),
            "rsi6": rsi6,
            "rsi14": rsi14,
            "hv21": hv21,
            "vol_coef": vol_coef,
            "bias10": bias10,
            "bias10_mom": bias10 - bias10.shift(5),
            "soxx_bias10": soxx_bias10,
            "ratio_mom5": ratio.pct_change(5),
            "rel_volume_ratio": rel_volume,
            "cumret30": close / close.shift(30) - 1.0,
            "dd30": close / close.rolling(30, min_periods=30).max() - 1.0,
            "trend20": close / close.shift(20) - 1.0,
            "trend60": close / close.shift(60) - 1.0,
            "trend120": close / close.shift(120) - 1.0,
            "ma20_gt_ma60": (ma20 > ma60).astype("float64"),
            "ma60_gt_ma200": (ma60 > ma200).astype("float64"),
            "price_gt_ma200": (close > ma200).astype("float64"),
            "vix": panel["vix"] if "vix" in panel.columns else np.nan,
        }
    )
    features["oversold6"] = ((30.0 - features["rsi6"]) / 30.0).clip(0.0, 1.0)
    features["overbought6"] = ((features["rsi6"] - 70.0) / 30.0).clip(0.0, 1.0)
    features["oversold14"] = ((35.0 - features["rsi14"]) / 35.0).clip(0.0, 1.0)
    features["bias_drop"] = (-features["bias10"] / 0.12).clip(0.0, 1.0)
    features["bias_pop"] = (features["bias10"] / 0.12).clip(0.0, 1.0)
    features["cum_drop"] = (-features["cumret30"] / 0.30).clip(0.0, 1.0)
    features["dd30_drop"] = (-features["dd30"] / 0.25).clip(0.0, 1.0)
    features["ratio_drop"] = (-features["ratio_mom5"] / 0.08).clip(0.0, 1.0)
    features["rel_volume_low"] = ((1.0 - features["rel_volume_ratio"]) / 0.50).clip(0.0, 1.0)
    features["soxx_drop"] = (-features["soxx_bias10"] / 0.08).clip(0.0, 1.0)
    features["sell_score"] = (
        0.45 * features["overbought6"]
        + 0.25 * features["bias_pop"]
        + 0.20 * (features["cumret30"] / 0.30).clip(0.0, 1.0)
        + 0.10 * ((features["vol_coef"] - 1.0).clip(0.0, 2.0) / 2.0)
    ).clip(0.0, 1.0)
    features["buy_score"] = (
        0.25 * features["oversold6"]
        + 0.20 * features["bias_drop"]
        + 0.20 * features["cum_drop"]
        + 0.15 * features["ratio_drop"]
        + 0.10 * features["soxx_drop"]
        + 0.10 * features["rel_volume_low"]
    ).clip(0.0, 1.0)
    features["panic_buy_score"] = (
        0.35 * features["oversold6"]
        + 0.30 * features["dd30_drop"]
        + 0.20 * ((features["vol_coef"] - 1.0).clip(0.0, 2.0) / 2.0)
        + 0.15 * features["ratio_drop"]
    ).clip(0.0, 1.0)
    features["trend_score"] = (
        0.25 * (features["trend20"] > 0).astype("float64")
        + 0.25 * (features["trend60"] > 0).astype("float64")
        + 0.25 * features["ma20_gt_ma60"]
        + 0.25 * features["price_gt_ma200"]
    ).clip(0.0, 1.0)

    result = features.loc[
        (features["datetime"] >= pd.Timestamp(start)) & (features["datetime"] <= pd.Timestamp(end))
    ].dropna(subset=["next_return", "hv21", "vol_coef", "buy_score", "panic_buy_score", "trend_score"])
    result = result.reset_index(drop=True)
    result.attrs["source_fields"] = FACTOR_LIBRARY_SOURCE_FIELDS
    result.attrs["source_coverage_start"] = panel.index.min().date().isoformat()
    result.attrs["source_coverage_end"] = panel.index.max().date().isoformat()
    return result


def _exposure_with_volcap(exposure: pd.Series, hv21: pd.Series, volcap: float | None) -> pd.Series:
    if volcap is None:
        return exposure.astype("float64")
    capped = np.minimum(exposure.astype("float64"), float(volcap) / (hv21.astype("float64") + 1e-12))
    return pd.Series(capped, index=exposure.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _metric_row(
    frame: pd.DataFrame,
    *,
    branch: str,
    candidate: str,
    exposure: pd.Series,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    row = evaluate_tail_risk_candidate(
        frame,
        branch=branch,
        candidate=candidate,
        exposure=exposure,
        baseline_return_pct=baseline_return_pct,
        max_drawdown_floor=max_drawdown_floor,
        params=params,
    )
    row["avg_exposure_pct"] = float(pd.Series(exposure, dtype="float64").mean() * 100.0)
    row["max_exposure"] = float(pd.Series(exposure, dtype="float64").max())
    return row


def evaluate_factor_library_reversal_candidates(
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float,
    reversal_thresholds: list[float] | None = None,
    hold_days: list[int] | None = None,
    scales: list[float] | None = None,
    volcaps: list[float | None] | None = None,
    trend_sell_thresholds: list[float] | None = None,
    base_scales: list[float] | None = None,
    reentry_boosts: list[float] | None = None,
    target_vols: list[float] | None = None,
    score_caps: list[float] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("factor-library reversal frame is empty")
    data = frame.reset_index(drop=True).copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    index = data.index
    reversal_thresholds = reversal_thresholds or [0.25, 0.35, 0.55, 0.65, 0.75]
    hold_days = hold_days or [1, 3, 5]
    scales = scales or [1.0, 4.0, 10.0]
    volcaps = volcaps or [None, 0.25, 0.80]
    trend_sell_thresholds = trend_sell_thresholds or [0.50, 0.70]
    base_scales = base_scales or [0.25, 1.0, 2.0]
    reentry_boosts = reentry_boosts or [0.0, 1.0]
    target_vols = target_vols or [0.15, 0.80]
    score_caps = score_caps or [3.0, 8.0]

    rows: list[dict[str, Any]] = []
    for score_name in ["buy_score", "panic_buy_score"]:
        score = data[score_name].astype("float64")
        for threshold in reversal_thresholds:
            signal = (score >= threshold).astype("float64")
            for hold in hold_days:
                held = signal.rolling(hold, min_periods=1).max()
                for scale in scales:
                    base_exposure = held * float(scale)
                    for volcap in volcaps:
                        capped = _exposure_with_volcap(base_exposure, data["hv21"], volcap)
                        for gate in ["none", "no_sell_vix30"]:
                            exposure = capped.copy()
                            if gate == "no_sell_vix30":
                                exposure = exposure.where((data["sell_score"] < 0.70) & (data["vix"] < 30.0), 0.0)
                            rows.append(
                                _metric_row(
                                    data,
                                    branch="factor_library_reversal_reentry",
                                    candidate=f"{score_name}_thr{threshold:.2f}_hold{hold}_scale{scale}_volcap{volcap}_{gate}",
                                    exposure=pd.Series(exposure, index=index),
                                    baseline_return_pct=baseline_return_pct,
                                    max_drawdown_floor=max_drawdown_floor,
                                    params={
                                        "score": score_name,
                                        "threshold": threshold,
                                        "hold": hold,
                                        "scale": scale,
                                        "volcap": volcap,
                                        "gate": gate,
                                    },
                                )
                            )

    trend_gates = {
        "trend_score_050": (data["trend_score"] >= 0.50).astype("float64"),
        "trend_score_075": (data["trend_score"] >= 0.75).astype("float64"),
        "price_ma200": data["price_gt_ma200"].astype("float64"),
        "ma_stack": ((data["ma20_gt_ma60"] > 0) & (data["ma60_gt_ma200"] > 0)).astype("float64"),
        "trend60": (data["trend60"] > 0).astype("float64"),
    }
    for trend_name, trend_gate in trend_gates.items():
        for sell_threshold in trend_sell_thresholds:
            sell_gate = data["sell_score"] < sell_threshold
            for buy_threshold in [0.35, 0.55, 0.65]:
                reentry = (data["buy_score"] >= buy_threshold).astype("float64")
                for base_scale in base_scales:
                    for boost in reentry_boosts:
                        raw = trend_gate * float(base_scale) + reentry * float(boost)
                        raw = raw.where(sell_gate, 0.0)
                        for volcap in volcaps:
                            exposure = _exposure_with_volcap(raw, data["hv21"], volcap)
                            rows.append(
                                _metric_row(
                                    data,
                                    branch="factor_library_trend_sell_gate",
                                    candidate=(
                                        f"{trend_name}_sell{sell_threshold}_buy{buy_threshold}"
                                        f"_base{base_scale}_boost{boost}_volcap{volcap}"
                                    ),
                                    exposure=pd.Series(exposure, index=index),
                                    baseline_return_pct=baseline_return_pct,
                                    max_drawdown_floor=max_drawdown_floor,
                                    params={
                                        "trend_gate": trend_name,
                                        "sell_threshold": sell_threshold,
                                        "buy_threshold": buy_threshold,
                                        "base_scale": base_scale,
                                        "reentry_boost": boost,
                                        "volcap": volcap,
                                    },
                                )
                            )

    for score_name in ["buy_score", "panic_buy_score", "trend_score"]:
        score = data[score_name].astype("float64")
        for intercept in [0.0, 0.4]:
            for slope in [1.0, 2.0, 4.0]:
                raw_score = (intercept + slope * score).clip(0.0, 5.0)
                for target_vol in target_vols:
                    raw = raw_score * float(target_vol) / (data["hv21"] + 1e-12)
                    for cap in score_caps:
                        capped = raw.clip(0.0, float(cap))
                        for sell_threshold in [None, 0.70]:
                            exposure = (
                                capped
                                if sell_threshold is None
                                else capped.where(data["sell_score"] < sell_threshold, 0.0)
                            )
                            rows.append(
                                _metric_row(
                                    data,
                                    branch="factor_library_vol_managed_score",
                                    candidate=(
                                        f"{score_name}_int{intercept}_slope{slope}_tv{target_vol}"
                                        f"_cap{cap}_sell{sell_threshold}"
                                    ),
                                    exposure=pd.Series(exposure, index=index),
                                    baseline_return_pct=baseline_return_pct,
                                    max_drawdown_floor=max_drawdown_floor,
                                    params={
                                        "score": score_name,
                                        "intercept": intercept,
                                        "slope": slope,
                                        "target_vol": target_vol,
                                        "cap": cap,
                                        "sell_threshold": sell_threshold,
                                    },
                                )
                            )

    return pd.DataFrame(rows)


def summarize_factor_library_reversal_branch(
    metrics: pd.DataFrame,
    *,
    branch: str,
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    rows = metrics.loc[metrics["branch"] == branch].copy()
    if rows.empty:
        raise ValueError(f"factor-library branch has no candidate rows: {branch}")
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


def compact_factor_library_search_summary(
    metrics: pd.DataFrame,
    *,
    branches: Iterable[str],
    baseline_return_pct: float,
    max_drawdown_floor: float,
) -> dict[str, Any]:
    branch_results = {
        branch: summarize_factor_library_reversal_branch(
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
