from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.backtest import evaluate_ml_position_candidates


HIGH_CHOP_ML_SOURCE_FIELDS = [
    "SOXL_OHLCV",
    "SOXX_OHLCV",
    "VIX",
    "HIGH_CHOP_PRESSURE",
    "CHOP_BREAKDOWN_RISK",
    "CHOP_REENTRY_QUALITY",
    "SOXX_LEAD_DECAY",
    "PANIC_REVERSAL",
    "VOLUME_CLIMAX",
]

DEFAULT_HIGH_CHOP_FEATURES = [
    "soxl_ret3",
    "soxl_ret5",
    "soxl_ret10",
    "soxl_ret20",
    "soxl_ret60",
    "soxx_ret3",
    "soxx_ret5",
    "soxx_ret10",
    "soxx_ret20",
    "soxx_ret60",
    "vix",
    "vix_chg5",
    "rv20",
    "rv60",
    "rv20_expansion",
    "range_pos_252",
    "dist_high_252",
    "trend_efficiency_20",
    "autocorr10",
    "autocorr20",
    "volume_climax_z20",
    "high_chop_flag",
    "post_high_chop_20d",
    "chop_breakdown_risk",
    "high_chop_pressure",
    "chop_reentry_quality",
    "soxx_lead_10",
    "levered_decay_20",
    "panic_reversal_5",
    "base_chop_control",
    "base_recovery",
    "base_canary",
    "base_constrained_ref",
]


def _wide_qlib_ohlcv_panel(frame: pd.DataFrame, *, symbols: Iterable[str]) -> pd.DataFrame:
    normalized = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"instrument", "datetime", "$close"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"high-chop ML qlib frame missing columns: {missing}")
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"]).dt.normalize()

    pieces: list[pd.DataFrame] = []
    for symbol in [item.upper() for item in symbols]:
        symbol_frame = normalized.loc[normalized["instrument"] == symbol].copy()
        if symbol_frame.empty:
            raise ValueError(f"high-chop ML qlib frame missing instrument: {symbol}")
        columns = {"$close": f"{symbol.lower()}_close"}
        for source in ["$open", "$high", "$low", "$volume", "$vix"]:
            if source in symbol_frame.columns:
                columns[source] = f"{symbol.lower()}_{source[1:]}"
        piece = (
            symbol_frame[["datetime", *columns]]
            .rename(columns=columns)
            .groupby("datetime", as_index=False)
            .last()
        )
        pieces.append(piece)

    panel = pieces[0]
    for piece in pieces[1:]:
        panel = panel.merge(piece, on="datetime", how="inner")
    panel = panel.sort_values("datetime").set_index("datetime")
    if "soxl_vix" in panel.columns:
        panel["vix"] = panel["soxl_vix"].ffill().bfill()
    else:
        panel["vix"] = np.nan
    return panel


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=min(window, min_periods))
    return (series - rolling.mean()) / (rolling.std() + 1e-12)


def build_high_chop_ml_frame(
    qlib_frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    label_horizon: int = 5,
    include_incomplete_latest: bool = False,
) -> pd.DataFrame:
    """Build high-level SOXL high-chop factors for LGB/XGB position models.

    The frame keeps qlib alignment at the input boundary but returns a model-ready
    daily table with forward return labels, regime flags, and base position
    priors used by ML overlays.
    """

    if label_horizon <= 0:
        raise ValueError("label_horizon must be positive")
    panel = _wide_qlib_ohlcv_panel(qlib_frame, symbols=["SOXL", "SOXX"])
    if "soxl_open" not in panel.columns:
        raise ValueError("high-chop ML qlib frame missing SOXL $open for open-to-open returns")
    open_ = panel["soxl_open"].astype("float64")
    close = panel["soxl_close"].astype("float64")
    soxx_close = panel["soxx_close"].astype("float64")
    volume = panel.get("soxl_volume", pd.Series(np.nan, index=panel.index)).astype("float64")
    vix = panel["vix"].astype("float64")
    daily_return = close.pct_change().fillna(0.0)
    soxx_return = soxx_close.pct_change().fillna(0.0)

    data = pd.DataFrame({"datetime": panel.index}, index=panel.index)
    data["forward_return"] = close.shift(-1) / close - 1.0
    data[f"label_return_{label_horizon}d"] = close.shift(-label_horizon) / close - 1.0
    data["open_to_open_return"] = open_.shift(-2) / open_.shift(-1) - 1.0
    data[f"label_open_to_open_{label_horizon}d"] = (
        open_.shift(-(label_horizon + 1)) / open_.shift(-1) - 1.0
    )

    for window in [3, 5, 10, 20, 60, 126, 252]:
        data[f"soxl_ret{window}"] = close / close.shift(window) - 1.0
        data[f"soxx_ret{window}"] = soxx_close / soxx_close.shift(window) - 1.0
    data["vix"] = vix
    data["vix_chg5"] = vix.pct_change(5)
    data["rv20"] = daily_return.rolling(20, min_periods=20).std() * np.sqrt(252)
    data["rv60"] = daily_return.rolling(60, min_periods=30).std() * np.sqrt(252)
    data["rv20_expansion"] = data["rv20"] / (data["rv20"].rolling(252, min_periods=126).median() + 1e-12)
    rolling_high = close.rolling(252, min_periods=100).max()
    rolling_low = close.rolling(252, min_periods=100).min()
    data["range_pos_252"] = (close - rolling_low) / (rolling_high - rolling_low)
    data["dist_high_252"] = close / rolling_high - 1.0
    data["trend_efficiency_20"] = data["soxl_ret20"].abs() / (
        daily_return.abs().rolling(20, min_periods=20).sum() + 1e-12
    )
    data["autocorr20"] = daily_return.rolling(20, min_periods=12).apply(
        lambda values: pd.Series(values).autocorr(1),
        raw=False,
    )
    data["autocorr10"] = daily_return.rolling(10, min_periods=10).corr(daily_return.shift(1))
    data["volume_climax_z20"] = _rolling_zscore(volume, 20, 10)

    rv20_q60 = data["rv20"].expanding(min_periods=120).quantile(0.60)
    high_chop = (
        (close >= 0.80 * rolling_high)
        & (data["range_pos_252"] >= 0.70)
        & (data["trend_efficiency_20"] <= 0.35)
        & (data["rv20"] >= rv20_q60)
    ).fillna(False)
    post_high_chop = high_chop.rolling(20, min_periods=1).max().astype(bool)
    breakdown = (
        post_high_chop
        & (
            (data["soxl_ret10"] < -0.08)
            | (close < close.rolling(10, min_periods=5).mean())
        )
    ).fillna(False)

    data["high_chop_flag"] = high_chop.astype("float64")
    data["post_high_chop_20d"] = post_high_chop.astype("float64")
    data["chop_breakdown_risk"] = breakdown.astype("float64")
    data["high_chop_pressure"] = (
        data["range_pos_252"].fillna(0.0)
        * data["rv20_expansion"].clip(lower=0.0).fillna(0.0)
        * (1.0 - data["trend_efficiency_20"].clip(lower=0.0, upper=1.0).fillna(1.0))
    )
    data["soxx_lead_10"] = data["soxx_ret10"] - data["soxl_ret10"] / 3.0
    data["levered_decay_20"] = data["soxl_ret20"] - 3.0 * data["soxx_ret20"]
    data["panic_reversal_5"] = (
        (-data["soxl_ret20"].clip(upper=0.0))
        * data["soxl_ret5"].clip(lower=0.0)
        * (data["soxx_ret5"] > 0.0).astype("float64")
    )
    data["chop_reentry_quality"] = (
        data["soxl_ret3"].clip(lower=0.0)
        + data["soxx_ret5"].clip(lower=0.0)
        + data["panic_reversal_5"].fillna(0.0)
        - data["vix_chg5"].clip(lower=0.0).fillna(0.0)
    )

    low_hv = data["rv20"] < data["rv20"].rolling(252, min_periods=126).quantile(0.70)
    canary = (
        (data["soxx_ret60"] > 0.06)
        & (data["vix"] < 26.0)
        & (data["autocorr10"] > 0.0)
    ).astype("float64")
    canary_held = canary.rolling(3, min_periods=1).max()
    tsmom = ((data["soxx_ret10"] > 0.0) & low_hv).astype("float64")
    tsmom252 = ((data["soxx_ret252"] > 0.15) & (data["vix"] < 30.0)).astype("float64")
    recovery = (
        ((data["soxl_ret10"] > 0.12) & (data["soxx_ret10"] > 0.02))
        | ((data["soxl_ret20"] < -0.20) & (data["soxl_ret3"] > 0.03) & (data["vix"] < 45.0))
        | ((data["post_high_chop_20d"] > 0.0) & (data["chop_reentry_quality"] > 0.04))
    ).astype("float64")
    constrained_ref = (2.894 * canary_held + 0.0503 * tsmom + 0.0005 * tsmom252).clip(0.0, 1.0)
    base_raw = (1.8 * canary_held + 0.15 * tsmom + 0.85 * recovery).clip(0.0, 1.0)
    cap = pd.Series(1.0, index=panel.index)
    cap.loc[post_high_chop] = 0.45
    cap.loc[high_chop] = 0.55
    cap.loc[breakdown] = 0.10
    data["base_chop_control"] = base_raw.clip(upper=cap.to_numpy(dtype="float64"))
    data["base_chop_control"] = data["base_chop_control"].where(
        ~((data["high_chop_flag"] > 0.0) & (base_raw > 0.0) & (data["chop_breakdown_risk"] <= 0.0)),
        data["base_chop_control"].clip(lower=0.04),
    )
    data["base_recovery"] = (0.7 * recovery + 0.3 * canary).clip(0.0, 1.0)
    data["base_canary"] = base_raw
    data["base_constrained_ref"] = constrained_ref

    result = data.loc[(data["datetime"] >= pd.Timestamp(start)) & (data["datetime"] <= pd.Timestamp(end))].copy()
    if include_incomplete_latest and not result.empty:
        latest_index = result["datetime"].idxmax()
        result.loc[latest_index, "forward_return"] = result.loc[latest_index, "forward_return"] if pd.notna(
            result.loc[latest_index, "forward_return"]
        ) else 0.0
        result.loc[latest_index, "open_to_open_return"] = (
            result.loc[latest_index, "open_to_open_return"]
            if pd.notna(result.loc[latest_index, "open_to_open_return"])
            else 0.0
        )
    required = ["forward_return", "rv20", "range_pos_252", "base_chop_control"]
    if not include_incomplete_latest:
        required.append(f"label_return_{label_horizon}d")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    result.attrs["source_fields"] = HIGH_CHOP_ML_SOURCE_FIELDS
    result.attrs["feature_cols"] = DEFAULT_HIGH_CHOP_FEATURES
    result.attrs["source_coverage_start"] = panel.index.min().date().isoformat()
    result.attrs["source_coverage_end"] = panel.index.max().date().isoformat()
    return result


def _slice_metrics(group: pd.DataFrame, mask_col: str) -> dict[str, Any]:
    sliced = group.loc[group[mask_col].fillna(False)]
    if sliced.empty:
        return {
            f"{mask_col}_days": 0,
            f"{mask_col}_return_pct": 0.0,
            f"{mask_col}_max_drawdown_pct": 0.0,
            f"{mask_col}_avg_abs_position": 0.0,
            f"{mask_col}_trade_count": 0,
        }
    returns = sliced["strategy_return"].astype("float64").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        f"{mask_col}_days": int(len(sliced)),
        f"{mask_col}_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        f"{mask_col}_max_drawdown_pct": float(drawdown.min() * 100.0),
        f"{mask_col}_avg_abs_position": float(sliced["target_position"].abs().mean()),
        f"{mask_col}_trade_count": int((sliced["turnover"].abs() > 1e-9).sum()),
    }


def append_high_chop_strategy_metrics(
    metrics: pd.DataFrame,
    positions: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    baseline_return_pct: float,
    max_drawdown_floor: float = -20.0,
    min_high_chop_avg_abs_position: float = 0.03,
    min_high_chop_trades: int = 1,
    min_post_chop_trades: int = 2,
) -> pd.DataFrame:
    """Attach high-chop slice metrics and goal gates to ML candidate metrics."""

    if metrics.empty:
        raise ValueError("metrics are empty")
    if positions.empty:
        raise ValueError("positions are empty")
    regime = frame[["datetime", "high_chop_flag", "post_high_chop_20d"]].copy()
    regime["datetime"] = pd.to_datetime(regime["datetime"]).dt.normalize()
    regime["high_chop"] = regime["high_chop_flag"].astype(float) > 0.0
    regime["post_high_chop"] = regime["post_high_chop_20d"].astype(float) > 0.0
    regime = regime[["datetime", "high_chop", "post_high_chop"]]

    rows: list[dict[str, Any]] = []
    joined = positions.copy()
    joined["datetime"] = pd.to_datetime(joined["datetime"]).dt.normalize()
    joined = joined.merge(regime, on="datetime", how="left")
    for (candidate, segment), group in joined.groupby(["candidate", "segment"], sort=False):
        row = {"candidate": str(candidate), "segment": str(segment)}
        row.update(_slice_metrics(group, "high_chop"))
        row.update(_slice_metrics(group, "post_high_chop"))
        rows.append(row)

    enriched = metrics.merge(pd.DataFrame(rows), on=["candidate", "segment"], how="left")
    for column in [
        "high_chop_days",
        "high_chop_trade_count",
        "post_high_chop_days",
        "post_high_chop_trade_count",
    ]:
        enriched[column] = enriched[column].fillna(0).astype(int)
    for column in [
        "high_chop_return_pct",
        "high_chop_max_drawdown_pct",
        "high_chop_avg_abs_position",
        "post_high_chop_return_pct",
        "post_high_chop_max_drawdown_pct",
        "post_high_chop_avg_abs_position",
    ]:
        enriched[column] = enriched[column].fillna(0.0).astype("float64")
    enriched["beats_ytd_baseline"] = enriched["return_pct"].astype("float64") > float(baseline_return_pct)
    enriched["passes_drawdown_gate"] = enriched["max_drawdown_pct"].astype("float64") >= float(max_drawdown_floor)
    enriched["passes_high_chop_trade_gate"] = (
        (enriched["high_chop_avg_abs_position"] >= float(min_high_chop_avg_abs_position))
        & (enriched["high_chop_trade_count"] >= int(min_high_chop_trades))
        & (enriched["post_high_chop_trade_count"] >= int(min_post_chop_trades))
    )
    enriched["passes_goal_gate"] = (
        enriched["beats_ytd_baseline"]
        & enriched["passes_drawdown_gate"]
        & enriched["passes_high_chop_trade_gate"]
    )
    for column in ["beats_ytd_baseline", "passes_drawdown_gate", "passes_high_chop_trade_gate", "passes_goal_gate"]:
        enriched[column] = enriched[column].map(bool).astype(object)
    return enriched


def evaluate_high_chop_ml_candidates(
    frame: pd.DataFrame,
    *,
    model_specs: list[dict[str, Any]],
    train_window: tuple[str, str],
    validation_window: tuple[str, str],
    test_window: tuple[str, str],
    position_specs: list[dict[str, Any]],
    label_col: str = "label_return_5d",
    return_col: str = "forward_return",
    feature_cols: list[str] | None = None,
    baseline_return_pct: float | None = None,
    max_drawdown_floor: float = -20.0,
    cost_per_turnover: float = 0.0003,
    selection_score_tolerance: float = 0.0,
    selection_tiebreakers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate LGB/XGB high-chop position models and append regime gates."""

    feature_cols = feature_cols or DEFAULT_HIGH_CHOP_FEATURES
    if baseline_return_pct is None:
        test_mask = (frame["datetime"] >= pd.Timestamp(test_window[0])) & (frame["datetime"] <= pd.Timestamp(test_window[1]))
        baseline_return_pct = float(((1.0 + frame.loc[test_mask, return_col].fillna(0.0)).prod() - 1.0) * 100.0)
    metrics, positions = evaluate_ml_position_candidates(
        frame,
        feature_cols=feature_cols,
        model_specs=model_specs,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        position_specs=position_specs,
        return_col=return_col,
        label_col=label_col,
        cost_per_turnover=cost_per_turnover,
        min_validation_trade_count=1,
        min_validation_avg_abs_position=0.03,
        validation_max_drawdown_floor=-35.0,
        max_validation_reversal_ratio=0.75,
        selection_score_tolerance=selection_score_tolerance,
        selection_tiebreakers=selection_tiebreakers,
    )
    metrics = append_high_chop_strategy_metrics(
        metrics,
        positions,
        frame,
        baseline_return_pct=baseline_return_pct,
        max_drawdown_floor=max_drawdown_floor,
    )
    return metrics, positions
