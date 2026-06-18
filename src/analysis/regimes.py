from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


MARKET_REGIME_SCHEMA_VERSION = "analysis.market_regime.v1"


@dataclass(frozen=True)
class MarketRegimeConfig:
    trend_window: int = 21
    volatility_window: int = 21
    drawdown_window: int = 63
    volume_window: int = 20
    high_vix_threshold: float = 30.0
    low_vix_threshold: float = 20.0
    stress_drawdown_threshold: float = -0.10
    bear_drawdown_threshold: float = -0.03
    uptrend_threshold: float = 0.03
    downtrend_threshold: float = -0.03
    sideways_trend_abs_threshold: float = 0.01
    high_volatility_threshold: float = 0.04
    low_volatility_threshold: float = 0.02
    volume_shock_z_threshold: float = 2.0
    source: str = "market_panel"


def _validate_positive_windows(config: MarketRegimeConfig) -> None:
    for name in ["trend_window", "volatility_window", "drawdown_window", "volume_window"]:
        value = int(getattr(config, name))
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _numeric(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.ffill().bfill().fillna(default).astype(float)


def _required_column(frame: pd.DataFrame, candidates: Iterable[str], *, purpose: str) -> str:
    column = _first_column(frame, candidates)
    if column is None:
        raise ValueError(f"market regime frame missing {purpose} column")
    return column


def _existing_or_computed(
    frame: pd.DataFrame,
    *,
    candidates: tuple[str, ...],
    compute: pd.Series,
) -> pd.Series:
    column = _first_column(frame, candidates)
    if column is not None:
        return _numeric(frame, column)
    return compute.astype(float).replace([float("inf"), float("-inf")], pd.NA).ffill().bfill().fillna(0.0)


def _rolling_z(values: pd.Series, *, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=2).mean()
    std = values.rolling(window, min_periods=2).std().replace(0.0, pd.NA)
    return ((values - mean) / std).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)


def _relative_strength(frame: pd.DataFrame, close: pd.Series, *, window: int) -> pd.Series:
    existing = _first_column(frame, ["relative_strength", "soxl_soxx_relative_5d"])
    if existing is not None:
        return _numeric(frame, existing)

    benchmark_column = _first_column(frame, ["benchmark_close", "soxx_close", "underlying_close"])
    if benchmark_column is None:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)

    benchmark = _numeric(frame, benchmark_column)
    return (close.pct_change(window) - benchmark.pct_change(window)).fillna(0.0)


def _regime_for_row(row: pd.Series, config: MarketRegimeConfig) -> tuple[str, str, str]:
    vix = float(row["vix"])
    trend = float(row["trend"])
    drawdown = float(row["drawdown"])
    realized_vol = float(row["realized_vol"])
    volume_z = float(row["volume_z"])

    tags: list[str] = []
    if vix >= config.high_vix_threshold:
        tags.append("high_vix")
    if vix <= config.low_vix_threshold:
        tags.append("low_vix")
    if drawdown <= config.stress_drawdown_threshold:
        tags.append("deep_drawdown")
    elif drawdown <= config.bear_drawdown_threshold:
        tags.append("drawdown")
    if realized_vol >= config.high_volatility_threshold:
        tags.append("high_volatility")
    elif realized_vol <= config.low_volatility_threshold:
        tags.append("low_volatility")
    if volume_z >= config.volume_shock_z_threshold:
        tags.append("volume_shock")
    if trend >= config.uptrend_threshold:
        tags.append("uptrend")
    elif trend <= config.downtrend_threshold:
        tags.append("downtrend")
    elif abs(trend) <= config.sideways_trend_abs_threshold:
        tags.append("sideways")

    if "high_vix" in tags or "deep_drawdown" in tags:
        return "stress", "|".join(tags), "压力 regime：VIX 高企或深度回撤，优先检查防守表现。"
    if "volume_shock" in tags:
        return "volume_shock", "|".join(tags), "成交量冲击 regime：成交量显著偏离滚动均值，检查换手和滑点。"
    if "high_volatility" in tags and "downtrend" in tags:
        return "high_vol_downtrend", "|".join(tags), "高波动下行 regime：波动和下行趋势同时出现，检查回撤控制。"
    if "high_volatility" in tags and "uptrend" in tags:
        return "high_vol_uptrend", "|".join(tags), "高波动上行 regime：趋势强但波动高，检查追涨和仓位弹性。"
    if "low_vix" in tags and "uptrend" in tags:
        return "bull", "|".join(tags), "低 VIX 上行 regime：风险偏好较高，检查模型是否能捕捉趋势。"
    if "downtrend" in tags or "drawdown" in tags:
        return "bear", "|".join(tags), "下行或回撤 regime：优先检查亏损控制和暴露收缩。"
    if "sideways" in tags and "low_volatility" in tags:
        return "sideways", "|".join(tags), "低波动震荡 regime：检查模型是否过度交易或信号噪声过高。"
    return "neutral", "|".join(tags or ["neutral"]), "中性 regime：没有明显压力、趋势或成交量冲击。"


def build_market_regime_frame(
    frame: pd.DataFrame,
    *,
    config: MarketRegimeConfig | None = None,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a stable date-level market regime frame for qlib/backtest analysis."""

    if frame.empty:
        raise ValueError("market regime frame is empty")
    config = config or MarketRegimeConfig()
    _validate_positive_windows(config)

    date_column = _required_column(frame, ["date", "datetime", "time"], purpose="date")
    close_column = _required_column(frame, ["close", "soxl_close", "price"], purpose="close/price")
    ordered = frame.copy()
    ordered["date"] = pd.to_datetime(ordered[date_column])
    ordered = ordered.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    close = _numeric(ordered, close_column)
    volume_column = _first_column(ordered, ["volume", "soxl_volume"])
    volume = _numeric(ordered, volume_column) if volume_column is not None else pd.Series([0.0] * len(ordered))
    vix_column = _first_column(ordered, ["vix", "vix_level"])
    vix = _numeric(ordered, vix_column) if vix_column is not None else pd.Series([0.0] * len(ordered))

    returns = close.pct_change().fillna(0.0)
    computed_trend = close / close.rolling(config.trend_window, min_periods=1).mean() - 1.0
    computed_drawdown = close / close.rolling(config.drawdown_window, min_periods=1).max() - 1.0
    computed_volatility = returns.rolling(config.volatility_window, min_periods=2).std().fillna(0.0)
    computed_volume_z = _rolling_z(volume, window=config.volume_window)

    result = pd.DataFrame(
        {
            "schema_version": MARKET_REGIME_SCHEMA_VERSION,
            "date": ordered["date"],
            "source": config.source,
            "vix": vix,
            "trend": _existing_or_computed(ordered, candidates=("trend", "trend_21", "trend_5"), compute=computed_trend),
            "drawdown": _existing_or_computed(
                ordered,
                candidates=("drawdown", "drawdown_21"),
                compute=computed_drawdown,
            ),
            "realized_vol": _existing_or_computed(
                ordered,
                candidates=("realized_vol", "realized_vol_21", "realized_vol_5"),
                compute=computed_volatility,
            ),
            "volume_z": _existing_or_computed(ordered, candidates=("volume_z", "volume_z_20"), compute=computed_volume_z),
            "relative_strength": _relative_strength(ordered, close, window=config.trend_window),
        }
    )

    regimes = result.apply(lambda row: _regime_for_row(row, config), axis=1)
    result["regime"] = [item[0] for item in regimes]
    result["regime_tags"] = [item[1] for item in regimes]
    result["regime_reason"] = [item[2] for item in regimes]

    if calendar is not None:
        valid_dates = pd.DatetimeIndex(pd.to_datetime(calendar))
        result = result.loc[result["date"].isin(valid_dates)].copy()

    ordered_columns = [
        "schema_version",
        "date",
        "source",
        "regime",
        "regime_tags",
        "regime_reason",
        "vix",
        "trend",
        "drawdown",
        "realized_vol",
        "volume_z",
        "relative_strength",
    ]
    return result.loc[:, ordered_columns].sort_values("date").reset_index(drop=True)


def attach_market_regimes(
    frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Attach date-level market regimes to a model daily backtest or qlib portfolio report."""

    if "date" not in frame.columns:
        raise ValueError("frame must include a date column")
    if "date" not in regime_frame.columns or "regime" not in regime_frame.columns:
        raise ValueError("regime_frame must include date and regime columns")

    left = frame.copy()
    left["date"] = pd.to_datetime(left["date"])
    context = regime_frame.copy()
    context["date"] = pd.to_datetime(context["date"])
    context = context.drop_duplicates(subset=["date"], keep="last")
    keep_columns = [
        column
        for column in [
            "date",
            "regime",
            "regime_tags",
            "regime_reason",
            "vix",
            "trend",
            "drawdown",
            "realized_vol",
            "volume_z",
            "relative_strength",
        ]
        if column in context.columns
    ]
    context = context.loc[:, keep_columns].rename(columns={"regime": "market_regime"})

    merged = left.merge(context, on="date", how="left")
    if overwrite or "regime" not in merged.columns:
        merged["regime"] = merged["market_regime"]
    return merged
