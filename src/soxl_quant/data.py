from __future__ import annotations

"""SOXL/SOXX/VIX 特征工程模块。

设计上有意把特征分成两类：
  1. qlib-native 的滚动特征：直接从二进制存储中读取（$close, Mean($close,21) 等）
  2. pandas 手工构建的跨序列/体制特征：更容易维护和解释

这样数据源保持透明，避免把跨品种逻辑藏在晦涩的 qlib expression 里。

【泛化优化提示】
  - 当前 hardcode 了 SOXL/SOXX 的列名映射，已通过参数化支持任意 ETF。
  - 后续做多资产训练时，建议增加 instrument_id 和品种波动率画像特征。
"""

import os
from pathlib import Path
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader

from .settings import TEST_SEG, TRAIN_SEG, VALID_SEG

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = ROOT / "data" / "external"


def _vix_term_feature_flag() -> bool:
    """Enable VIX term-structure + SKEW + VVIX features.

    Gated behind an env var so existing actor bundles (trained with 59-feature
    schema) stay replay-compatible by default. New profiles (v22+) should set
    `SOXL_USE_VIX_TERM=1` in their training entrypoint before importing data.
    """
    return os.environ.get("SOXL_USE_VIX_TERM", "0") == "1"


VIX_TERM_FEATURE_COLUMNS: List[str] = [
    "vvix_zscore_252",
    "vix_term_slope",
    "vix_short_ratio",
    "skew_zscore_252",
    "vix_term_slope_velocity_5",
]

FEATURE_COLUMNS: List[str] = [
    "return",
    "anchor_gap_180",
    "anchor_slope_180",
    "trend_gap_21",
    "trend_gap_63",
    "trend_slope_21",
    "trend_slope_63",
    "rsi_14",
    "mom_21",
    "mom_63",
    "realized_vol_10",
    "realized_vol_21",
    "realized_vol_63",
    "vol_of_vol_21",
    "intraday_range",
    "volume_shock",
    "drawdown_21",
    "drawdown_63",
    "cum_drop_20",
    "cum_drop_30",
    "days_since_high_63",
    "drawdown_streak_63",
    "below_ma63_days",
    "below_anchor_ratio_63",
    "new_low_ratio_21",
    "rolling_new_low_ratio_63",
    "loss_streak_21",
    "negative_gap_frequency_21",
    "downside_ratio_10",
    "downside_ratio_21",
    "downside_semivol_21",
    "vol_coef_21",
    "vix_level",
    "vix_zscore",
    "vix_percentile_252",
    "vix_momentum_10",
    "tracking_error_5",
    "tracking_error_21",
    "tracking_error_vol_21",
    "decay_63",
    "relative_strength_21",
    "regime_risk",
    # VIX-free extensions (arXiv-inspired stress proxies)
    "rsi_6",
    "realized_kurtosis_21",
    "jump_variation_5",
    "vol_stress_score",
    "regime_risk_no_vix",
    "relative_volume_zscore",
    # Temporal differential / momentum velocity features (v3)
    "mom_5",
    "mom_10",
    "anchor_gap_velocity_10",
    "trend_gap_21_velocity",
    "rsi_14_velocity_5",
    "regime_risk_velocity_5",
    "volume_shock_velocity_5",
    "expectancy_proxy_21",
    "sharpe_proxy_21",
    "realized_skew_21",
    "gain_streak_10",
]
if _vix_term_feature_flag():
    FEATURE_COLUMNS.extend(VIX_TERM_FEATURE_COLUMNS)
STATIC_INDEX = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}


def feature_schema_signature() -> str:
    payload = "|".join(FEATURE_COLUMNS).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _write_qlib_bin(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.hstack([[0.0], values.astype(np.float32)]).astype("<f4")
    with path.open("wb") as f:
        binary.tofile(f)


def refresh_qlib_binary(
    csv_path: Path,
    qlib_dir: Path,
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
    leverage_ratio: float = 3.0,
) -> pd.DataFrame:
    """刷新最小化的 qlib 二进制存储。

    这是整个数据管道的入口：读取 CSV → 生成 calendar/instruments/features → 写入 qlib_dir。
    生成的目录结构满足 qlib 的 FileStorage 要求，训练、回测、运行时都从这里读数。
    
    【参数说明】
      leveraged_etf  : 杠杆 ETF 代码（如 SOXL、TQQQ）
      underlying_etf : 底层指数 ETF 代码（如 SOXX、QQQ）
      leverage_ratio : 杠杆倍数（默认 3.0），用于后续 tracking_error 计算
    
    【优化提示】当前每次调用都会重写整个 qlib_dir。若数据量变大，可改为增量更新。
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").set_index("date")
    qlib_dir.mkdir(parents=True, exist_ok=True)
    (qlib_dir / "calendars").mkdir(exist_ok=True)
    (qlib_dir / "instruments").mkdir(exist_ok=True)
    calendar = df.index.strftime("%Y-%m-%d").tolist()
    (qlib_dir / "calendars/day.txt").write_text("\n".join(calendar), encoding="utf-8")

    start = calendar[0]
    end = calendar[-1]
    instruments = f"{leveraged_etf}\t{start}\t{end}\n{underlying_etf}\t{start}\t{end}\n"
    (qlib_dir / "instruments/all.txt").write_text(instruments, encoding="utf-8")

    lev_lower = leveraged_etf.lower()
    und_lower = underlying_etf.lower()
    for inst, mapping in {
        leveraged_etf: {
            "open": f"{lev_lower}_open",
            "high": f"{lev_lower}_high",
            "low": f"{lev_lower}_low",
            "close": f"{lev_lower}_close",
            "volume": f"{lev_lower}_volume",
            "vix": "vix",
        },
        underlying_etf: {
            "open": f"{und_lower}_open",
            "high": f"{und_lower}_high",
            "low": f"{und_lower}_low",
            "close": f"{und_lower}_close",
            "volume": f"{und_lower}_volume",
        },
    }.items():
        for field_name, col_name in mapping.items():
            _write_qlib_bin(qlib_dir / f"features/{inst}/{field_name}.day.bin", df[col_name].to_numpy())

    (qlib_dir / "qlib.yaml").write_text(
        "\n".join(
            [
                "# Qlib configuration",
                f"provider_uri: {qlib_dir}",
                "redis:",
                "  host: null",
                "  port: null",
                "market:",
                "  market: all",
                "  market_type: stock",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return df


def init_qlib(provider_uri: Path) -> None:
    try:
        qlib.init(provider_uri=str(provider_uri), region="us", expression_cache=None, dataset_cache=None)
    except Exception:
        qlib.init(provider_uri=str(provider_uri))


def build_market_frame(
    start_time: str,
    end_time: str,
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 qlib 存储中读取指定时间窗口的原始行情数据。"""
    lev_fields = [
        "$close",
        "$open",
        "$high",
        "$low",
        "$volume",
        "$vix",
        "Mean($close, 5)",
        "Mean($close, 21)",
        "Mean($close, 63)",
        "Mean($close, 180)",
        "Ref($close, 21)",
        "Ref($close, 63)",
        "Std($close / Ref($close, 1) - 1, 10)",
        "Std($close / Ref($close, 1) - 1, 21)",
        "Std($close / Ref($close, 1) - 1, 63)",
        "Mean($volume, 21)",
    ]
    und_fields = [
        "$close",
        "$open",
        "$high",
        "$low",
        "$volume",
        "Ref($close, 5)",
        "Ref($close, 21)",
    ]
    lev = D.features([leveraged_etf], lev_fields, start_time=start_time, end_time=end_time).xs(leveraged_etf, level="instrument")
    und = D.features([underlying_etf], und_fields, start_time=start_time, end_time=end_time).xs(underlying_etf, level="instrument")
    return lev, und


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def _streaks(mask: pd.Series) -> pd.Series:
    streak = 0
    out = []
    for flag in mask.fillna(False):
        streak = streak + 1 if bool(flag) else 0
        out.append(streak)
    return pd.Series(out, index=mask.index, dtype=float)


def _load_external_series(filename: str, value_col: str) -> pd.Series:
    """Load an external CSV from ``data/external/`` into a date-indexed Series.

    Handles the two layouts shipping with this repo:
      - DATE, VALUE
      - DATE, OPEN, HIGH, LOW, CLOSE  (we use CLOSE)
    """
    path = EXTERNAL_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing external series: {path}")
    df = pd.read_csv(path)
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if value_col not in df.columns:
        raise KeyError(f"{filename} missing column {value_col}; got {list(df.columns)}")
    series = df[value_col].astype(float)
    series.index.name = "date"
    return series


def _align_external(series: pd.Series, index: pd.Index) -> pd.Series:
    """Reindex an external series onto the feature frame's trading calendar.

    Forward-fills gaps (e.g. SKEW publishes daily, but reindexing to the ETF
    calendar may introduce misalignment). Leading NaNs before external-data
    coverage are filled with 0.0 so the feature is simply silent for that
    period instead of dropping rows.
    """
    aligned = series.reindex(index, method="ffill")
    return aligned.fillna(0.0)


def engineer_feature_frame(
    lev: pd.DataFrame,
    und: pd.DataFrame,
    leverage_ratio: float = 3.0,
    leveraged_etf: str = "SOXL",
    ablate_vix: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """构建体制感知（regime-aware）特征矩阵，供训练、回测、运行时使用。

    特征设计围绕 LETF（杠杆 ETF）的路径依赖性：
      - 长期趋势锚定状态（anchor_gap_180, anchor_slope_180）
      - 持续回撤与下行压力（drawdown_63, downside_ratio_21, loss_streak_21）
      - LETF vs 底层指数的 tracking error 和 decay（tracking_error_21, decay_63）
      - VIX 驱动的压力环境（vix_level, vix_zscore, vix_percentile_252）
    
    【优化提示】
      1. 当前 tracking_error 假设 leverage_ratio 固定（如 3.0）。
         若未来测试 2x ETF（如 TQQQ 的同类 SSO），需确保传入正确杠杆倍数。
      2. 做多资产训练时，建议在此加入品种画像特征：
         - median_vol_252（该 ETF 的年化波动中位数）
         - beta_to_spy（对 SPY 的 63 天滚动 beta）
         - sector_id（半导体/科技/宽基）
    """
    close = lev["$close"].astype(float)
    high = lev["$high"].astype(float)
    low = lev["$low"].astype(float)
    volume = lev["$volume"].astype(float)
    vix = lev["$vix"].astype(float)
    und_close = und["$close"].astype(float)

    lev_ret = close.pct_change()
    und_ret = und_close.pct_change()

    ma5 = lev["Mean($close, 5)"].astype(float)
    ma21 = lev["Mean($close, 21)"].astype(float)
    ma63 = lev["Mean($close, 63)"].astype(float)
    ma180 = lev["Mean($close, 180)"].astype(float)
    volume_ma21 = lev["Mean($volume, 21)"].astype(float)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean().clip(lower=1e-8)
    rsi = 100 - 100 / (1 + gain / loss)

    realized_vol_10 = lev["Std($close / Ref($close, 1) - 1, 10)"].astype(float) * np.sqrt(252)
    realized_vol_21 = lev["Std($close / Ref($close, 1) - 1, 21)"].astype(float) * np.sqrt(252)
    realized_vol_63 = lev["Std($close / Ref($close, 1) - 1, 63)"].astype(float) * np.sqrt(252)
    vol_of_vol_21 = realized_vol_21.rolling(21).std()

    rolling_high_21 = close.rolling(21).max()
    rolling_high_63 = close.rolling(63).max()
    drawdown_21 = close / rolling_high_21 - 1.0
    drawdown_63 = close / rolling_high_63 - 1.0
    cum_drop_20 = close / close.rolling(20).max() - 1.0
    cum_drop_30 = close / close.rolling(30).max() - 1.0

    days_since_high_63 = close.rolling(63).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True) / 63.0
    drawdown_streak_63 = _streaks(drawdown_63 < 0).clip(upper=63) / 63.0
    below_ma63_days = _streaks(close < ma63).clip(upper=63) / 63.0
    below_anchor_ratio_63 = (close < ma180).rolling(63).mean()
    new_low_ratio_21 = (close <= close.rolling(21).min()).rolling(21).mean()
    rolling_new_low_ratio_63 = (close <= close.rolling(63).min()).rolling(63).mean()
    prev_close = close.shift(1)
    negative_gap_frequency_21 = ((lev["$open"].astype(float) / prev_close - 1.0) < -0.01).rolling(21).mean()
    downside_ratio_10 = (lev_ret < 0).rolling(10).mean()
    downside_ratio_21 = (lev_ret < 0).rolling(21).mean()
    downside_semivol_21 = lev_ret.where(lev_ret < 0, 0.0).pow(2).rolling(21).mean().pow(0.5) * np.sqrt(252)
    loss_streak_21 = _streaks(lev_ret < 0).clip(upper=21) / 21.0
    vol_coef_21 = realized_vol_21 / (realized_vol_21.rolling(252).median() + 1e-8)

    vix_zscore = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
    vix_percentile_252 = _rolling_percentile(vix, 252)
    vix_momentum_10 = vix.pct_change(10)

    tracking_error = lev_ret - leverage_ratio * und_ret
    tracking_error_5 = tracking_error.rolling(5).mean()
    tracking_error_21 = tracking_error.rolling(21).mean()
    tracking_error_vol_21 = tracking_error.rolling(21).std()
    decay_63 = (lev_ret.rolling(63).sum() - leverage_ratio * und_ret.rolling(63).sum())
    relative_strength_21 = lev_ret.rolling(21).sum() - und_ret.rolling(21).sum()

    trend_gap_21 = ma5 / ma21 - 1.0
    trend_gap_63 = ma21 / ma63 - 1.0
    anchor_gap_180 = close / ma180 - 1.0
    trend_slope_21 = ma21 / ma21.shift(10) - 1.0
    trend_slope_63 = ma63 / ma63.shift(21) - 1.0
    anchor_slope_180 = ma180 / ma180.shift(21) - 1.0
    intraday_range = (high - low) / close
    volume_shock = volume / volume_ma21 - 1.0
    mom_21 = lev_ret.rolling(21).sum()
    mom_63 = lev_ret.rolling(63).sum()

    # --- VIX-free stress proxies (ported from soxl_backtest/ + arXiv research) ---
    # RSI6 fast-reversion gauge
    delta6 = close.diff()
    gain6 = delta6.where(delta6 > 0, 0.0).ewm(span=6, adjust=False).mean()
    loss6 = (-delta6.where(delta6 < 0, 0.0)).ewm(span=6, adjust=False).mean().clip(lower=1e-8)
    rsi_6 = 100 - 100 / (1 + gain6 / loss6)

    # Realized kurtosis (tail-risk proxy, arXiv 2510.03236)
    realized_kurtosis_21 = lev_ret.rolling(21).kurt().fillna(0.0)

    # Jump variation: sum of squared returns on days where |return| > 3 * rolling_std
    rolling_std_21 = lev_ret.rolling(21).std().fillna(lev_ret.std())
    jump_mask = abs(lev_ret) > 3 * rolling_std_21
    jump_variation_5 = (lev_ret.where(jump_mask, 0.0).pow(2)).rolling(5).sum()

    # Relative volume z-score (EnhancedPanicCalculator volume_multiplier concept)
    avg_volume_20 = volume.rolling(20).mean()
    relative_volume_zscore = (volume - avg_volume_20) / (avg_volume_20.rolling(20).std() + 1e-8)

    # Vol-stress score: composite price-only stress index
    rv10_zscore = (realized_vol_10 - realized_vol_10.rolling(252).mean()) / (realized_vol_10.rolling(252).std() + 1e-8)
    dd63_zscore = (drawdown_63 - drawdown_63.rolling(252).mean()) / (drawdown_63.rolling(252).std() + 1e-8)
    ir_zscore = (intraday_range - intraday_range.rolling(252).mean()) / (intraday_range.rolling(252).std() + 1e-8)
    vol_stress_score = (
        0.30 * rv10_zscore.fillna(0.0)
        + 0.25 * dd63_zscore.fillna(0.0)
        + 0.20 * ir_zscore.fillna(0.0)
        + 0.15 * volume_shock.fillna(0.0)
        + 0.10 * realized_kurtosis_21.rank(pct=True).fillna(0.0)
    )
    # Normalize to [0, 1] percentile-like scale so it can slot into regime_risk weights
    vol_stress_score = vol_stress_score.rank(pct=True).fillna(0.0)

    # regime_risk without VIX (ablation feature)
    regime_risk_no_vix = (
        0.22 * vol_stress_score
        + 0.18 * realized_vol_21.rank(pct=True).fillna(0.0)
        + 0.18 * drawdown_63.abs().rank(pct=True).fillna(0.0)
        + 0.15 * downside_ratio_21.fillna(0.0)
        + 0.12 * downside_semivol_21.rank(pct=True).fillna(0.0)
        + 0.08 * tracking_error_vol_21.rank(pct=True).fillna(0.0)
        + 0.07 * loss_streak_21.fillna(0.0)
    )

    regime_risk = (
        0.22 * vix_percentile_252.fillna(0.0)
        + 0.18 * realized_vol_21.rank(pct=True).fillna(0.0)
        + 0.18 * drawdown_63.abs().rank(pct=True).fillna(0.0)
        + 0.15 * downside_ratio_21.fillna(0.0)
        + 0.12 * downside_semivol_21.rank(pct=True).fillna(0.0)
        + 0.08 * tracking_error_vol_21.rank(pct=True).fillna(0.0)
        + 0.07 * loss_streak_21.fillna(0.0)
    )

    # --- Temporal differential / momentum velocity features (v3) ---
    # Short-term momentum for faster regime-transition detection
    mom_5 = lev_ret.rolling(5).sum()
    mom_10 = lev_ret.rolling(10).sum()

    # Velocity features: rate-of-change in key regime indicators
    anchor_gap_velocity_10 = anchor_gap_180 - anchor_gap_180.shift(10)
    trend_gap_21_velocity = trend_gap_21 - trend_gap_21.shift(5)
    rsi_14_velocity_5 = (rsi / 100.0).diff(5)
    regime_risk_velocity_5 = regime_risk.diff(5)
    volume_shock_velocity_5 = volume_shock - volume_shock.shift(5)

    # Expectancy proxy: forward-statistical edge from recent history
    # = (up_ratio * avg_up_return + down_ratio * avg_down_return) / avg_abs_return
    up_mask = lev_ret > 0
    down_mask = lev_ret < 0
    up_days_ratio = up_mask.rolling(21).mean()
    down_days_ratio = down_mask.rolling(21).mean()
    avg_up_return = lev_ret.where(up_mask, 0.0).rolling(21).mean()
    avg_down_return = lev_ret.where(down_mask, 0.0).rolling(21).mean()
    avg_abs_return = lev_ret.abs().rolling(21).mean().clip(lower=1e-8)
    expectancy_proxy_21 = (up_days_ratio * avg_up_return + down_days_ratio * avg_down_return) / avg_abs_return

    # Sharpe proxy: risk-adjusted momentum over 21 days
    sharpe_proxy_21 = lev_ret.rolling(21).mean() / (lev_ret.rolling(21).std().clip(lower=1e-8)) * np.sqrt(252)

    # Realized skewness: tail asymmetry companion to kurtosis
    realized_skew_21 = lev_ret.rolling(21).skew().fillna(0.0)

    # Gain streak: symmetric to loss_streak_21
    gain_streak_10 = _streaks(lev_ret > 0).clip(upper=10) / 10.0

    # --- VIX term-structure / SKEW / VVIX (v22) ---
    # Gated on SOXL_USE_VIX_TERM so existing 59-feature bundles stay replay-safe.
    use_vix_term = _vix_term_feature_flag()
    if use_vix_term:
        idx = close.index
        vvix = _align_external(_load_external_series("VVIX.csv", "VVIX"), idx)
        vix3m = _align_external(_load_external_series("VIX3M.csv", "CLOSE"), idx)
        vix9d = _align_external(_load_external_series("VIX9D.csv", "CLOSE"), idx)
        skew = _align_external(_load_external_series("SKEW.csv", "SKEW"), idx)

        vvix_zscore_252 = (
            (vvix - vvix.rolling(252).mean()) / (vvix.rolling(252).std() + 1e-8)
        ).fillna(0.0)
        vix_term_slope = (vix / vix3m.replace(0.0, np.nan) - 1.0).fillna(0.0)
        vix_short_ratio = (vix9d / vix.replace(0.0, np.nan) - 1.0).fillna(0.0)
        skew_zscore_252 = (
            (skew - skew.rolling(252).mean()) / (skew.rolling(252).std() + 1e-8)
        ).fillna(0.0)
        vix_term_slope_velocity_5 = (vix_term_slope - vix_term_slope.shift(5)).fillna(0.0)
    else:
        vvix_zscore_252 = None
        vix_term_slope = None
        vix_short_ratio = None
        skew_zscore_252 = None
        vix_term_slope_velocity_5 = None

    feature_df = pd.DataFrame(
        {
            "return": lev_ret,
            "anchor_gap_180": anchor_gap_180,
            "anchor_slope_180": anchor_slope_180,
            "trend_gap_21": trend_gap_21,
            "trend_gap_63": trend_gap_63,
            "trend_slope_21": trend_slope_21,
            "trend_slope_63": trend_slope_63,
            "rsi_14": rsi / 100.0,
            "mom_21": mom_21,
            "mom_63": mom_63,
            "realized_vol_10": realized_vol_10,
            "realized_vol_21": realized_vol_21,
            "realized_vol_63": realized_vol_63,
            "vol_of_vol_21": vol_of_vol_21,
            "intraday_range": intraday_range,
            "volume_shock": volume_shock,
            "drawdown_21": drawdown_21,
            "drawdown_63": drawdown_63,
            "cum_drop_20": cum_drop_20,
            "cum_drop_30": cum_drop_30,
            "days_since_high_63": days_since_high_63,
            "drawdown_streak_63": drawdown_streak_63,
            "below_ma63_days": below_ma63_days,
            "below_anchor_ratio_63": below_anchor_ratio_63,
            "new_low_ratio_21": new_low_ratio_21,
            "rolling_new_low_ratio_63": rolling_new_low_ratio_63,
            "loss_streak_21": loss_streak_21,
            "negative_gap_frequency_21": negative_gap_frequency_21,
            "downside_ratio_10": downside_ratio_10,
            "downside_ratio_21": downside_ratio_21,
            "downside_semivol_21": downside_semivol_21,
            "vol_coef_21": vol_coef_21,
            "vix_level": vix / 50.0,
            "vix_zscore": vix_zscore,
            "vix_percentile_252": vix_percentile_252,
            "vix_momentum_10": vix_momentum_10,
            "tracking_error_5": tracking_error_5 * 20.0,
            "tracking_error_21": tracking_error_21 * 20.0,
            "tracking_error_vol_21": tracking_error_vol_21 * 20.0,
            "decay_63": decay_63 * 10.0,
            "relative_strength_21": relative_strength_21 * 10.0,
            "regime_risk": regime_risk,
            "rsi_6": rsi_6 / 100.0,
            "realized_kurtosis_21": realized_kurtosis_21,
            "jump_variation_5": jump_variation_5 * 100.0,
            "vol_stress_score": vol_stress_score,
            "regime_risk_no_vix": regime_risk_no_vix,
            "relative_volume_zscore": relative_volume_zscore,
            # Temporal differential / momentum velocity features (v3)
            "mom_5": mom_5,
            "mom_10": mom_10,
            "anchor_gap_velocity_10": anchor_gap_velocity_10,
            "trend_gap_21_velocity": trend_gap_21_velocity,
            "rsi_14_velocity_5": rsi_14_velocity_5,
            "regime_risk_velocity_5": regime_risk_velocity_5,
            "volume_shock_velocity_5": volume_shock_velocity_5,
            "expectancy_proxy_21": expectancy_proxy_21,
            "sharpe_proxy_21": sharpe_proxy_21,
            "realized_skew_21": realized_skew_21,
            "gain_streak_10": gain_streak_10,
            "close": close,
        }
    ).replace([np.inf, -np.inf], np.nan)
    if use_vix_term:
        feature_df["vvix_zscore_252"] = vvix_zscore_252
        feature_df["vix_term_slope"] = vix_term_slope
        feature_df["vix_short_ratio"] = vix_short_ratio
        feature_df["skew_zscore_252"] = skew_zscore_252
        feature_df["vix_term_slope_velocity_5"] = vix_term_slope_velocity_5
    feature_df = feature_df.dropna()

    if ablate_vix:
        feature_df["vix_level"] = 0.0
        feature_df["vix_zscore"] = 0.0
        feature_df["vix_percentile_252"] = 0.0
        feature_df["vix_momentum_10"] = 0.0
        feature_df["regime_risk"] = feature_df["regime_risk_no_vix"]

    price = feature_df.pop("close")
    feature_df = feature_df[FEATURE_COLUMNS]
    feature_df.index.name = "datetime"
    multi_index = pd.MultiIndex.from_arrays(
        [feature_df.index, np.repeat(leveraged_etf, len(feature_df))],
        names=["datetime", "instrument"],
    )
    feature_df.index = multi_index
    price.index = multi_index
    return feature_df, price


def build_feature_frame(
    start_time: str,
    end_time: str,
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
    leverage_ratio: float = 3.0,
    ablate_vix: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    lev, und = build_market_frame(start_time, end_time, leveraged_etf=leveraged_etf, underlying_etf=underlying_etf)
    return engineer_feature_frame(lev, und, leverage_ratio=leverage_ratio, leveraged_etf=leveraged_etf, ablate_vix=ablate_vix)


class SoxlDataHandler(DataHandlerLP):
    def __init__(
        self,
        start_time: str,
        end_time: str,
        leveraged_etf: str = "SOXL",
        underlying_etf: str = "SOXX",
        leverage_ratio: float = 3.0,
        ablate_vix: bool = False,
    ):
        # Keep both the feature frame and aligned price frame so training,
        # backtest, and runtime all read from the same daily source of truth.
        feature_df, price = build_feature_frame(
            start_time,
            end_time,
            leveraged_etf=leveraged_etf,
            underlying_etf=underlying_etf,
            leverage_ratio=leverage_ratio,
            ablate_vix=ablate_vix,
        )
        self.price_frame = price.to_frame("price")
        self.feature_frame = feature_df
        loader_df = pd.concat({"feature": feature_df}, axis=1)
        super().__init__(
            instruments=[leveraged_etf],
            start_time=start_time,
            end_time=end_time,
            data_loader=StaticDataLoader(loader_df),
            infer_processors=[],
            learn_processors=[],
            shared_processors=[],
            process_type=DataHandlerLP.PTYPE_A,
            drop_raw=False,
        )


def build_dataset(
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
    leverage_ratio: float = 3.0,
    train_seg: Tuple[str, str] = TRAIN_SEG,
    valid_seg: Tuple[str, str] = VALID_SEG,
    test_seg: Tuple[str, str] = TEST_SEG,
    ablate_vix: bool = False,
) -> DatasetH:
    """构建 qlib DatasetH，划分 train / valid / test 三段。

    【数据切分策略】
      train: 2010-03-11 ~ 2018-12-31（约 9 年，策略学习期）
      valid: 2019-01-01 ~ 2019-12-31（1 年，模型选择期）
      test : 2020-01-01 ~ 2026-03-31（6+ 年，最终评估期）
    
    这是仓库的硬性规则，不要直接用 2022 数据做训练，否则会造成数据泄漏。
    """
    return DatasetH(
        handler=SoxlDataHandler(
            start_time=train_seg[0],
            end_time=test_seg[1],
            leveraged_etf=leveraged_etf,
            underlying_etf=underlying_etf,
            leverage_ratio=leverage_ratio,
            ablate_vix=ablate_vix,
        ),
        segments={"train": train_seg, "valid": valid_seg, "test": test_seg},
        fetch_kwargs={"level": "datetime"},
    )


def flatten_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if "feature" in df.columns.get_level_values(0):
            df = df.xs("feature", axis=1, level=0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df.sort_index()


def segment_frame(dataset: DatasetH, segment: str) -> pd.DataFrame:
    df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
    df = flatten_feature_frame(df)
    return df.droplevel("instrument").copy()
