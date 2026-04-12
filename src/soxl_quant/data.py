from __future__ import annotations

"""SOXL/SOXX/VIX feature engineering for the daily RL harness.

The design intentionally splits features into two groups:

- qlib-native rolling features that can be read directly from the binary store
- pandas-engineered cross-series/regime features that are easier to maintain

This keeps the data source explicit and avoids hiding cross-instrument logic in
opaque expressions.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader

from .settings import TEST_SEG, TRAIN_SEG, VALID_SEG

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
    "below_anchor_ratio_63",
    "new_low_ratio_21",
    "loss_streak_21",
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
]
STATIC_INDEX = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}


def _write_qlib_bin(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.hstack([[0.0], values.astype(np.float32)]).astype("<f4")
    with path.open("wb") as f:
        binary.tofile(f)


def refresh_qlib_binary(csv_path: Path, qlib_dir: Path) -> pd.DataFrame:
    """Refresh the minimal qlib binary store used by the SOXL package."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").set_index("date")
    qlib_dir.mkdir(parents=True, exist_ok=True)
    (qlib_dir / "calendars").mkdir(exist_ok=True)
    (qlib_dir / "instruments").mkdir(exist_ok=True)
    calendar = df.index.strftime("%Y-%m-%d").tolist()
    (qlib_dir / "calendars/day.txt").write_text("\n".join(calendar), encoding="utf-8")

    start = calendar[0]
    end = calendar[-1]
    instruments = f"SOXL\t{start}\t{end}\nSOXX\t{start}\t{end}\n"
    (qlib_dir / "instruments/all.txt").write_text(instruments, encoding="utf-8")

    for inst, mapping in {
        "SOXL": {
            "open": "soxl_open",
            "high": "soxl_high",
            "low": "soxl_low",
            "close": "soxl_close",
            "volume": "soxl_volume",
            "vix": "vix",
        },
        "SOXX": {
            "open": "soxx_open",
            "high": "soxx_high",
            "low": "soxx_low",
            "close": "soxx_close",
            "volume": "soxx_volume",
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


def build_market_frame(start_time: str, end_time: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    soxl_fields = [
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
    soxx_fields = [
        "$close",
        "$open",
        "$high",
        "$low",
        "$volume",
        "Ref($close, 5)",
        "Ref($close, 21)",
    ]
    soxl = D.features(["SOXL"], soxl_fields, start_time=start_time, end_time=end_time).xs("SOXL", level="instrument")
    soxx = D.features(["SOXX"], soxx_fields, start_time=start_time, end_time=end_time).xs("SOXX", level="instrument")
    return soxl, soxx


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def _streaks(mask: pd.Series) -> pd.Series:
    streak = 0
    out = []
    for flag in mask.fillna(False):
        streak = streak + 1 if bool(flag) else 0
        out.append(streak)
    return pd.Series(out, index=mask.index, dtype=float)


def engineer_feature_frame(soxl: pd.DataFrame, soxx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the regime-aware feature frame consumed by training and runtime.

    The feature set emphasizes LETF-specific path dependency:

    - long-trend anchor state
    - persistent drawdown / downside pressure
    - SOXL vs SOXX tracking error and decay
    - VIX-derived stress context
    """
    close = soxl["$close"].astype(float)
    high = soxl["$high"].astype(float)
    low = soxl["$low"].astype(float)
    volume = soxl["$volume"].astype(float)
    vix = soxl["$vix"].astype(float)
    soxx_close = soxx["$close"].astype(float)

    soxl_ret = close.pct_change()
    soxx_ret = soxx_close.pct_change()

    ma5 = soxl["Mean($close, 5)"].astype(float)
    ma21 = soxl["Mean($close, 21)"].astype(float)
    ma63 = soxl["Mean($close, 63)"].astype(float)
    ma180 = soxl["Mean($close, 180)"].astype(float)
    volume_ma21 = soxl["Mean($volume, 21)"].astype(float)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean().clip(lower=1e-8)
    rsi = 100 - 100 / (1 + gain / loss)

    realized_vol_10 = soxl["Std($close / Ref($close, 1) - 1, 10)"].astype(float) * np.sqrt(252)
    realized_vol_21 = soxl["Std($close / Ref($close, 1) - 1, 21)"].astype(float) * np.sqrt(252)
    realized_vol_63 = soxl["Std($close / Ref($close, 1) - 1, 63)"].astype(float) * np.sqrt(252)
    vol_of_vol_21 = realized_vol_21.rolling(21).std()

    rolling_high_21 = close.rolling(21).max()
    rolling_high_63 = close.rolling(63).max()
    drawdown_21 = close / rolling_high_21 - 1.0
    drawdown_63 = close / rolling_high_63 - 1.0
    cum_drop_20 = close / close.rolling(20).max() - 1.0
    cum_drop_30 = close / close.rolling(30).max() - 1.0

    days_since_high_63 = close.rolling(63).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True) / 63.0
    below_anchor_ratio_63 = (close < ma180).rolling(63).mean()
    new_low_ratio_21 = (close <= close.rolling(21).min()).rolling(21).mean()
    downside_ratio_10 = (soxl_ret < 0).rolling(10).mean()
    downside_ratio_21 = (soxl_ret < 0).rolling(21).mean()
    downside_semivol_21 = soxl_ret.where(soxl_ret < 0, 0.0).pow(2).rolling(21).mean().pow(0.5) * np.sqrt(252)
    loss_streak_21 = _streaks(soxl_ret < 0).clip(upper=21) / 21.0
    vol_coef_21 = realized_vol_21 / (realized_vol_21.rolling(252).median() + 1e-8)

    vix_zscore = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
    vix_percentile_252 = _rolling_percentile(vix, 252)
    vix_momentum_10 = vix.pct_change(10)

    tracking_error = soxl_ret - 3.0 * soxx_ret
    tracking_error_5 = tracking_error.rolling(5).mean()
    tracking_error_21 = tracking_error.rolling(21).mean()
    tracking_error_vol_21 = tracking_error.rolling(21).std()
    decay_63 = (soxl_ret.rolling(63).sum() - 3.0 * soxx_ret.rolling(63).sum())
    relative_strength_21 = soxl_ret.rolling(21).sum() - soxx_ret.rolling(21).sum()

    trend_gap_21 = ma5 / ma21 - 1.0
    trend_gap_63 = ma21 / ma63 - 1.0
    anchor_gap_180 = close / ma180 - 1.0
    trend_slope_21 = ma21 / ma21.shift(10) - 1.0
    trend_slope_63 = ma63 / ma63.shift(21) - 1.0
    anchor_slope_180 = ma180 / ma180.shift(21) - 1.0
    intraday_range = (high - low) / close
    volume_shock = volume / volume_ma21 - 1.0
    mom_21 = soxl_ret.rolling(21).sum()
    mom_63 = soxl_ret.rolling(63).sum()

    regime_risk = (
        0.22 * vix_percentile_252.fillna(0.0)
        + 0.18 * realized_vol_21.rank(pct=True).fillna(0.0)
        + 0.18 * drawdown_63.abs().rank(pct=True).fillna(0.0)
        + 0.15 * downside_ratio_21.fillna(0.0)
        + 0.12 * downside_semivol_21.rank(pct=True).fillna(0.0)
        + 0.08 * tracking_error_vol_21.rank(pct=True).fillna(0.0)
        + 0.07 * loss_streak_21.fillna(0.0)
    )

    feature_df = pd.DataFrame(
        {
            "return": soxl_ret,
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
            "below_anchor_ratio_63": below_anchor_ratio_63,
            "new_low_ratio_21": new_low_ratio_21,
            "loss_streak_21": loss_streak_21,
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
            "close": close,
        }
    ).replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna()

    price = feature_df.pop("close")
    feature_df = feature_df[FEATURE_COLUMNS]
    feature_df.index.name = "datetime"
    multi_index = pd.MultiIndex.from_arrays(
        [feature_df.index, np.repeat("SOXL", len(feature_df))],
        names=["datetime", "instrument"],
    )
    feature_df.index = multi_index
    price.index = multi_index
    return feature_df, price


def build_feature_frame(start_time: str, end_time: str) -> Tuple[pd.DataFrame, pd.Series]:
    soxl, soxx = build_market_frame(start_time, end_time)
    return engineer_feature_frame(soxl, soxx)


class SoxlDataHandler(DataHandlerLP):
    def __init__(self, start_time: str, end_time: str):
        # Keep both the feature frame and aligned price frame so training,
        # backtest, and runtime all read from the same daily source of truth.
        feature_df, price = build_feature_frame(start_time, end_time)
        self.price_frame = price.to_frame("price")
        self.feature_frame = feature_df
        loader_df = pd.concat({"feature": feature_df}, axis=1)
        super().__init__(
            instruments=["SOXL"],
            start_time=start_time,
            end_time=end_time,
            data_loader=StaticDataLoader(loader_df),
            infer_processors=[],
            learn_processors=[],
            shared_processors=[],
            process_type=DataHandlerLP.PTYPE_A,
            drop_raw=False,
        )


def build_dataset() -> DatasetH:
    return DatasetH(
        handler=SoxlDataHandler(start_time=TRAIN_SEG[0], end_time=TEST_SEG[1]),
        segments={"train": TRAIN_SEG, "valid": VALID_SEG, "test": TEST_SEG},
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
