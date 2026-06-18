from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SupervisedDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str = "target_return"


def _calendar_dates(root: Path = ROOT) -> pd.DatetimeIndex:
    calendar_path = root / "data" / "qlib" / "soxl" / "calendars" / "day.txt"
    values = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DatetimeIndex(pd.to_datetime(values), name="date")


def load_soxl_support_frame(*, start: str, end: str, root: Path = ROOT) -> pd.DataFrame:
    panel_path = root / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv"
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    calendar = _calendar_dates(root)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    valid_dates = calendar[(calendar >= start_ts) & (calendar <= end_ts)]
    frame = panel.loc[panel["date"].isin(valid_dates)].copy()
    if frame.empty:
        raise ValueError(f"no SOXL support rows between {start} and {end}")
    result = pd.DataFrame(
        {
            "date": frame["date"],
            "open": frame["soxl_open"].astype(float),
            "high": frame["soxl_high"].astype(float),
            "low": frame["soxl_low"].astype(float),
            "close": frame["soxl_close"].astype(float),
            "volume": frame["soxl_volume"].astype(float),
            "soxx_close": frame["soxx_close"].astype(float),
            "soxx_volume": frame["soxx_volume"].astype(float),
            "vix": frame["vix"].astype(float).ffill().bfill(),
        }
    ).sort_values("date")
    result.attrs["data_source"] = "qlib_calendar_plus_soxl_support_panel"
    result.attrs["panel_path"] = str(panel_path)
    return result.reset_index(drop=True)


def build_supervised_dataset(frame: pd.DataFrame, *, horizon: int = 1) -> SupervisedDataset:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    data = frame.sort_values("date").copy()
    close = data["close"].astype(float)
    soxx_close = data["soxx_close"].astype(float)
    volume = data["volume"].astype(float)
    vix = data["vix"].astype(float)

    data["soxl_return_1d"] = close.pct_change()
    data["soxx_return_1d"] = soxx_close.pct_change()
    data["soxl_return_5d"] = close.pct_change(5)
    data["soxx_return_5d"] = soxx_close.pct_change(5)
    data["soxl_soxx_relative_5d"] = data["soxl_return_5d"] - data["soxx_return_5d"]
    data["vix_level"] = vix
    data["vix_change_1d"] = vix.pct_change()
    data["intraday_range"] = (data["high"].astype(float) - data["low"].astype(float)) / close
    data["volume_z_20"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-12)
    data["trend_5"] = close / close.rolling(5).mean() - 1.0
    data["trend_21"] = close / close.rolling(21).mean() - 1.0
    data["realized_vol_5"] = data["soxl_return_1d"].rolling(5).std()
    data["realized_vol_21"] = data["soxl_return_1d"].rolling(21).std()
    data["drawdown_21"] = close / close.rolling(21).max() - 1.0
    data["target_return"] = close.pct_change(horizon).shift(-horizon)

    feature_columns = (
        "soxl_return_1d",
        "soxx_return_1d",
        "soxl_return_5d",
        "soxx_return_5d",
        "soxl_soxx_relative_5d",
        "vix_level",
        "vix_change_1d",
        "intraday_range",
        "volume_z_20",
        "trend_5",
        "trend_21",
        "realized_vol_5",
        "realized_vol_21",
        "drawdown_21",
    )
    keep_columns = ("date", "close", "vix", *feature_columns, "target_return")
    supervised = data.loc[:, keep_columns].replace([float("inf"), float("-inf")], pd.NA)
    supervised = supervised.dropna(subset=["target_return"]).copy()
    supervised.loc[:, feature_columns] = supervised.loc[:, feature_columns].fillna(0.0)
    return SupervisedDataset(frame=supervised.reset_index(drop=True), feature_columns=feature_columns)
