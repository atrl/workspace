from __future__ import annotations

import numpy as np
import pandas as pd

import src.soxl_quant.data as data_mod
from src.soxl_quant.core.tushare import normalize_tushare_frame, tushare_csv_path
from src.soxl_quant.data import engineer_feature_frame, feature_columns_for_packs


def _frame_from_panel(df: pd.DataFrame, prefix: str, *, include_vix: bool = False) -> pd.DataFrame:
    payload = {
        "$close": df[f"{prefix}_close"],
        "$open": df[f"{prefix}_open"],
        "$high": df[f"{prefix}_high"],
        "$low": df[f"{prefix}_low"],
        "$volume": df[f"{prefix}_volume"],
        "Mean($close, 5)": df[f"{prefix}_close"].rolling(5).mean(),
        "Mean($close, 21)": df[f"{prefix}_close"].rolling(21).mean(),
        "Mean($close, 63)": df[f"{prefix}_close"].rolling(63).mean(),
        "Mean($close, 180)": df[f"{prefix}_close"].rolling(180).mean(),
        "Std($close / Ref($close, 1) - 1, 10)": df[f"{prefix}_close"].pct_change().rolling(10).std(),
        "Std($close / Ref($close, 1) - 1, 21)": df[f"{prefix}_close"].pct_change().rolling(21).std(),
        "Std($close / Ref($close, 1) - 1, 63)": df[f"{prefix}_close"].pct_change().rolling(63).std(),
        "Mean($volume, 21)": df[f"{prefix}_volume"].rolling(21).mean(),
    }
    if include_vix:
        payload["$vix"] = df["vix"]
    return pd.DataFrame(payload, index=df.index)


def test_normalize_tushare_frame_sorts_and_adds_date_column():
    raw = pd.DataFrame(
        {
            "ts_code": ["SOXX", "SOXX"],
            "trade_date": ["20240103", "20240102"],
            "turnover_ratio": [0.34, 0.31],
        }
    )

    normalized = normalize_tushare_frame(raw)

    assert normalized["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert normalized["turnover_ratio"].tolist() == [0.31, 0.34]


def test_tushare_pack_is_optional_when_cache_is_missing(tmp_path, monkeypatch):
    panel = pd.read_csv(data_mod.ROOT / "data/processed/panels/soxl_soxx_vix_full.csv", parse_dates=["date"])
    panel = panel.set_index("date").tail(360)
    monkeypatch.setattr(data_mod, "EXTERNAL_DATA_DIR", tmp_path)

    soxl = _frame_from_panel(panel, "soxl", include_vix=True)
    soxx = _frame_from_panel(panel, "soxx")

    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("tushare_us_daily_adj",))
    flat = feature_df.droplevel("instrument")

    expected = set(feature_columns_for_packs(("tushare_us_daily_adj",))) - set(feature_columns_for_packs(()))
    assert expected.issubset(flat.columns)
    assert (flat[list(expected)].abs().sum(axis=1) == 0.0).all()


def test_tushare_pack_uses_lagged_cached_turnover(tmp_path, monkeypatch):
    panel = pd.read_csv(data_mod.ROOT / "data/processed/panels/soxl_soxx_vix_full.csv", parse_dates=["date"])
    panel = panel.set_index("date").tail(360)
    monkeypatch.setattr(data_mod, "EXTERNAL_DATA_DIR", tmp_path)

    turnover = pd.Series(np.linspace(0.10, 2.50, len(panel)), index=panel.index)
    cached = pd.DataFrame(
        {
            "date": panel.index.strftime("%Y-%m-%d"),
            "ts_code": "SOXX",
            "trade_date": panel.index.strftime("%Y%m%d"),
            "turnover_ratio": turnover.to_numpy(),
            "amount": np.linspace(1000.0, 3000.0, len(panel)),
            "vol": np.linspace(100.0, 500.0, len(panel)),
            "adj_factor": 1.0,
            "vwap": panel["soxx_close"].to_numpy(),
        }
    )
    path = tushare_csv_path("us_daily_adj", "SOXX", root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_csv(path, index=False)

    soxl = _frame_from_panel(panel, "soxl", include_vix=True)
    soxx = _frame_from_panel(panel, "soxx")

    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("tushare_us_daily_adj",))
    flat = feature_df.droplevel("instrument")
    shifted = turnover.shift(1)
    expected = ((shifted - shifted.rolling(21).mean()) / (shifted.rolling(21).std() + 1e-8)).fillna(0.0)

    aligned = expected.reindex(flat.index)
    actual = flat["TUSHARE_SOXX_TURNOVER_ZSCORE_21"].reindex(flat.index)
    assert np.allclose(actual.tail(50).to_numpy(), aligned.tail(50).to_numpy(), atol=1e-8)
