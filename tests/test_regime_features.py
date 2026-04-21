from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd

from src.soxl_quant.data import FEATURE_COLUMNS, _build_transition_state_features, engineer_feature_frame


def test_regime_feature_frame_has_expected_columns():
    df = pd.read_csv("data/soxl_soxx_vix_full.csv", parse_dates=["date"]).set_index("date")
    soxl = pd.DataFrame(
        {
            "$close": df["soxl_close"],
            "$open": df["soxl_open"],
            "$high": df["soxl_high"],
            "$low": df["soxl_low"],
            "$volume": df["soxl_volume"],
            "$vix": df["vix"],
            "Mean($close, 5)": df["soxl_close"].rolling(5).mean(),
            "Mean($close, 21)": df["soxl_close"].rolling(21).mean(),
            "Mean($close, 63)": df["soxl_close"].rolling(63).mean(),
            "Mean($close, 180)": df["soxl_close"].rolling(180).mean(),
            "Ref($close, 21)": df["soxl_close"].shift(21),
            "Ref($close, 63)": df["soxl_close"].shift(63),
            "Std($close / Ref($close, 1) - 1, 10)": df["soxl_close"].pct_change().rolling(10).std(),
            "Std($close / Ref($close, 1) - 1, 21)": df["soxl_close"].pct_change().rolling(21).std(),
            "Std($close / Ref($close, 1) - 1, 63)": df["soxl_close"].pct_change().rolling(63).std(),
            "Mean($volume, 21)": df["soxl_volume"].rolling(21).mean(),
        }
    )
    soxx = pd.DataFrame(
        {
            "$close": df["soxx_close"],
            "$open": df["soxx_open"],
            "$high": df["soxx_high"],
            "$low": df["soxx_low"],
            "$volume": df["soxx_volume"],
            "Ref($close, 5)": df["soxx_close"].shift(5),
            "Ref($close, 21)": df["soxx_close"].shift(21),
        }
    )

    feature_df, price = engineer_feature_frame(soxl, soxx)
    assert list(feature_df.columns) == FEATURE_COLUMNS
    assert "SOXL" in feature_df.index.get_level_values("instrument").unique()
    assert feature_df["regime_risk"].between(0, 1).mean() > 0.95
    assert price.notna().all()


def test_transition_v23_feature_flag_extends_schema_without_default_drift():
    import src.soxl_quant.data as data_mod

    previous = os.environ.get("SOXL_USE_TRANSITION_V23")
    try:
        os.environ.pop("SOXL_USE_TRANSITION_V23", None)
        data_mod = importlib.reload(data_mod)
        base_columns = list(data_mod.FEATURE_COLUMNS)
        assert "stress_then_relief_flag" not in base_columns
        assert "trend_restart_flag" not in base_columns

        os.environ["SOXL_USE_TRANSITION_V23"] = "1"
        data_mod = importlib.reload(data_mod)
        assert "stress_then_relief_flag" in data_mod.FEATURE_COLUMNS
        assert "trend_restart_flag" in data_mod.FEATURE_COLUMNS
        assert len(data_mod.FEATURE_COLUMNS) == len(base_columns) + 6
    finally:
        if previous is None:
            os.environ.pop("SOXL_USE_TRANSITION_V23", None)
        else:
            os.environ["SOXL_USE_TRANSITION_V23"] = previous
        importlib.reload(data_mod)


def test_transition_v23_flags_fire_only_in_expected_states():
    previous = os.environ.get("SOXL_USE_TRANSITION_V23")
    try:
        index = pd.date_range("2020-01-01", periods=30, freq="B")
        transition = _build_transition_state_features(
            anchor_gap_180=pd.Series(np.full(len(index), -0.05), index=index),
            below_anchor_ratio_63=pd.Series(np.linspace(0.40, 0.90, len(index)), index=index),
            relative_strength_21=pd.Series(np.linspace(0.00, 0.12, len(index)), index=index),
            regime_risk=pd.Series(np.r_[np.full(21, 0.72), np.full(9, 0.45)], index=index),
            vix_zscore=pd.Series(np.r_[np.full(21, 1.20), np.linspace(1.20, 0.10, 9)], index=index),
            downside_ratio_10=pd.Series(np.r_[np.full(21, 0.70), np.linspace(0.70, 0.30, 9)], index=index),
            trend_slope_10=pd.Series(np.r_[np.full(21, 0.00), np.full(9, 0.03)], index=index),
        )

        assert transition["days_above_stress_threshold_21"].max() >= 21.0
        assert set(transition["stress_then_relief_flag"].dropna().unique()).issubset({0.0, 1.0})
        assert set(transition["trend_restart_flag"].dropna().unique()).issubset({0.0, 1.0})
        assert transition["stress_then_relief_flag"].sum() > 0
        assert transition["trend_restart_flag"].sum() > 0

        flagged = transition["stress_then_relief_flag"] == 1.0
        assert (transition["days_above_stress_threshold_21"][flagged] >= 5.0).all()
        assert (transition["vix_zscore_delta_5"][flagged] < -0.35).all()
        assert (transition["downside_ratio_10_delta_5"][flagged] < -0.05).all()
    finally:
        if previous is None:
            os.environ.pop("SOXL_USE_TRANSITION_V23", None)
        else:
            os.environ["SOXL_USE_TRANSITION_V23"] = previous
