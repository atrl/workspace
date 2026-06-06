from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.settings import CSV_PATH
from src.soxl_quant.data import (
    FEATURE_COLUMNS,
    _build_transition_state_features,
    _align_external,
    engineer_feature_frame,
    feature_columns_for_packs,
    _load_external_series,
)


def test_regime_feature_frame_has_expected_columns():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
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


def test_transition_signal_pack_includes_signed_tracking_error_reversion():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
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
        }
    )

    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("transition_signals",))
    flat = feature_df.droplevel("instrument")

    expected_columns = {
        "SIGNED_TE_REVERSION_5",
        "SIGNED_TE_REVERSION_10",
        "SIGNED_TE_REVERSION_21",
    }
    assert expected_columns.issubset(set(feature_columns_for_packs(("transition_signals",))))
    assert expected_columns.issubset(set(flat.columns))
    assert flat[list(expected_columns)].replace([np.inf, -np.inf], np.nan).notna().all().all()

    close = soxl["$close"].astype(float).reindex(flat.index)
    und_close = soxx["$close"].astype(float).reindex(flat.index)
    tracking_error = close.pct_change() - 3.0 * und_close.pct_change()
    rolling_signed_te = tracking_error.rolling(10).sum()
    expected = -(
        (rolling_signed_te - rolling_signed_te.rolling(252).mean())
        / (rolling_signed_te.rolling(252).std() + 1e-8)
    )
    expected = expected.reindex(flat.index).dropna()
    actual = flat["SIGNED_TE_REVERSION_10"].reindex(expected.index)
    assert np.allclose(actual.tail(100).to_numpy(), expected.tail(100).to_numpy(), atol=1e-8)


def test_transition_signal_pack_includes_intraday_rejection_event():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
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
        }
    )

    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("transition_signals",))
    flat = feature_df.droplevel("instrument")

    assert "INTRADAY_REJECTION_EVENT" in feature_columns_for_packs(("transition_signals",))
    assert "INTRADAY_REJECTION_EVENT" in flat.columns
    assert flat["INTRADAY_REJECTION_EVENT"].notna().all()
    assert (flat["INTRADAY_REJECTION_EVENT"] >= 0.0).all()

    aligned = soxl.reindex(flat.index)
    intraday_range = (aligned["$high"] - aligned["$low"]) / aligned["$close"]
    range_zscore = (
        (intraday_range - intraday_range.rolling(252).mean())
        / (intraday_range.rolling(252).std() + 1e-8)
    ).clip(lower=0.0)
    day_range = (aligned["$high"] - aligned["$low"]).clip(lower=1e-8)
    close_location = ((aligned["$close"] - aligned["$low"]) / day_range).clip(0.0, 1.0)
    lower_wick_ratio = (
        (pd.concat([aligned["$open"], aligned["$close"]], axis=1).min(axis=1) - aligned["$low"])
        / day_range
    ).clip(0.0, 1.0)
    avg_volume_20 = aligned["$volume"].rolling(20).mean()
    relative_volume_zscore = (
        (aligned["$volume"] - avg_volume_20)
        / (avg_volume_20.rolling(20).std() + 1e-8)
    ).clip(lower=0.0)
    expected = (
        lower_wick_ratio * close_location * range_zscore * (1.0 + relative_volume_zscore)
    ).fillna(0.0)
    actual = flat["INTRADAY_REJECTION_EVENT"].reindex(expected.index)
    assert np.allclose(actual.tail(100).to_numpy(), expected.tail(100).to_numpy(), atol=1e-8)


def test_soxx_spread_oscillation_pack_builds_ratio_and_chop_features():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
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
        }
    )

    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("soxx_spread_oscillation",))
    flat = feature_df.droplevel("instrument")

    expected_columns = {
        "SOXL_SOXX_RATIO_Z_252",
        "SOXL_SOXX_RATIO_WIDEN_5",
        "SOXL_SOXX_SPREAD_VOL_Z_21",
        "SOXL_SOXX_CHOP_DIVERGENCE_21",
    }
    assert expected_columns.issubset(set(feature_columns_for_packs(("soxx_spread_oscillation",))))
    assert expected_columns.issubset(set(flat.columns))
    assert flat[list(expected_columns)].replace([np.inf, -np.inf], np.nan).notna().all().all()

    close = soxl["$close"].astype(float).reindex(flat.index)
    und_close = soxx["$close"].astype(float).reindex(flat.index)
    log_ratio = np.log(close / (und_close + 1e-8))
    ratio_z = ((log_ratio - log_ratio.rolling(252).mean()) / (log_ratio.rolling(252).std() + 1e-8)).fillna(0.0)
    expected_widen = ratio_z.diff(5).fillna(0.0)
    actual_widen = flat["SOXL_SOXX_RATIO_WIDEN_5"].reindex(expected_widen.index)
    assert np.allclose(actual_widen.tail(100).to_numpy(), expected_widen.tail(100).to_numpy(), atol=1e-8)


def test_release_signal_pack_curates_reentry_signals():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
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
        }
    )

    curated_columns = {
        "RB_COOLDOWN_PROXY",
        "TRANS_VIX_DIVERGENCE",
        "CICC_RETURN_VOLUME_SYNC",
    }
    assert curated_columns.issubset(set(feature_columns_for_packs(("release_signals_v1",))))

    release_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("release_signals_v1",))
    reference_df, _ = engineer_feature_frame(
        soxl,
        soxx,
        feature_packs=("rulebase_signals", "transition_signals", "cicc_price_volume"),
    )
    release_flat = release_df.droplevel("instrument")
    reference_flat = reference_df.droplevel("instrument")
    for column in curated_columns:
        assert column in release_flat.columns
        assert np.allclose(
            release_flat[column].tail(100).to_numpy(),
            reference_flat[column].tail(100).to_numpy(),
            atol=1e-8,
        )


def test_macro_extreme_risk_pack_builds_lagged_narrow_macro_features():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
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
        }
    )

    expected_columns = {
        "MACRO_EXTREME_VIX_TERM_SLOPE_LAG1",
        "MACRO_EXTREME_VIX_SHORT_RATIO_LAG1",
        "MACRO_EXTREME_VVIX_ZSCORE_252_LAG1",
        "MACRO_EXTREME_SKEW_ZSCORE_252_LAG1",
        "MACRO_EXTREME_YC_10Y2Y_ZSCORE_252_LAG1",
        "MACRO_EXTREME_YC_10Y2Y_DELTA_21_LAG1",
        "MACRO_EXTREME_DXY_ZSCORE_252_LAG1",
        "MACRO_EXTREME_DXY_ROC_63_LAG1",
        "MACRO_EXTREME_OIL_ROC_63_LAG1",
        "MACRO_EXTREME_RISK_PRESSURE_LAG1",
        "MACRO_EXTREME_RISK_PRESSURE_RELIEF_5_LAG1",
    }
    pack_columns = set(feature_columns_for_packs(("macro_extreme_risk_v1",)))
    assert expected_columns.issubset(pack_columns)
    assert "MACRO_CREDIT_SPREAD" not in pack_columns

    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("macro_extreme_risk_v1",))
    flat = feature_df.droplevel("instrument")
    assert expected_columns.issubset(set(flat.columns))
    assert flat[list(expected_columns)].replace([np.inf, -np.inf], np.nan).notna().all().all()

    aligned_index = flat.index
    vix = soxl["$vix"].astype(float)
    vix3m = _align_external(_load_external_series("VIX3M.csv", "CLOSE"), soxl.index)
    raw_vix_term_slope = (vix / vix3m.replace(0.0, np.nan) - 1.0).fillna(0.0)
    expected_lagged_slope = raw_vix_term_slope.shift(1).reindex(aligned_index)
    actual_lagged_slope = flat["MACRO_EXTREME_VIX_TERM_SLOPE_LAG1"].reindex(aligned_index)
    assert np.allclose(
        actual_lagged_slope.tail(250).to_numpy(),
        expected_lagged_slope.tail(250).to_numpy(),
        atol=1e-8,
    )
    assert not np.allclose(
        actual_lagged_slope.tail(250).to_numpy(),
        raw_vix_term_slope.reindex(aligned_index).tail(250).to_numpy(),
        atol=1e-8,
    )
    assert (flat["MACRO_EXTREME_RISK_PRESSURE_LAG1"] >= 0.0).all()
    assert flat["MACRO_EXTREME_RISK_PRESSURE_LAG1"].std() > 0.0


def test_transition_v23_flags_fire_only_in_expected_states():
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
