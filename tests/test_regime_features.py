from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.settings import CSV_PATH
from src.soxl_quant.data import (
    FEATURE_COLUMNS,
    _build_transition_state_features,
    engineer_feature_frame,
    feature_columns_for_packs,
)


def _soxl_soxx_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return soxl, soxx


def test_regime_feature_frame_has_expected_columns():
    soxl, soxx = _soxl_soxx_frames()

    feature_df, price = engineer_feature_frame(soxl, soxx)
    assert list(feature_df.columns) == FEATURE_COLUMNS
    assert "SOXL" in feature_df.index.get_level_values("instrument").unique()
    assert feature_df["regime_risk"].between(0, 1).mean() > 0.95
    assert price.notna().all()


def test_qlib_alpha158_pack_builds_full_prefixed_schema_and_reference_formulas():
    soxl, soxx = _soxl_soxx_frames()

    selected_columns = feature_columns_for_packs(("qlib_alpha158",))
    alpha_columns = [column for column in selected_columns if column.startswith("QLIB_ALPHA158_")]
    assert len(alpha_columns) == 158
    assert {
        "QLIB_ALPHA158_KMID",
        "QLIB_ALPHA158_OPEN0",
        "QLIB_ALPHA158_ROC5",
        "QLIB_ALPHA158_RSV20",
        "QLIB_ALPHA158_VMA60",
        "QLIB_ALPHA158_VSUMD60",
    }.issubset(alpha_columns)

    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("qlib_alpha158",))
    flat = feature_df.droplevel("instrument")
    assert set(alpha_columns).issubset(flat.columns)
    assert flat[alpha_columns].replace([np.inf, -np.inf], np.nan).notna().all().all()

    close = soxl["$close"].astype(float).reindex(flat.index)
    open_ = soxl["$open"].astype(float).reindex(flat.index)
    high = soxl["$high"].astype(float).reindex(flat.index)
    low = soxl["$low"].astype(float).reindex(flat.index)
    volume = soxl["$volume"].astype(float).reindex(flat.index)
    expected = pd.DataFrame(
        {
            "QLIB_ALPHA158_KMID": (close - open_) / open_,
            "QLIB_ALPHA158_OPEN0": open_ / close,
            "QLIB_ALPHA158_ROC5": close.shift(5) / close,
            "QLIB_ALPHA158_MA20": close.rolling(20).mean() / close,
            "QLIB_ALPHA158_RSV20": (close - low.rolling(20).min())
            / (high.rolling(20).max() - low.rolling(20).min() + 1e-12),
            "QLIB_ALPHA158_VMA20": volume.rolling(20).mean() / (volume + 1e-12),
        }
    ).dropna()
    actual = flat[expected.columns].reindex(expected.index)
    assert np.allclose(actual.tail(100).to_numpy(), expected.tail(100).to_numpy(), atol=1e-8)


def test_qlib_alpha158_pack_does_not_look_ahead_when_latest_bar_changes():
    soxl, soxx = _soxl_soxx_frames()
    original_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("qlib_alpha158",))

    mutated_soxl = soxl.copy()
    latest = mutated_soxl.index[-1]
    mutated_soxl.loc[latest, "$open"] *= 1.40
    mutated_soxl.loc[latest, "$high"] *= 1.45
    mutated_soxl.loc[latest, "$low"] *= 0.70
    mutated_soxl.loc[latest, "$close"] *= 1.35
    mutated_soxl.loc[latest, "$volume"] *= 3.0
    mutated_df, _ = engineer_feature_frame(mutated_soxl, soxx, feature_packs=("qlib_alpha158",))

    alpha_columns = [column for column in original_df.columns if column.startswith("QLIB_ALPHA158_")]
    original_flat = original_df.droplevel("instrument")[alpha_columns]
    mutated_flat = mutated_df.droplevel("instrument")[alpha_columns]
    common_index = original_flat.index.intersection(mutated_flat.index)
    prior_index = common_index[common_index < latest]
    assert len(prior_index) > 100
    assert np.allclose(
        original_flat.loc[prior_index].to_numpy(),
        mutated_flat.loc[prior_index].to_numpy(),
        atol=1e-12,
        equal_nan=True,
    )


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
