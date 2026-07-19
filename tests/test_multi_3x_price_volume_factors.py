from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt


def _panel(periods: int = 180, instruments: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    frames: list[pd.DataFrame] = []
    for offset, instrument in enumerate(instruments):
        underlying_ret = 0.0004 + 0.006 * np.sin(np.arange(periods) / (7.0 + offset))
        underlying = 100.0 * np.exp(np.cumsum(underlying_ret))
        residual = 0.0015 * np.cos(np.arange(periods) / (5.0 + offset))
        close = 25.0 * np.exp(np.cumsum(3.0 * underlying_ret + residual))
        open_ = close * (1.0 - 0.003 * np.sin(np.arange(periods) / 3.0))
        high = np.maximum(open_, close) * 1.012
        low = np.minimum(open_, close) * 0.988
        volume = 1_000_000.0 * (1.0 + 0.20 * np.sin(np.arange(periods) / 9.0 + offset))
        frames.append(
            pd.DataFrame(
                {
                    "instrument": instrument,
                    "datetime": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "underlying_close": underlying,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_price_volume_factor_contract_is_asset_local_bounded_and_lagged():
    from src.factor.multi_3x_price_volume import (
        PRICE_ONLY_BASELINE_FEATURES,
        PRICE_VOLUME_PRESSURE_FEATURES,
        build_multi_3x_price_volume_frame,
    )

    frame = build_multi_3x_price_volume_frame(
        _panel(), normalization_window=80, min_periods=30, label_horizon=10
    )

    expected = set(PRICE_ONLY_BASELINE_FEATURES + PRICE_VOLUME_PRESSURE_FEATURES)
    assert expected.issubset(frame.columns)
    assert frame[list(expected)].notna().any().all()
    for feature in expected:
        assert frame[feature].dropna().between(0.0, 1.0).all(), feature
    assert "instrument" not in PRICE_VOLUME_PRESSURE_FEATURES
    assert not any("vix" in feature or "cross_asset" in feature for feature in PRICE_VOLUME_PRESSURE_FEATURES)
    assert frame.sort_values("datetime").groupby("instrument")["breakdown_label"].tail(11).isna().all()


def test_price_volume_features_are_invariant_to_future_price_and_volume_mutation():
    from src.factor.multi_3x_price_volume import (
        ALL_FACTOR_FEATURES,
        build_multi_3x_price_volume_frame,
    )

    panel = _panel()
    cutoff = pd.Timestamp("2022-06-30")
    original = build_multi_3x_price_volume_frame(
        panel, normalization_window=80, min_periods=30, label_horizon=10
    )
    mutated = panel.copy()
    future = mutated["datetime"] > cutoff
    mutated.loc[future, ["open", "high", "low", "close", "underlying_close"]] *= 5.0
    mutated.loc[future, "volume"] *= 25.0
    rebuilt = build_multi_3x_price_volume_frame(
        mutated, normalization_window=80, min_periods=30, label_horizon=10
    )

    columns = ["instrument", "datetime", *ALL_FACTOR_FEATURES]
    left = original.loc[original["datetime"] <= cutoff, columns].reset_index(drop=True)
    right = rebuilt.loc[rebuilt["datetime"] <= cutoff, columns].reset_index(drop=True)
    pdt.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


def test_high_volume_failed_breakout_raises_pressure_after_observation():
    from src.factor.multi_3x_price_volume import build_multi_3x_price_volume_frame

    base = _panel(instruments=("AAA",))
    stressed = base.copy()
    asset = stressed.sort_values("datetime").reset_index(drop=True)
    shock_rows = range(90, 100)
    for row in shock_rows:
        prior_high = float(asset.loc[row - 20 : row - 1, "high"].max())
        asset.loc[row, "high"] = prior_high * 1.01
        asset.loc[row, "close"] = prior_high * 0.990
        asset.loc[row, "open"] = prior_high * 0.995
        asset.loc[row, "low"] = prior_high * 0.980
        asset.loc[row, "volume"] *= 12.0
    stressed = asset

    base_features = build_multi_3x_price_volume_frame(
        base, normalization_window=60, min_periods=20, label_horizon=10
    ).sort_values("datetime")
    stressed_features = build_multi_3x_price_volume_frame(
        stressed, normalization_window=60, min_periods=20, label_horizon=10
    ).sort_values("datetime")

    # The feature is delayed by one session, so compare immediately after the
    # observed failed-breakout cluster.
    compare = slice(91, 103)
    stressed_mean = stressed_features.iloc[compare]["failed_breakout_pressure_20"].mean()
    base_mean = base_features.iloc[compare]["failed_breakout_pressure_20"].mean()
    assert stressed_mean > base_mean + 0.10


def test_constant_volume_and_zero_intraday_range_do_not_create_infinity():
    from src.factor.multi_3x_price_volume import ALL_FACTOR_FEATURES, build_multi_3x_price_volume_frame

    panel = _panel(instruments=("AAA",))
    panel["volume"] = 1_000_000.0
    panel.loc[40:55, "open"] = panel.loc[40:55, "close"].to_numpy()
    panel.loc[40:55, "high"] = panel.loc[40:55, "close"].to_numpy()
    panel.loc[40:55, "low"] = panel.loc[40:55, "close"].to_numpy()
    frame = build_multi_3x_price_volume_frame(
        panel, normalization_window=60, min_periods=20, label_horizon=10
    )

    values = frame[ALL_FACTOR_FEATURES].to_numpy(dtype=float)
    assert not np.isinf(values).any()


def test_core_factors_do_not_change_when_unrelated_asset_is_added_or_mutated():
    from src.factor.multi_3x_price_volume import PRICE_VOLUME_PRESSURE_FEATURES, build_multi_3x_price_volume_frame

    one = _panel(instruments=("AAA",))
    multi = _panel(instruments=("AAA", "BBB"))
    multi.loc[multi["instrument"] == "BBB", ["close", "high", "low", "volume"]] *= 50.0
    single_result = build_multi_3x_price_volume_frame(
        one, normalization_window=80, min_periods=30, label_horizon=10
    )
    multi_result = build_multi_3x_price_volume_frame(
        multi, normalization_window=80, min_periods=30, label_horizon=10
    )

    left = single_result.loc[single_result["instrument"] == "AAA", PRICE_VOLUME_PRESSURE_FEATURES].reset_index(drop=True)
    right = multi_result.loc[multi_result["instrument"] == "AAA", PRICE_VOLUME_PRESSURE_FEATURES].reset_index(drop=True)
    pdt.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)
