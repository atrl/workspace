from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt


def _pair_panel(periods: int = 240) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=periods)
    underlying_ret = 0.0005 + 0.007 * np.sin(np.arange(periods) / 8.0)
    underlying = 100.0 * np.exp(np.cumsum(underlying_ret))
    residual = 0.001 * np.cos(np.arange(periods) / 5.0)
    close = 30.0 * np.exp(np.cumsum(3.0 * underlying_ret + residual))
    open_ = close * (1.0 - 0.002 * np.sin(np.arange(periods) / 3.0))
    return pd.DataFrame(
        {
            "instrument": "AAA",
            "datetime": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1_000_000.0,
            "underlying_close": underlying,
        }
    )


def _bellwether_panel(periods: int = 240) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=periods)
    pieces = []
    for offset, symbol in enumerate(("NVDA", "AVGO", "AMD", "TSM", "ASML")):
        ret = 0.0004 + 0.008 * np.sin(np.arange(periods) / (7.0 + offset) + offset)
        close = 100.0 * np.exp(np.cumsum(ret))
        pieces.append(pd.DataFrame({"instrument": symbol, "datetime": dates, "close": close}))
    return pd.concat(pieces, ignore_index=True)


def test_path_and_fragility_factors_are_bounded_lagged_and_future_invariant():
    from src.factor.soxl_exhaustion_structure import (
        LEVERAGE_FRAGILITY_FEATURES,
        PRICE_PATH_EXHAUSTION_FEATURES,
        build_exhaustion_fragility_frame,
    )

    panel = _pair_panel()
    original = build_exhaustion_fragility_frame(panel, normalization_window=80, min_periods=30)
    features = PRICE_PATH_EXHAUSTION_FEATURES + LEVERAGE_FRAGILITY_FEATURES
    assert original[features].notna().any().all()
    assert original[features].apply(lambda values: values.dropna().between(0.0, 1.0).all()).all()

    cutoff = pd.Timestamp("2021-08-31")
    mutated = panel.copy()
    mutated.loc[mutated["datetime"] > cutoff, ["open", "high", "low", "close", "underlying_close"]] *= 4.0
    rebuilt = build_exhaustion_fragility_frame(mutated, normalization_window=80, min_periods=30)
    columns = ["instrument", "datetime", *features]
    pdt.assert_frame_equal(
        original.loc[original["datetime"] <= cutoff, columns].reset_index(drop=True),
        rebuilt.loc[rebuilt["datetime"] <= cutoff, columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_failed_breakouts_and_beta_asymmetry_raise_pressure_after_observation():
    from src.factor.soxl_exhaustion_structure import build_exhaustion_fragility_frame

    base = _pair_panel()
    stressed = base.copy().reset_index(drop=True)
    for row in range(150, 160):
        prior_high = float(stressed.loc[row - 20 : row - 1, "high"].max())
        stressed.loc[row, ["open", "high", "low", "close"]] = [
            prior_high * 0.995,
            prior_high * 1.01,
            prior_high * 0.965,
            prior_high * 0.975,
        ]
        stressed.loc[row, "underlying_close"] = stressed.loc[row - 1, "underlying_close"] * 1.006
    base_f = build_exhaustion_fragility_frame(base, normalization_window=80, min_periods=30)
    stress_f = build_exhaustion_fragility_frame(stressed, normalization_window=80, min_periods=30)
    after = stressed.loc[160:170, "datetime"]
    assert stress_f.loc[stress_f["datetime"].isin(after), "failed_breakout_persistence_10"].mean() > base_f.loc[
        base_f["datetime"].isin(after), "failed_breakout_persistence_10"
    ].mean()
    assert stress_f.loc[stress_f["datetime"].isin(after), "up_day_leverage_shortfall_20"].mean() > base_f.loc[
        base_f["datetime"].isin(after), "up_day_leverage_shortfall_20"
    ].mean()


def test_fixed_bellwether_structure_uses_frozen_proxy_and_lags_one_session():
    from src.factor.soxl_exhaustion_structure import (
        FIXED_BELLWETHERS,
        STRUCTURE_FEATURES,
        build_fixed_bellwether_structure_frame,
    )

    panel = _bellwether_panel()
    dates = sorted(panel["datetime"].unique())
    soxx = pd.DataFrame(
        {"datetime": dates, "soxx_close": 100.0 * np.exp(np.cumsum(0.0005 + 0.004 * np.sin(np.arange(len(dates)) / 9.0)))}
    )
    frame = build_fixed_bellwether_structure_frame(panel, soxx, normalization_window=80, min_periods=30)
    assert tuple(FIXED_BELLWETHERS) == ("NVDA", "AVGO", "AMD", "TSM", "ASML")
    assert frame[STRUCTURE_FEATURES].notna().any().all()
    assert frame[STRUCTURE_FEATURES].apply(lambda values: values.dropna().between(0.0, 1.0).all()).all()
    assert frame.attrs["membership_contract"] == "fixed_bellwether_proxy_not_historical_constituents"

    cutoff = pd.Timestamp("2021-08-31")
    mutated = panel.copy()
    mutated.loc[mutated["datetime"] > cutoff, "close"] *= 10.0
    rebuilt = build_fixed_bellwether_structure_frame(mutated, soxx, normalization_window=80, min_periods=30)
    pdt.assert_frame_equal(
        frame.loc[frame["datetime"] <= cutoff, ["datetime", *STRUCTURE_FEATURES]].reset_index(drop=True),
        rebuilt.loc[rebuilt["datetime"] <= cutoff, ["datetime", *STRUCTURE_FEATURES]].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_leadership_concentration_remains_defined_during_synchronized_down_days():
    from src.factor.soxl_exhaustion_structure import build_fixed_bellwether_structure_frame

    periods = 240
    dates = pd.bdate_range("2021-01-04", periods=periods)
    common_ret = np.where(np.arange(periods) % 4 == 0, 0.006, -0.004)
    pieces = []
    for offset, symbol in enumerate(("NVDA", "AVGO", "AMD", "TSM", "ASML")):
        close = 100.0 * np.exp(np.cumsum(common_ret * (1.0 + offset * 0.05)))
        pieces.append(pd.DataFrame({"instrument": symbol, "datetime": dates, "close": close}))
    panel = pd.concat(pieces, ignore_index=True)
    soxx = pd.DataFrame({"datetime": dates, "soxx_close": 100.0 * np.exp(np.cumsum(common_ret))})
    frame = build_fixed_bellwether_structure_frame(panel, soxx, normalization_window=80, min_periods=30)
    assert frame["fixed_bellwether_leadership_concentration_20"].notna().sum() >= 100
