from __future__ import annotations

import pandas as pd

from src.analysis import MarketRegimeConfig, attach_market_regimes, build_market_regime_frame


def _panel_with_precomputed_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=9, freq="D"),
            "close": [100.0, 103.0, 103.5, 104.0, 90.0, 96.0, 91.0, 91.5, 89.0],
            "volume": [100, 105, 102, 180, 120, 125, 130, 110, 115],
            "soxx_close": [50.0, 50.8, 51.0, 51.2, 48.0, 49.0, 48.5, 48.8, 48.1],
            "vix": [16.0, 18.0, 19.0, 22.0, 35.0, 24.0, 23.0, 18.0, 22.0],
            "trend": [0.050, 0.040, 0.006, 0.004, -0.120, 0.050, -0.050, 0.004, -0.050],
            "drawdown": [0.000, 0.000, -0.005, -0.010, -0.160, -0.060, -0.090, -0.005, -0.050],
            "realized_vol": [0.010, 0.015, 0.012, 0.020, 0.060, 0.055, 0.052, 0.010, 0.020],
            "volume_z": [0.0, 0.4, 0.1, 2.8, 0.6, 0.5, 0.4, 0.1, 0.2],
        }
    )


def test_build_market_regime_frame_classifies_core_market_states_with_review_columns():
    regimes = build_market_regime_frame(
        _panel_with_precomputed_signals(),
        config=MarketRegimeConfig(
            high_vix_threshold=30.0,
            low_vix_threshold=20.0,
            stress_drawdown_threshold=-0.10,
            uptrend_threshold=0.03,
            downtrend_threshold=-0.03,
            sideways_trend_abs_threshold=0.01,
            high_volatility_threshold=0.04,
            low_volatility_threshold=0.02,
            volume_shock_z_threshold=2.0,
        ),
    )

    assert regimes["schema_version"].unique().tolist() == ["analysis.market_regime.v1"]
    assert regimes["regime"].tolist() == [
        "bull",
        "bull",
        "sideways",
        "volume_shock",
        "stress",
        "high_vol_uptrend",
        "high_vol_downtrend",
        "sideways",
        "bear",
    ]
    assert {
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
    }.issubset(regimes.columns)

    stress = regimes.loc[regimes["regime"] == "stress"].iloc[0]
    assert "high_vix" in stress["regime_tags"]
    assert "deep_drawdown" in stress["regime_tags"]
    assert "压力" in stress["regime_reason"]

    volume_shock = regimes.loc[regimes["regime"] == "volume_shock"].iloc[0]
    assert "volume_shock" in volume_shock["regime_tags"]
    assert "成交量冲击" in volume_shock["regime_reason"]


def test_build_market_regime_frame_computes_missing_signals_and_aligns_to_calendar():
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=7, freq="D"),
            "close": [100.0, 101.0, 103.0, 102.0, 105.0, 99.0, 100.0],
            "volume": [100, 101, 102, 103, 180, 104, 105],
            "soxx_close": [50.0, 50.3, 51.0, 50.7, 51.2, 50.0, 50.4],
            "vix": [17.0, 18.0, 19.0, 21.0, 24.0, 33.0, 25.0],
        }
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))

    regimes = build_market_regime_frame(
        panel,
        calendar=calendar,
        config=MarketRegimeConfig(trend_window=3, volatility_window=3, drawdown_window=3, volume_window=3),
    )

    assert regimes["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-05", "2026-01-06"]
    assert regimes[["trend", "drawdown", "realized_vol", "volume_z", "relative_strength"]].notna().all().all()
    assert "stress" in set(regimes["regime"])
    assert regimes["source"].unique().tolist() == ["market_panel"]


def test_attach_market_regimes_merges_daily_backtest_without_overwriting_existing_labels_by_default():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "model": ["ridge_mse", "ridge_mse", "gcn_mse", "gcn_mse"],
            "nav": [1.0, 1.01, 1.0, 1.02],
            "regime": ["manual", "manual", "manual", "manual"],
        }
    )
    regime_frame = build_market_regime_frame(_panel_with_precomputed_signals().head(2))

    preserved = attach_market_regimes(daily, regime_frame)
    assert preserved["regime"].unique().tolist() == ["manual"]
    assert "market_regime" in preserved.columns
    assert preserved.loc[preserved["date"] == pd.Timestamp("2026-01-01"), "market_regime"].unique().tolist() == ["bull"]

    overwritten = attach_market_regimes(daily, regime_frame, overwrite=True)
    assert set(overwritten["regime"]) == {"bull"}
    assert "regime_reason" in overwritten.columns


def test_attach_market_regimes_can_refresh_existing_market_context_columns():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "model": ["ridge_mse", "ridge_mse"],
            "nav": [1.0, 1.01],
            "regime": ["legacy", "legacy"],
        }
    )
    first_regime_frame = build_market_regime_frame(_panel_with_precomputed_signals().head(2))
    first = attach_market_regimes(daily, first_regime_frame, overwrite=True)

    second_regime_frame = first_regime_frame.assign(regime="stress", regime_reason="刷新后的压力 regime")
    refreshed = attach_market_regimes(first, second_regime_frame, overwrite=True)

    assert refreshed["regime"].unique().tolist() == ["stress"]
    assert refreshed["market_regime"].unique().tolist() == ["stress"]
    assert refreshed["regime_reason"].unique().tolist() == ["刷新后的压力 regime"]
    assert not any(column.endswith("_x") or column.endswith("_y") for column in refreshed.columns)
