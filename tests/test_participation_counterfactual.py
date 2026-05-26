from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.participation_counterfactual import (
    ReentryParticipationOverlayConfig,
    TrendParticipationGuardConfig,
    apply_reentry_participation_overlay,
    apply_trend_participation_guard_overlay,
)


def _feature_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": dates,
            "mom_21": [0.18, 0.16, 0.20],
            "relative_strength_21": [1.20, 1.10, 1.30],
            "vol_of_vol_21": [0.16, 0.15, 0.18],
            "tracking_error_vol_21": [0.13, 0.12, 0.14],
            "vix_percentile_252": [0.70, 0.72, 0.65],
        }
    )


def test_reentry_participation_overlay_lifts_low_and_half_positions_only():
    dates = pd.date_range("2025-04-01", periods=3, freq="B")
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "position": [0.20, 0.52, 0.70],
            "price": [100.0, 103.0, 107.0],
        }
    )

    out = apply_reentry_participation_overlay(
        daily,
        _feature_frame(dates),
        ReentryParticipationOverlayConfig(
            enabled=True,
            min_target=0.60,
            max_target=0.65,
            max_base_position=0.55,
            min_tracking_error_vol_21=0.10,
            max_equity_drawdown=None,
        ),
    )

    assert out["base_position"].round(4).tolist() == [0.20, 0.52, 0.70]
    assert out["reentry_participation_ready"].tolist() == [True, True, False]
    assert out["position"].round(4).tolist() == [0.60, 0.60, 0.70]


def test_reentry_participation_overlay_blocks_lift_after_equity_drawdown():
    dates = pd.date_range("2025-04-01", periods=3, freq="B")
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "position": [0.0, 0.0, 0.0],
            "price": [100.0, 80.0, 82.0],
        }
    )

    out = apply_reentry_participation_overlay(
        daily,
        _feature_frame(dates),
        ReentryParticipationOverlayConfig(
            enabled=True,
            min_target=0.80,
            max_target=0.80,
            max_base_position=0.10,
            max_equity_drawdown=0.10,
        ),
    )

    assert out["equity_drawdown"].round(4).tolist() == [0.0, 0.16, 0.16]
    assert out["reentry_participation_ready"].tolist() == [True, False, False]
    assert out["position"].round(4).tolist() == [0.80, 0.0, 0.0]


def test_reentry_participation_overlay_requires_configured_feature_columns():
    dates = pd.date_range("2025-04-01", periods=3, freq="B")
    daily = pd.DataFrame({"datetime": dates, "position": [0.2, 0.2, 0.2], "price": [100, 101, 102]})
    features = _feature_frame(dates).drop(columns=["tracking_error_vol_21"])

    try:
        apply_reentry_participation_overlay(
            daily,
            features,
            ReentryParticipationOverlayConfig(
                enabled=True,
                min_tracking_error_vol_21=0.10,
                max_equity_drawdown=None,
            ),
        )
    except ValueError as exc:
        assert "tracking_error_vol_21" in str(exc)
    else:
        raise AssertionError("missing tracking_error_vol_21 did not raise ValueError")


def _trend_feature_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": dates,
            "anchor_gap_180": [0.60, 0.70, 0.20, 0.80],
            "mom_63": [0.50, 0.55, 0.60, 0.70],
            "trend_slope_63": [0.12, 0.16, 0.18, 0.20],
            "trend_gap_63": [0.14, 0.18, 0.22, 0.25],
            "vix_percentile_252": [0.25, 0.35, 0.30, 0.75],
            "realized_vol_63": [0.50, 0.55, 0.60, 0.65],
            "drawdown_63": [-0.08, -0.10, -0.12, -0.14],
        }
    )


def test_trend_participation_guard_overlay_lifts_high_trend_low_risk_rows_and_recomputes_nav():
    dates = pd.date_range("2024-04-01", periods=4, freq="B")
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "position": [0.10, 0.20, 0.10, 0.10],
            "strategy_nav": [100.0, 101.0, 102.0, 103.0],
            "buyhold_nav": [100.0, 110.0, 121.0, 108.9],
        }
    )

    out = apply_trend_participation_guard_overlay(
        daily,
        _trend_feature_frame(dates),
        TrendParticipationGuardConfig(
            enabled=True,
            min_target=0.50,
            max_target=0.60,
            max_base_position=0.30,
            min_anchor_gap_180=0.50,
            min_mom_63=0.45,
            min_trend_slope_63=0.10,
            min_trend_gap_63=0.10,
            max_vix_percentile_252=0.40,
            max_realized_vol_63=0.70,
            min_drawdown_63=-0.25,
            max_equity_drawdown=None,
        ),
    )

    assert out["base_position"].round(4).tolist() == [0.10, 0.20, 0.10, 0.10]
    assert out["trend_participation_ready"].tolist() == [True, True, False, False]
    assert out["trend_participation_lifted"].tolist() == [True, True, False, False]
    assert out["position"].round(4).tolist() == [0.50, 0.50, 0.10, 0.10]
    assert round(float(out["strategy_nav"].iloc[-1]), 4) == 109.1475
    assert out["base_strategy_nav"].round(4).tolist() == [100.0, 101.0, 102.0, 103.0]


def test_trend_participation_guard_overlay_blocks_lift_after_equity_drawdown():
    dates = pd.date_range("2024-04-01", periods=3, freq="B")
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "position": [0.0, 0.0, 0.0],
            "strategy_nav": [100.0, 100.0, 100.0],
            "buyhold_nav": [100.0, 80.0, 82.0],
        }
    )
    features = _trend_feature_frame(pd.date_range("2024-04-01", periods=4, freq="B")).iloc[:3].copy()

    out = apply_trend_participation_guard_overlay(
        daily,
        features,
        TrendParticipationGuardConfig(
            enabled=True,
            min_target=0.80,
            max_target=0.80,
            max_base_position=0.10,
            min_anchor_gap_180=0.50,
            min_mom_63=0.45,
            min_trend_slope_63=0.10,
            min_trend_gap_63=0.10,
            max_vix_percentile_252=0.40,
            max_realized_vol_63=0.70,
            min_drawdown_63=-0.25,
            max_equity_drawdown=0.10,
        ),
    )

    assert out["trend_participation_equity_allowed"].tolist() == [True, False, False]
    assert out["trend_participation_ready"].tolist() == [True, False, False]
    assert out["position"].round(4).tolist() == [0.80, 0.0, 0.0]
    assert out["equity_drawdown"].round(4).tolist() == [0.0, 0.16, 0.16]


def test_trend_participation_guard_overlay_requires_configured_feature_columns():
    dates = pd.date_range("2024-04-01", periods=4, freq="B")
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "position": [0.10, 0.10, 0.10, 0.10],
            "strategy_nav": [100.0, 101.0, 102.0, 103.0],
            "buyhold_nav": [100.0, 110.0, 121.0, 108.9],
        }
    )

    try:
        apply_trend_participation_guard_overlay(
            daily,
            _trend_feature_frame(dates).drop(columns=["anchor_gap_180"]),
            TrendParticipationGuardConfig(enabled=True),
        )
    except ValueError as exc:
        assert "anchor_gap_180" in str(exc)
    else:
        raise AssertionError("missing anchor_gap_180 did not raise ValueError")


if __name__ == "__main__":
    test_reentry_participation_overlay_lifts_low_and_half_positions_only()
    test_reentry_participation_overlay_blocks_lift_after_equity_drawdown()
    test_reentry_participation_overlay_requires_configured_feature_columns()
    test_trend_participation_guard_overlay_lifts_high_trend_low_risk_rows_and_recomputes_nav()
    test_trend_participation_guard_overlay_blocks_lift_after_equity_drawdown()
    test_trend_participation_guard_overlay_requires_configured_feature_columns()
    print("test_participation_counterfactual: OK")
