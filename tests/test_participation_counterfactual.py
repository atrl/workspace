from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.participation_counterfactual import (
    ReentryParticipationOverlayConfig,
    apply_reentry_participation_overlay,
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


if __name__ == "__main__":
    test_reentry_participation_overlay_lifts_low_and_half_positions_only()
    test_reentry_participation_overlay_blocks_lift_after_equity_drawdown()
    test_reentry_participation_overlay_requires_configured_feature_columns()
    print("test_participation_counterfactual: OK")
