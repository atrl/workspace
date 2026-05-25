from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.reentry_labels import (
    ReentryOracleLabelConfig,
    build_reentry_oracle_label_frame,
)


def test_reentry_oracle_labels_lift_only_low_base_safe_forward_rebounds():
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "position": [0.20, 0.20, 0.70, 0.20, 0.20],
            "price": [100.0, 110.0, 120.0, 115.0, 130.0],
        }
    )

    labels = build_reentry_oracle_label_frame(
        daily,
        ReentryOracleLabelConfig(
            horizon_days=2,
            max_base_position=0.35,
            target_position=0.65,
            min_forward_return=0.15,
            max_forward_drawdown=0.05,
        ),
    )

    assert labels["oracle_reentry"].tolist() == [True, False, False, False, False]
    assert labels["expert_position"].round(4).tolist() == [0.65, 0.20, 0.70, 0.20, 0.20]
    assert labels["forward_return"].round(4).tolist()[:2] == [0.20, 0.0455]
    assert labels["forward_drawdown"].round(4).tolist()[:2] == [0.0, 0.0]


def test_reentry_oracle_labels_block_high_forward_drawdown_even_if_return_recovers():
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "position": [0.0, 0.0, 0.0],
            "price": [100.0, 90.0, 130.0],
        }
    )

    labels = build_reentry_oracle_label_frame(
        daily,
        ReentryOracleLabelConfig(
            horizon_days=2,
            target_position=0.80,
            min_forward_return=0.20,
            max_forward_drawdown=0.05,
        ),
    )

    assert round(float(labels.loc[0, "forward_return"]), 4) == 0.30
    assert round(float(labels.loc[0, "forward_drawdown"]), 4) == 0.10
    assert labels["oracle_reentry"].tolist() == [False, False, False]
    assert labels["expert_position"].tolist() == [0.0, 0.0, 0.0]


def test_reentry_oracle_labels_require_replay_columns():
    daily = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=3), "position": [0.0, 0.0, 0.0]})

    try:
        build_reentry_oracle_label_frame(daily, ReentryOracleLabelConfig())
    except ValueError as exc:
        assert "price" in str(exc)
    else:
        raise AssertionError("missing price did not raise ValueError")


if __name__ == "__main__":
    test_reentry_oracle_labels_lift_only_low_base_safe_forward_rebounds()
    test_reentry_oracle_labels_block_high_forward_drawdown_even_if_return_recovers()
    test_reentry_oracle_labels_require_replay_columns()
    print("test_reentry_labels: OK")
