from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.barrier_labels import (
    BarrierActionLabelConfig,
    TripleBarrierConfig,
    build_barrier_action_label_frame,
    compute_triple_barrier_outcomes,
)


def test_triple_barrier_outcomes_use_first_hit_order():
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=5, freq="B"),
            "price": [100.0, 96.0, 106.0, 104.0, 110.0],
        }
    )

    out = compute_triple_barrier_outcomes(
        daily,
        TripleBarrierConfig(horizon_days=2, upper_return=0.05, lower_return=0.03),
    )

    assert out["barrier_label"].tolist() == [-1, 1, 0, 0, 0]
    assert out["barrier_hit"].tolist()[:3] == ["lower", "upper", "none"]
    assert out["hit_offset"].tolist()[:2] == [1, 1]
    assert out["has_forward_window"].tolist() == [True, True, True, False, False]


def test_barrier_action_labels_stage_reentry_and_derisk_actions():
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=5, freq="B"),
            "position": [0.20, 0.70, 0.10, 0.20, 0.20],
            "price": [100.0, 106.0, 100.0, 94.0, 101.0],
        }
    )

    labels = build_barrier_action_label_frame(
        daily,
        BarrierActionLabelConfig(
            barrier=TripleBarrierConfig(horizon_days=2, upper_return=0.05, lower_return=0.04),
            max_reentry_position=0.35,
            min_derisk_position=0.50,
            reentry_position=0.75,
            derisk_position=0.0,
        ),
    )

    assert labels["barrier_label"].tolist() == [1, -1, -1, 0, 0]
    assert labels["reentry_label"].tolist() == [True, False, False, False, False]
    assert labels["derisk_label"].tolist() == [False, True, False, False, False]
    assert labels["action_label"].tolist() == [1, -1, 0, 0, 0]
    assert labels["expert_position"].round(4).tolist() == [0.75, 0.0, 0.10, 0.20, 0.20]


def test_barrier_action_labels_require_position_and_price():
    daily = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=3), "price": [1.0, 1.1, 1.2]})

    try:
        build_barrier_action_label_frame(daily, BarrierActionLabelConfig())
    except ValueError as exc:
        assert "position" in str(exc)
    else:
        raise AssertionError("missing position did not raise ValueError")


if __name__ == "__main__":
    test_triple_barrier_outcomes_use_first_hit_order()
    test_barrier_action_labels_stage_reentry_and_derisk_actions()
    test_barrier_action_labels_require_position_and_price()
    print("test_barrier_labels: OK")
