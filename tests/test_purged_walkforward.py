from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.purged_walkforward import (
    build_purged_walkforward_splits,
    single_position_frame,
)


def test_purged_walkforward_split_excludes_rows_with_forward_label_leakage():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=10, freq="B"),
            "value": range(10),
        }
    )

    splits = build_purged_walkforward_splits(
        frame,
        {"window": ("2026-01-08", "2026-01-09")},
        purge_rows=2,
    )

    split = splits[0]
    assert split.name == "window"
    assert split.train["value"].tolist() == [0, 1, 2]
    assert split.test["value"].tolist() == [5, 6]
    assert split.train_end.date().isoformat() == "2026-01-05"
    assert split.test_start.date().isoformat() == "2026-01-08"


def test_purged_walkforward_split_drops_windows_without_enough_train_rows():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=8, freq="B"),
            "value": range(8),
        }
    )

    splits = build_purged_walkforward_splits(
        frame,
        {"early": ("2026-01-05", "2026-01-06")},
        purge_rows=2,
        min_train_rows=2,
    )

    assert splits == []


def test_single_position_frame_rejects_duplicate_position_columns():
    frame = pd.DataFrame(
        [
            ["2026-01-01", 0.1, 0.2],
            ["2026-01-02", 0.3, 0.4],
        ],
        columns=["datetime", "position", "position"],
    )

    try:
        single_position_frame(frame)
    except ValueError as exc:
        assert "duplicate `position`" in str(exc)
    else:
        raise AssertionError("duplicate position columns did not raise ValueError")


def test_single_position_frame_keeps_only_datetime_and_requested_position():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=2, freq="B"),
            "position": [0.1, 0.2],
            "oracle_position": [0.8, 1.2],
            "feature": [5.0, 6.0],
        }
    )

    out = single_position_frame(frame, position_col="oracle_position")

    assert out.columns.tolist() == ["datetime", "position"]
    assert out["position"].tolist() == [0.8, 1.0]


if __name__ == "__main__":
    test_purged_walkforward_split_excludes_rows_with_forward_label_leakage()
    test_purged_walkforward_split_drops_windows_without_enough_train_rows()
    test_single_position_frame_rejects_duplicate_position_columns()
    test_single_position_frame_keeps_only_datetime_and_requested_position()
    print("test_purged_walkforward: OK")
