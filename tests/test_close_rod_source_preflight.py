import pickle

import numpy as np
import pandas as pd

from src.workflow.close_rod_source_preflight import _requirements, _native_endpoints


def test_halfday_endpoints_and_float64_native_dataset_roundtrip(tmp_path):
    sessions = pd.DataFrame({
        "previous_actual_close_at": pd.to_datetime(["2024-07-02 16:00", "2024-07-03 13:00"]),
        "actual_close_at": pd.to_datetime(["2024-07-03 13:00", "2024-07-05 16:00"]),
        "regular_session": [False, True],
    }, index=pd.to_datetime(["2024-07-03", "2024-07-05"]))
    required = _requirements(sessions, pd.Timestamp("2024-07-08"))
    half = required.loc[required.trade_date.eq(pd.Timestamp("2024-07-03"))]
    assert not half.slot.str.startswith("late_").any()
    regular = required.loc[required.trade_date.eq(pd.Timestamp("2024-07-05"))]
    assert regular.loc[regular.slot.eq("previous_actual_close"), "bar_end"].eq(pd.Timestamp("2024-07-03 13:00")).all()
    assert regular.loc[regular.slot.eq("late_entry"), "bar_end"].eq(pd.Timestamp("2024-07-05 15:33")).all()
    assert regular.loc[regular.slot.eq("late_entry"), "price_stamp"].eq(pd.Timestamp("2024-07-05 15:32")).all()
    assert regular.loc[regular.slot.eq("late_exit"), "bar_end"].eq(pd.Timestamp("2024-07-05 16:00")).all()
    frames = {}
    for symbol in ("SOXL", "SOXX"):
        stamps = pd.DatetimeIndex(required.loc[required.instrument.eq(symbol), "bar_end"].unique()).sort_values()
        frames[symbol] = pd.DataFrame({"timestamp": stamps.tz_localize("America/New_York"),
            "open": 100.123456, "high": 101.123456, "low": 99.123456, "close": 100.654321,
            "price_valid": True, "is_final": True})
    panel, status, manifest, dataset, precision = _native_endpoints(tmp_path,
        {"provider_output": "provider", "as_of": "2024-07-08"}, required, frames)
    assert status.valid.all() and manifest["status"] == "ok"
    assert precision.max_absolute_representation_error.gt(0).any()
    restored = pickle.loads(pickle.dumps(dataset)).prepare("all", col_set="feature")
    assert all(dtype == np.dtype("float64") for dtype in restored.dtypes)
    pd.testing.assert_frame_equal(restored, dataset.prepare("all", col_set="feature"))
