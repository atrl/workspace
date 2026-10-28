import numpy as np
import pandas as pd

from src.analysis.f696_joint_action_labels import _candidate_target
from src.data.early_volume_timing_residual import volume_timing_residual


def test_complementary_action_targets_preserve_frozen_constraints():
    binding = _candidate_target(np.array([.20, .25, .55]), binding=True, cap=.85)
    assert binding is not None and binding[0] == "binding_cash_to_soxl"
    np.testing.assert_allclose(binding[1].sum(), 1.)
    assert binding[1][0] <= .30 and binding[1][2] >= .15
    assert 3 * binding[1][0] + binding[1][1] <= .95 + 1e-10

    original = np.array([.20, .25, .55])
    rotated = _candidate_target(original, binding=False, cap=1.)
    assert rotated is not None and rotated[0] == "nonbinding_soxl_cash_to_soxx"
    np.testing.assert_allclose(rotated[1].sum(), 1.)
    np.testing.assert_allclose(3 * rotated[1][0] + rotated[1][1], 3 * original[0] + original[1])
    assert rotated[1][0] <= .30 and rotated[1][2] >= .15


def test_zero_volume_minutes_are_valid_when_session_has_trades():
    day = pd.Timestamp("2020-01-02")
    stamps = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=15, freq="1min")
    rows = []
    for symbol in ("SOXL", "SOXX"):
        for i, stamp in enumerate(stamps):
            rows.append((symbol, stamp, 100., 101. if i % 2 else 99., 0. if i == 0 else float(i)))
    frame = pd.DataFrame(rows, columns=["instrument", "datetime", "$open", "$close", "$volume"])
    frame = frame.set_index(["instrument", "datetime"])
    result = volume_timing_residual(frame, pd.DatetimeIndex([day, day + pd.Timedelta(days=1)]))
    assert result.source_known.tolist() == [True, False]
    assert np.isfinite(result.volume_timing_residual.iloc[0])
    assert pd.isna(result.volume_timing_residual.iloc[1])
