import numpy as np
import pandas as pd
import pytest

from src.analysis.soxl_historical_analog import (
    _cohort, cluster_nonoverlap, forward_outcomes, shape_features, state_features,
)


def _market(rows=100):
    values = np.exp(np.linspace(0, .2, rows) + .03 * np.sin(np.arange(rows) / 4))
    return pd.DataFrame({"soxx_open": 100 * values, "soxx_close": 101 * values,
                         "soxl_open": 50 * values ** 2, "soxl_close": 51 * values ** 2,
                         "vix": 20 + 2 * np.cos(np.arange(rows) / 5)},
                        index=pd.bdate_range("2020-01-01", periods=rows))


def test_shape_keeps_first_open_to_first_close_drop():
    market = _market(44)
    market[["soxx_open", "soxx_close"]] = 90.0
    market.iloc[0, market.columns.get_loc("soxx_open")] = 100.0
    _, amplitude, details = shape_features(market, 43, 44)
    assert details["soxx_window_return_pct"] == pytest.approx(-10.0)
    assert details["soxx_window_max_drawdown_pct"] == pytest.approx(-10.0)
    assert amplitude[0] == pytest.approx(np.log(.9))


def test_state_distance_inputs_ignore_future_and_soxl_path():
    market = _market()
    before, blocks = state_features(market, 70, (20, 60))
    changed = market.copy()
    changed.loc[changed.index[71]:, :] *= 5
    changed["soxl_close"] *= 3
    after, _ = state_features(changed, 70, (20, 60))
    np.testing.assert_array_equal(before, after)
    assert blocks == [2, 4, 2, 2]
    # The 60-day return really needs the anchor-minus-60 close.
    changed.loc[changed.index[10], "soxx_close"] *= 1.1
    after_boundary, _ = state_features(changed, 70, (20, 60))
    assert after_boundary[1] != before[1]


def test_clusters_and_cohort_purge_use_full_support_not_anchor_only():
    frame = pd.DataFrame({
        "anchor_date": pd.to_datetime(["2023-03-01", "2023-04-01", "2023-09-01"]),
        "support_start": pd.to_datetime(["2022-12-01", "2023-02-01", "2023-07-01"]),
        "support_end": pd.to_datetime(["2023-04-01", "2023-05-01", "2023-10-01"]),
        "distance": [.1, .2, .3],
    })
    clustered = cluster_nonoverlap(frame, "test")
    assert clustered.representative.tolist() == [True, False, True]
    assert clustered.event_cluster_id.iloc[0] == clustered.event_cluster_id.iloc[1]
    assert len(_cohort(frame, "untrained_2023_2025")) == 2


def test_incomplete_target_forward20_is_not_filled_from_18_sessions():
    outcome = forward_outcomes(_market(19), 0)
    assert outcome["forward_18_complete"] is True
    assert outcome["forward_20_complete"] is False
    assert outcome["forward_20_observed_sessions"] == 18
    assert outcome["soxx_forward_20_return_pct"] is None
