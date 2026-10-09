from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.analysis.allocation_attribution import attribute_allocations
from src.backtest.benchmark_relative_replay import replay_fixed_target


def fixture():
    calendar = pd.bdate_range("2020-01-02", periods=8)
    frame = pd.DataFrame({"signal_date": calendar[:-2], "trade_date": calendar[1:-1],
        "exit_date": calendar[2:], "soxl_return": [.03, -.05, .015, .04, -.01, .02],
        "soxx_return": [.01, -.018, .005, .013, -.003, .008]})
    frame["decision_at"] = frame.signal_date.dt.tz_localize("America/New_York")+pd.Timedelta(hours=16, minutes=15)
    frame["return_available_at"] = frame.exit_date.dt.tz_localize("America/New_York")+pd.Timedelta(hours=9, minutes=31)
    paths = {}
    for cost in (10., 25.):
        for name, weights in (("soxx", [0, 1, 0]), ("candidate", [.3, .4, .3])):
            paths[name, cost] = replay_fixed_target(frame, target=weights, cost_bps=cost, rebalance_sessions=2)
    return frame, paths, {"full": ("2020-01-02", "2020-01-09")}


def test_exact_cost_and_gross_log_decomposition_without_mutating_paths():
    frame, paths, windows = fixture()
    before = deepcopy(paths)
    out = attribute_allocations(paths, frame, windows=windows)
    assert len(out) == 4
    for row in out.itertuples():
        p, b = paths[row.strategy, row.cost_bps], paths["soxx", row.cost_bps]
        expected = np.prod(1+p.strategy_return)/np.prod(1+b.strategy_return)
        assert row.relative_wealth == pytest.approx(expected, abs=1e-14)
        assert row.net_log_excess == pytest.approx(row.fixed_positions_gross_log_excess+row.incremental_fee_log_drag, abs=1e-14)
        assert row.net_simple_difference_bp == pytest.approx(row.beta_level_simple_bp+row.beta_timing_simple_bp+row.leveraged_residual_simple_bp+row.fees_simple_bp, abs=1e-10)
        pd.testing.assert_frame_equal(p, before[row.strategy, row.cost_bps], check_exact=True)
    assert out.loc[out.strategy.eq("soxx"), "net_log_excess"].eq(0).all()


@pytest.mark.parametrize("damage", ["fees", "dates", "missing_cost", "missing_benchmark", "nan", "zero_variance"])
def test_rejects_broken_evidence(damage):
    frame, paths, windows = fixture()
    if damage == "fees": paths["candidate", 10.].loc[0, "fees"] += .001
    if damage == "dates": paths["candidate", 10.].loc[0, "signal_date"] -= pd.Timedelta(days=1)
    if damage == "missing_cost": del paths["candidate", 25.]
    if damage == "missing_benchmark": paths = {key: value for key, value in paths.items() if key[0] != "soxx"}
    if damage == "nan": frame.loc[0, "soxl_return"] = np.nan
    if damage == "zero_variance":
        # Reconcile the constant market first; do not fail only on stale NAV.
        frame["soxx_return"] = 0.
        paths = {(name, cost): replay_fixed_target(frame, target=weights, cost_bps=cost, rebalance_sessions=2)
            for cost in (10., 25.) for name, weights in (("soxx", [0, 1, 0]), ("candidate", [.3, .4, .3]))}
    with pytest.raises(ValueError):
        attribute_allocations(paths, frame, windows=windows)


def test_empty_window_is_not_silently_dropped():
    frame, paths, _ = fixture()
    with pytest.raises(ValueError, match="three observations"):
        attribute_allocations(paths, frame, windows={"missing": ("2001-01-01", "2001-02-01")})
