"""Numerical boundary correction tests; no fitted or historical policies."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import src.analysis.continuous_target_adapter as m


def _schedule(weights):
    frame = pd.DataFrame(weights, columns=list(m.TARGET_COLUMNS))
    frame["strategy_return"] = np.arange(len(frame)) * .001
    frame["opaque_metadata"] = [f"source-{i}" for i in range(len(frame))]
    return frame


def test_nonnegative_rows_are_bitwise_unchanged_and_input_not_mutated():
    source = _schedule([[0., 1., 0.], [.15, .8, .05], [np.nextafter(0., 1.), 1., 0.]])
    before = source.copy(deep=True)
    actual, audit = m.canonicalize_target_roundoff(source)
    pd.testing.assert_frame_equal(actual, before, check_exact=True)
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    assert audit["affected_rows"] == 0 and audit["maximum_absolute_weight_change"] == 0
    assert actual is not source


def test_only_negative_ulp_rows_are_normalized_and_other_columns_preserved():
    eps = np.finfo(float).eps
    source = _schedule([[.2, .7, .1], [0., 1. + eps, -eps]])
    before = source.copy(deep=True)
    actual, audit = m.canonicalize_target_roundoff(source)
    np.testing.assert_array_equal(actual.loc[0, list(m.TARGET_COLUMNS)].to_numpy(),
                                  before.loc[0, list(m.TARGET_COLUMNS)].to_numpy())
    np.testing.assert_array_equal(actual.loc[1, list(m.TARGET_COLUMNS)].to_numpy(), [0., 1., 0.])
    pd.testing.assert_frame_equal(actual.drop(columns=list(m.TARGET_COLUMNS)), before.drop(columns=list(m.TARGET_COLUMNS)))
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    assert audit["affected_rows"] == audit["negative_components"] == 1
    assert audit["maximum_absolute_weight_change"] == eps


@pytest.mark.parametrize("weight", [-1e-12, -1e-6, np.nextafter(-m.ROUNDOFF_BOUND, -np.inf)])
def test_material_negatives_fail_closed(weight):
    with pytest.raises(ValueError, match="exceeds eight"):
        m.canonicalize_target_roundoff(_schedule([[0., 1. - weight, weight]]))


def test_eight_epsilon_boundary_is_inclusive_but_bad_sum_or_larger_correction_fails():
    bound = m.ROUNDOFF_BOUND
    actual, audit = m.canonicalize_target_roundoff(_schedule([[0., 1. + bound, -bound]]))
    np.testing.assert_array_equal(actual[list(m.TARGET_COLUMNS)].to_numpy(), [[0., 1., 0.]])
    assert audit["maximum_absolute_weight_change"] == bound
    with pytest.raises(ValueError, match="target error exceeds"):
        m.canonicalize_target_roundoff(_schedule([[0., 1. + 4 * bound, -bound]]))
    with pytest.raises(ValueError, match="target correction exceeds"):
        m.canonicalize_target_roundoff(_schedule([[0., 1. + 2 * bound, -bound]]))


def test_wrapper_delegates_copy_without_changing_research_returns(monkeypatch):
    source = _schedule([[0., 1., -np.finfo(float).eps / 4]])
    before = source.copy(deep=True)
    observed = {}
    def fake(frame, schedule, **kwargs):
        observed["schedule"] = schedule
        assert schedule[m.TARGET_COLUMNS[2]].iloc[0] == 0
        return {"audit": {"status": "passed", "tolerance": 1e-7}, "reconciliation": "native"}
    monkeypatch.setattr(m, "_sealed_native_open_reconciliation", fake)
    result = m.native_open_reconciliation(None, source, provider_uri="unused")
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_series_equal(observed["schedule"].strategy_return, before.strategy_return)
    assert result["audit"]["tolerance"] == 1e-7
    assert result["audit"]["target_roundoff_adapter"]["affected_rows"] == 1


def test_native_qlib_reconciles_synthetic_null_switch_without_changing_replay(tmp_path):
    from qlib.data import D
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.model.benchmark_anchor_utility import rollout_alphas
    from src.analysis.integrated_allocation import native_allocation_reconciliation
    from src.analysis.convex_qlib import native_open_reconciliation as old_native

    dates = pd.bdate_range("2020-01-02", periods=6)
    records = []
    for symbol, values in (("SOXL", [100., 100., 100., 101., 100., 101.]),
                           ("SOXX", [100., 100., 100., 100.3, 100.1, 100.5])):
        for date, opening in zip(dates, values):
            records.append({"instrument": symbol, "datetime": date, "open": opening, "high": opening * 1.01,
                "low": opening * .99, "close": opening, "volume": 100000.})
    provider = tmp_path / "provider"
    build_qlib_store_from_frame(pd.DataFrame(records), provider_uri=provider, as_of=str(dates[-1].date()))
    opens = D.features(["SOXL", "SOXX"], ["$open"], start_time=dates[0], end_time=dates[-1])["$open"].unstack("instrument").astype(float)
    returns = opens.reindex(dates[2:5]).to_numpy() / opens.reindex(dates[1:4]).to_numpy() - 1
    states = rollout_alphas(np.array([[.15, .05, 0.], [0., 0., 0.], [0., 0., 0.]]), returns, cost_bps=10.)
    market = pd.DataFrame({"signal_date": dates[:3], "trade_date": dates[1:4], "exit_date": dates[2:5],
        "soxl_return": returns[:, 0], "soxx_return": returns[:, 1]})
    path = market[["signal_date", "trade_date", "exit_date"]].copy()
    path["strategy_return"] = states["net_return"]
    path["fees"], path["turnover"] = states["fees"], states["turnover"]
    path["cost_bps"], path["execute_open_target"] = 10., True
    for position, asset in enumerate(("soxl", "soxx", "cash")):
        path[f"{asset}_weight"] = states["open_net_weights"][:, position]
    assert path.cash_weight.min() < 0
    before = deepcopy(path)
    with pytest.raises(ValueError, match="invalid continuous Qlib net targets"):
        native_allocation_reconciliation(market, path, provider_uri=provider, cost_bps=10., native_reconciler=old_native)
    result = native_allocation_reconciliation(market, path, provider_uri=provider, cost_bps=10.,
                                             native_reconciler=m.native_open_reconciliation)
    assert result["audit"]["status"] == "passed"
    assert result["audit"]["maximum_session_return_error"] < 1e-12
    assert result["audit"]["tolerance"] == 1e-7
    assert result["audit"]["target_roundoff_adapter"]["affected_rows"] > 0
    pd.testing.assert_frame_equal(path, before, check_exact=True)
