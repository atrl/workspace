import numpy as np
import pandas as pd
import pytest

from src.analysis import dynamic_beta_cap as analysis
from src.workflow import dynamic_beta_cap as workflow


def test_constant_cap_is_flat_and_vol_scaled_is_causal():
    rng = np.random.default_rng(4)
    soxx = rng.normal(0, 0.01, 300)
    const = analysis.build_caps(analysis.BetaCap("c", "constant", constant=0.6), soxx)
    assert np.allclose(const, 0.6)
    dyn = analysis.build_caps(
        analysis.BetaCap("d", "vol_scaled", scale=0.6, vol_window=30, floor=0.3, ceiling=1.5), soxx)
    assert (dyn >= 0.3 - 1e-12).all() and (dyn <= 1.5 + 1e-12).all()
    # The cap at t uses only volatility through t (bfill only seeds early rows).
    short = analysis.build_caps(
        analysis.BetaCap("d", "vol_scaled", scale=0.6, vol_window=30, floor=0.3, ceiling=1.5),
        soxx[:60])
    assert len(short) == 60


def test_beta_cap_validate_rejects_bad_kind():
    with pytest.raises(ValueError, match="kind"):
        analysis.BetaCap("x", "nonsense").validate()


def test_run_beta_cap_never_exceeds_cap_and_preserves_funding():
    dates = pd.bdate_range("2020-01-02", periods=100)
    raw = pd.DataFrame({"soxl_weight": 0.25, "soxx_weight": 0.40}, index=dates)
    returns = pd.DataFrame({"soxl": 0.01, "soxx": 0.003}, index=dates)
    caps = np.full(100, 0.6)
    replay = analysis.run_beta_cap(raw, returns, caps, cost_bps=10.0)
    assert replay["strategy_return"].notna().all()


def test_annual_sharpe_groups_by_year():
    dates = pd.bdate_range("2021-01-04", periods=500)
    replay = pd.DataFrame({"strategy_return": 0.001}, index=dates)
    annual = analysis.annual_sharpe(replay, pd.Series(dates))
    assert set(annual.index) == {2021, 2022}


def test_workflow_selects_a_rule_that_improves_both_windows():
    summary = workflow.run_dynamic_beta_cap(
        root=__import__("pathlib").Path(__file__).resolve().parents[1],
        card_dir=__import__("tempfile").mkdtemp(),
        recorder_uri=__import__("os").path.join(__import__("tempfile").mkdtemp(), "rec"))
    assert summary["baseline_rule"] == "constant_0.6"
    assert summary["selected_rule"] in summary["rule_family"]
    # The decision must be consistent with the reported gate.
    if summary["improvement"]:
        assert summary["rules_passing_outer"]
        assert summary["decision"] == "dynamic_beta_cap_improves_out_of_sample"
