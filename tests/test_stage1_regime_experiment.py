import numpy as np
import pandas as pd
import pytest

from src.workflow.stage1_regime_experiment import development_gates, select_candidate, conditional_regime_metrics


def sample_tables():
    rows, annual = [], []
    for name in ("active_hold_baseline", "no_hold_reference", "candidate"):
        candidate = name == "candidate"
        for cost in (10., 25.):
            rows.append(dict(variant=name, cost_bps=cost, segment="development_walk_forward", return_pct=120. if candidate else 100., max_drawdown_pct=-20., worst_day_pct=-5., cvar5_pct=-3., worst_rolling_5d_pct=-8., open_trade_days=80 if candidate else 100, total_turnover=50 if candidate else 60))
            for year in (2017, 2018):
                annual.append(dict(variant=name, cost_bps=cost, segment="development_walk_forward", year=year, terminal_wealth=1.5 if candidate else 1.4))
    return pd.DataFrame(rows), pd.DataFrame(annual)


def test_tail_failure_cannot_be_overridden_by_aggregate_return_or_churn():
    comparison, annual = sample_tables()
    comparison.loc[comparison.variant.eq("candidate"), "worst_day_pct"] = -5.1
    gates = development_gates(comparison, annual[annual.cost_bps.eq(10)], ["candidate"])
    assert not gates.all_pass.any()
    selected, _ = select_candidate(gates, comparison, annual)
    assert not selected["historical_all_pass"]


def test_reused_outer_cannot_select_candidate_or_change_historical_gates():
    comparison, annual = sample_tables()
    gates = development_gates(comparison, annual[annual.cost_bps.eq(10)], ["candidate"])
    extra = comparison.assign(segment="outer_reused_diagnostic", return_pct=-99999.)
    got = development_gates(pd.concat([comparison, extra]), annual[annual.cost_bps.eq(10)], ["candidate"])
    pd.testing.assert_frame_equal(gates, got)
    assert select_candidate(got, pd.concat([comparison, extra]), annual)[0]["historical_all_pass"]


def test_missing_annual_coverage_blocks():
    comparison, annual = sample_tables()
    annual = annual[~(annual.variant.eq("candidate") & annual.year.eq(2018))]
    with pytest.raises(ValueError, match="coverage"):
        development_gates(comparison, annual[annual.cost_bps.eq(10)], ["candidate"])


def test_regime_drawdown_does_not_stitch_disjoint_days():
    dates = pd.bdate_range("2020-01-01", periods=3)
    replay = pd.DataFrame(dict(signal_date=dates, trade_date=dates, variant="x", segment="development_walk_forward", cost_bps=10, strategy_return=[-.1,.3,-.1], turnover=.1, execute_open_trade=True))
    feature = pd.DataFrame({"regime_up_probability":[1.,0.,1.], "regime_range_probability":[0.,1.,0.], "regime_down_probability":[0.,0.,0.]}, index=pd.DatetimeIndex(dates, name="signal_date"))
    result = conditional_regime_metrics(replay, feature).set_index("regime")
    assert result.loc["up", "episodes"] == 2
    assert np.isclose(result.loc["up", "worst_contiguous_episode_drawdown_pct"], -10.)
    assert np.isclose(result.loc["up", "conditional_compounded_return_pct"], -19.)
