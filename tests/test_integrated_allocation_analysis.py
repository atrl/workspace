from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.integrated_allocation import (
    StabilityGateSpec, annual_paired_metrics, native_allocation_reconciliation,
    paired_metrics, seed_dispersion, stable_excess_gates, stationary_bootstrap_excess,
    validate_replay,
)


def _path(dates, returns, cost=10.):
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
        "exit_date": dates + pd.offsets.BDay(2), "strategy_return": np.broadcast_to(returns, len(dates)),
        "turnover": 0., "fees": 0., "cost_bps": cost})


def _annual_paths(log_excess=.0003):
    dates = pd.bdate_range("2016-01-04", "2022-12-30")
    baseline = np.sin(np.arange(len(dates))) * .004
    return ({cost: _path(dates, np.expm1(np.log1p(baseline) + log_excess), cost) for cost in (10., 25.)},
            {cost: _path(dates, baseline, cost) for cost in (10., 25.)})


def _seed_table():
    return pd.DataFrame([{"seed": seed, "cost_bps": cost, "log_excess_sum": .1}
                         for seed in (17, 41, 71) for cost in (10., 25.)])


def test_paired_bootstrap_keeps_shared_market_path_and_is_deterministic():
    dates = pd.bdate_range("2020-01-02", periods=160)
    benchmark = _path(dates, np.sin(np.arange(160)) * .1)
    candidate = benchmark.copy()
    candidate["strategy_return"] = np.expm1(np.log1p(benchmark.strategy_return) + .0003)
    a = stationary_bootstrap_excess(candidate, benchmark, mean_block=20, resamples=250)
    b = stationary_bootstrap_excess(candidate, benchmark, mean_block=20, resamples=250)
    assert a == b
    assert a["daily_lower_bound"] == pytest.approx(.0003, abs=1e-14)
    assert a["daily_upper_bound"] == pytest.approx(.0003, abs=1e-14)
    zero = stationary_bootstrap_excess(benchmark, benchmark, mean_block=20, resamples=250)
    assert zero["daily_lower_bound"] == 0.
    assert zero["lower_bound_positive"] is False


def test_first_day_loss_is_included_in_initial_capital_drawdown():
    dates = pd.bdate_range("2020-01-02", periods=6)
    candidate, benchmark = _path(dates, [-.2, 0, 0, 0, 0, 0]), _path(dates, 0.)
    metrics = paired_metrics(candidate, benchmark)
    assert metrics["max_drawdown_pct"] == pytest.approx(-20.)
    assert metrics["relative_max_drawdown_pct"] == pytest.approx(-20.)
    assert metrics["qlib_max_drawdown_without_initial_mark_pct"] == 0.
    assert metrics["cagr_pct"] == pytest.approx(100 * (.8 ** (252 / 6) - 1))


def test_cvar_does_not_dilute_a_cash_path_loss_across_all_zero_days():
    dates = pd.bdate_range("2020-01-02", periods=100)
    candidate = _path(dates, np.r_[-.2, np.zeros(99)])
    metrics = paired_metrics(candidate, _path(dates, 0.))
    assert metrics["cvar5_pct"] == pytest.approx(-4.)


def test_coverage_nonfinite_and_fee_fail_closed():
    candidate, benchmark = _annual_paths()
    with pytest.raises(ValueError, match="noncontiguous"):
        paired_metrics(candidate[10.].drop(index=5), benchmark[10.])
    with pytest.raises(ValueError, match="identical coverage"):
        paired_metrics(candidate[10.].iloc[:-1], benchmark[10.])
    broken = candidate[10.].copy()
    broken.loc[0, "strategy_return"] = np.nan
    with pytest.raises(ValueError, match="invalid replay"):
        validate_replay(broken)
    broken = candidate[10.].copy()
    broken.loc[0, "fees"] = .01
    with pytest.raises(ValueError, match="cash fees"):
        validate_replay(broken)


def test_single_crisis_cannot_qualify_stable_excess():
    dates = pd.bdate_range("2016-01-04", "2022-12-30")
    excess = np.where(dates.year == 2020, .01, -.0001)
    candidate, benchmark = _annual_paths(excess)
    result = stable_excess_gates(candidate, benchmark,
        spec=replace(StabilityGateSpec(), bootstrap_resamples=200), seed_table=_seed_table())
    assert result["gates"].aggregate_positive_net_log_excess.all()
    assert not result["gates"].minimum_positive_signal_years.any()
    assert not result["gates"].leave_any_one_year_out_log_excess_positive.any()
    assert not result["historical_qualification_pass"]


def test_history_pass_stays_separate_from_forward_and_uses_candidate_family_adjustment():
    candidate, benchmark = _annual_paths()
    result = stable_excess_gates(candidate, benchmark,
        spec=replace(StabilityGateSpec(), bootstrap_resamples=200), seed_table=_seed_table())
    assert result["historical_qualification_pass"]
    assert not result["new_forward_qualified"] and not result["promotion_ready"]
    assert set(result["bootstrap"].mean_block) == {20, 60}
    assert np.allclose(result["bootstrap"].alpha, .05 / 3)
    assert set(result["annual"].year) == set(range(2016, 2023))
    missing = stable_excess_gates(candidate, benchmark, spec=replace(StabilityGateSpec(), bootstrap_resamples=200))
    assert not missing["historical_qualification_pass"]
    assert not missing["gates"].seed_evidence_complete.any()


def test_same_two_seeds_must_pass_both_costs_and_none_can_be_dropped():
    candidate, benchmark = _annual_paths()
    seeds = _seed_table()
    seeds.loc[(seeds.seed.eq(71) & seeds.cost_bps.eq(10)) | (seeds.seed.eq(17) & seeds.cost_bps.eq(25)), "log_excess_sum"] = -.1
    result = stable_excess_gates(candidate, benchmark,
        spec=replace(StabilityGateSpec(), bootstrap_resamples=200), seed_table=seeds)
    assert result["gates"].positive_seeds_at_this_cost.eq(2).all()
    assert result["gates"].seeds_positive_at_both_costs.eq(1).all()
    assert not result["historical_qualification_pass"]
    with pytest.raises(ValueError, match="incomplete declared seed"):
        stable_excess_gates(candidate, benchmark, seed_table=seeds.iloc[:-1])


def test_cost_scenario_coverage_and_missing_year_block():
    candidate, benchmark = _annual_paths()
    with pytest.raises(ValueError, match="every preregistered cost"):
        stable_excess_gates({10.: candidate[10.]}, benchmark)
    altered = {cost: path.loc[path.signal_date.dt.year.ge(2017)] for cost, path in candidate.items()}
    other = {cost: path.loc[path.signal_date.dt.year.ge(2017)] for cost, path in benchmark.items()}
    with pytest.raises(ValueError, match="annual coverage"):
        stable_excess_gates(altered, other)


def test_seed_averages_are_not_reported_as_a_tradable_ensemble():
    candidate, benchmark = _annual_paths()
    table = seed_dispersion({"17": candidate[10.], "41": benchmark[10.]}, benchmark[10.])
    assert table["best_seed_selected"] is False
    assert table["ensemble_requires_separate_own_holdings_replay"] is True
    assert table["seeds_are_not_independent_market_observations"] is True
    assert not table["all_seeds_positive"]


def test_native_full_soxl_and_zero_cash_are_reconciled_without_old_caps():
    import qlib
    from qlib.data import D
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    provider = Path(__file__).resolve().parents[1] / "data/qlib/soxl_soxx_soxs"
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    prices = D.features(["SOXL", "SOXX"], ["$open"], start_time="2016-01-04", end_time="2016-01-08", freq="day")
    prices = prices["$open"].unstack("instrument").astype(float)
    dates = pd.DatetimeIndex(prices.index)
    holdings = np.array([0., 0., 1.])
    rows, replay = [], []
    for i, (desired, execute) in enumerate([(np.array([1., 0., 0.]), True), (None, False), (np.array([0., 1., 0.]), True)]):
        trade, exit_day = dates[i + 1], dates[i + 2]
        asset_returns = np.r_[(prices.loc[exit_day] / prices.loc[trade] - 1).to_numpy(), 0.]
        target = desired if execute else holdings.copy()
        executable = fee_affordable_target(target, pretrade_weights=holdings, cost_bps=25.) if execute else None
        settled = settle_target_session(holdings, open_target_weights=executable, asset_returns=asset_returns, cost_bps=25.)
        row = {"signal_date": dates[i], "trade_date": trade, "exit_date": exit_day,
               "soxl_return": asset_returns[0], "soxx_return": asset_returns[1]}
        rows.append(row)
        replay.append({**row, "strategy_return": settled["net_return"], "turnover": settled["turnover"],
                       "fees": settled["fees"], "cost_bps": 25., "execute_open_target": execute,
                       **{f"target_{asset}_weight": float(target[j]) for j, asset in enumerate(("soxl", "soxx", "cash"))}})
        holdings = settled["ending_weights"]
    result = native_allocation_reconciliation(pd.DataFrame(rows), pd.DataFrame(replay), provider_uri=provider, cost_bps=25.)
    assert result["audit"]["status"] == "passed"
    assert result["audit"]["maximum_session_return_error"] < 1e-12
    assert result["audit"]["legacy_controller_executed"] is False
    assert result["audit"]["risk_annualization_sessions"] == 252
    open_returns = result["reconciliation"].native_next_open_return.to_numpy()
    expected_annual = np.prod(1 + open_returns) ** (252 / len(open_returns)) - 1
    assert result["risk_analysis_open_to_open"].loc["annualized_return", "risk"] == pytest.approx(expected_annual)
    nav = np.r_[1., np.cumprod(1 + open_returns)]
    expected_dd = (nav / np.maximum.accumulate(nav) - 1).min()
    assert result["audit"]["native_open_to_open_anchored_max_drawdown_pct"] == pytest.approx(100 * expected_dd)
    assert len(result["report"]) == 3
