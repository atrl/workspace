"""Small independent accounting identities; no market replay or model search."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis.expert_information_attribution import attribute_expert_information


@pytest.fixture
def evidence():
    dates = pd.bdate_range("2024-01-02", periods=8)
    soxx = np.array([-.03, -.02, -.01, .01, .02, .03])
    residual = np.array([.001, -.002, .001, .001, -.002, .001])
    market = pd.DataFrame({"signal_date": dates[:-2], "trade_date": dates[1:-1],
        "exit_date": dates[2:], "soxx_return": soxx, "soxl_return": 2.8 * soxx + residual})
    weights = {
        "benchmark": np.tile([0., 1., 0.], (6, 1)),
        "lever": np.tile([.2, .8, 0.], (6, 1)),
        "defense": np.tile([0., .8, .2], (6, 1)),
        "fixed": np.tile([.03, .94, .03], (6, 1)),
        "online": np.array([[.01, .95, .04], [.02, .94, .04], [.03, .94, .03],
                            [.04, .93, .03], [.02, .95, .03], [.05, .93, .02]]),
    }
    paths = {}
    for cost in (10., 25.):
        for j, (name, target) in enumerate(weights.items()):
            path = market[["signal_date", "trade_date", "exit_date"]].copy()
            fee = (np.arange(6) + 1 + j) * cost / 10000 / 10
            # Independently value fractional shares plus cash after an explicit
            # fee payment, rather than using the implementation's return formula.
            starting_nav = 100_000.
            dollars = (starting_nav - starting_nav * fee[:, None]) * target
            shares = dollars[:, :2] / np.array([20., 100.])
            next_prices = np.column_stack([20 * (1 + market.soxl_return), 100 * (1 + soxx)])
            path["strategy_return"] = ((shares * next_prices).sum(axis=1) + dollars[:, 2]) / starting_nav - 1
            path["fees"], path["turnover"], path["cost_bps"] = fee, fee / (cost / 10000), cost
            path[["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]] = target
            paths[(name, cost)] = path
    return paths, market


def analyze(paths, market, **kwargs):
    return attribute_expert_information(paths, market,
        windows={"all": ("2024-01-01", "2024-01-31")}, candidate_name="online",
        reference_name="fixed", benchmark_name="benchmark", leveraged_name="lever",
        defensive_name="defense", **kwargs)


def test_exact_components_match_independent_share_accounting_and_known_beta(evidence):
    paths, market = evidence
    original = {key: frame.copy(deep=True) for key, frame in paths.items()}
    result = analyze(paths, market)
    np.testing.assert_allclose(result["structure"].realized_soxl_beta_to_soxx, 2.8, atol=1e-14)
    for row in result["attribution"].itertuples():
        candidate, reference = paths[("online", row.cost_bps)], paths[("fixed", row.cost_bps)]
        delta_l = candidate.open_soxl_weight - reference.open_soxl_weight
        delta_beta = 2.8 * delta_l + candidate.open_soxx_weight - reference.open_soxx_weight
        expected_level = delta_beta.mean() * market.soxx_return.sum() * 10000
        expected_timing = ((delta_beta - delta_beta.mean()) * market.soxx_return).sum() * 10000
        expected_residual = (delta_l * (market.soxl_return - 2.8 * market.soxx_return)).sum() * 10000
        np.testing.assert_allclose([row.mean_beta_exposure_component_bp, row.beta_timing_component_bp,
            row.leveraged_residual_component_bp], [expected_level, expected_timing, expected_residual], atol=1e-12)
        components = (row.mean_beta_exposure_component_bp + row.beta_timing_component_bp
                      + row.leveraged_residual_component_bp + row.fee_difference_component_bp)
        expected_difference = (candidate.strategy_return - reference.strategy_return).sum() * 10000
        assert components == pytest.approx(expected_difference, abs=1e-10)
        assert row.maximum_daily_identity_error < 1e-14
        relative = np.prod(1 + candidate.strategy_return) / np.prod(1 + reference.strategy_return) - 1
        assert row.compounded_relative_return_bp == pytest.approx(relative * 10000, abs=1e-10)
        assert "not_compounded" in row.component_unit
        assert "not_prediction_or_promotion" in row.evidence_scope
    assert len(result["accounting"]) == 10
    assert result["accounting"].maximum_net_reconstruction_error.max() < 1e-14
    for key in paths:
        pd.testing.assert_frame_equal(paths[key], original[key], check_exact=True)


@pytest.mark.parametrize("change", ["truncate_path", "shift_path_dates", "truncate_market", "extra_path_wrong_dates"])
def test_rejects_any_same_window_mismatch_without_inner_join(evidence, change):
    paths, market = evidence
    if change == "truncate_path":
        paths[("online", 10.)] = paths[("online", 10.)].iloc[1:].copy()
    elif change == "shift_path_dates":
        for name in ("signal_date", "trade_date", "exit_date"):
            paths[("online", 10.)][name] = [day.to_pydatetime() + timedelta(days=1)
                                           for day in paths[("online", 10.)][name]]
    elif change == "truncate_market":
        market = market.iloc[1:].copy()
    else:
        paths[("extra", 10.)] = paths[("fixed", 10.)].iloc[1:].copy()
    with pytest.raises(ValueError, match="identical signal/trade/exit dates"):
        analyze(paths, market)


@pytest.mark.parametrize("change", ["fee_units", "return", "negative_weight", "unfunded", "wrong_cost", "missing_cost", "nonfinite"])
def test_rejects_invalid_funding_and_cost_contract(evidence, change):
    paths, market = evidence
    path = paths[("online", 10.)]
    if change == "fee_units":
        path["fees"] *= 2; path["turnover"] *= 2
    elif change == "return":
        path.loc[0, "strategy_return"] += .001
    elif change == "negative_weight":
        path.loc[0, ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]] = [-.01, .95, .06]
    elif change == "unfunded":
        path.loc[0, "open_cash_weight"] += .1
    elif change == "wrong_cost":
        path["cost_bps"] = 25.; path["turnover"] = path.fees / .0025
    elif change == "missing_cost":
        paths[("online", 10.)] = path.drop(columns="cost_bps")
    else:
        market.loc[0, "soxl_return"] = np.nan
    with pytest.raises(ValueError, match="blocked:"):
        analyze(paths, market)


def test_identical_candidate_has_zero_attribution_and_undefined_timing_correlation(evidence):
    paths, market = evidence
    for cost in (10., 25.):
        paths[("online", cost)] = paths[("fixed", cost)].copy()
    result = analyze(paths, market)["attribution"]
    assert (result.net_difference_sum_bp == 0).all()
    assert (result.compounded_relative_return_bp == 0).all()
    assert result.delta_beta_future_soxx_return_correlation.isna().all()
    assert result.delta_soxl_weight_future_residual_correlation.isna().all()


def test_all_named_strategies_required_at_every_cost(evidence):
    paths, market = evidence
    del paths[("defense", 25.)]
    with pytest.raises(ValueError, match="all named strategies"):
        analyze(paths, market)
