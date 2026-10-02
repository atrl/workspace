import numpy as np
import pandas as pd
import pytest
import torch

from src.data.integrated_allocation import PRIMARY_FEATURES, PRIMARY_LABELS, FORECAST_FEATURES
from src.workflow import stable_excess_integration as workflow


def _forecast_frame():
    dates = pd.bdate_range("2011-01-03", "2017-12-29")
    random = np.random.default_rng(17)
    frame = pd.DataFrame(random.normal(size=(len(dates), len(PRIMARY_FEATURES))), index=dates, columns=PRIMARY_FEATURES)
    frame.index.name = "datetime"
    frame.loc[:, list(FORECAST_FEATURES)] = 0.
    for name in PRIMARY_LABELS:
        frame[name] = random.normal(0, .01, len(dates))
    frame["signal_date"] = dates
    frame["trade_date"] = dates + pd.offsets.BDay(1)
    frame["exit_date"] = dates + pd.offsets.BDay(2)
    for name, offset, hour in [("feature_available_at", 0, 16), ("decision_at", 1, 10),
                               ("return_available_at", 2, 10), ("forecast_available_at", 6, 10),
                               ("auxiliary_available_at", 5, 16)]:
        frame[name] = (dates + pd.offsets.BDay(offset)).tz_localize("America/New_York") + pd.Timedelta(hours=hour)
    return frame


def test_annual_forecasts_exclude_unmatured_labels_and_future_training_years():
    frame = _forecast_frame()
    first, audits, models = workflow.annual_forecasts(frame, as_of="2017-12-29")
    altered = frame.copy()
    # These 2016 signals have 5-day risk labels unavailable before the 2017
    # model cutoff. Changing them must not change any 2017 prediction.
    cutoff = pd.Timestamp("2017-01-01", tz="America/New_York")
    unmatured = altered.forecast_available_at.ge(cutoff)
    altered.loc[unmatured, ["cash_advantage_1d", "future_vol_5d"]] += 500.
    second, _, models_second = workflow.annual_forecasts(altered, as_of="2017-12-29")
    pd.testing.assert_frame_equal(first.loc[:, list(FORECAST_FEATURES)], second.loc[:, list(FORECAST_FEATURES)])
    np.testing.assert_array_equal(models[2017].named_steps["ridge"].coef_, models_second[2017].named_steps["ridge"].coef_)
    available = audits.loc[audits.available]
    assert all(pd.Timestamp(row.label_available_at_max) < pd.Timestamp(row.decision_at) for row in available.itertuples())
    assert not first.loc[first.index.year == 2011, "forecast_available"].any()


class HoldingsModel(torch.nn.Module):
    def __init__(self, intercept, sensitivity, soxx):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.intercept, self.sensitivity, self.soxx = intercept, sensitivity, soxx
        self.seen = []

    def forward(self, features, holdings):
        self.seen.append(holdings.detach().cpu().numpy().copy())
        soxl = self.intercept + self.sensitivity * holdings[:, 0]
        soxx = torch.full_like(soxl, self.soxx)
        return torch.stack([soxl, soxx, 1 - soxl - soxx], dim=-1)


def _market():
    dates = pd.bdate_range("2016-01-04", periods=3)
    return pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
        "exit_date": dates + pd.offsets.BDay(2), "soxl_return": [.10, -.05, .02],
        "soxx_return": [.03, -.01, .01], "feature": 0.}, index=dates)


def test_ensemble_members_observe_same_owned_holdings_and_do_not_average_returns():
    a, b = HoldingsModel(.1, .7, .3), HoldingsModel(.6, -.2, .1)
    market = _market()
    replay = workflow.replay_models(market, {17: {2016: a}, 41: {2016: b}}, cost_bps=10., feature_columns=["feature"], seeds=(17, 41))
    for number in range(len(market)):
        np.testing.assert_array_equal(a.seen[number], b.seen[number])
        np.testing.assert_allclose(a.seen[number][0], replay.iloc[number][[f"pretrade_{asset}_weight" for asset in ("soxl", "soxx", "cash")]].to_numpy(dtype=float))
    assert replay.iloc[0].soxl_weight == pytest.approx(.35)
    aa, bb = HoldingsModel(.1, .7, .3), HoldingsModel(.6, -.2, .1)
    ra = workflow.replay_models(market, {17: {2016: aa}}, cost_bps=10., feature_columns=["feature"], seeds=(17,))
    rb = workflow.replay_models(market, {41: {2016: bb}}, cost_bps=10., feature_columns=["feature"], seeds=(41,))
    assert not np.allclose(replay.soxl_weight, (ra.soxl_weight + rb.soxl_weight) / 2)
    assert not np.allclose(replay.strategy_return, (ra.strategy_return + rb.strategy_return) / 2)


def test_benchmark_only_pays_one_entry_fee_and_holds_quantities():
    market = _market()
    path = workflow.benchmark_path(market, cost_bps=25., stock_weight=1.)
    assert path.execute_open_target.tolist() == [True, False, False]
    assert path.turnover.iloc[1:].eq(0).all() and path.fees.iloc[1:].eq(0).all()
    expected = np.prod(1 + market.soxx_return) / 1.0025
    assert path.equity.iloc[-1] == pytest.approx(expected)


def _evidence(passed, lower, boolean_gates=(True, True)):
    return {"gates": pd.DataFrame({"cost_bps": [10., 25.], "criterion": boolean_gates,
                                   "all_pass": passed, "seed_evidence_complete": True, "positive_years": 5}),
            "historical_qualification_pass": passed,
            "bootstrap": pd.DataFrame({"annualized_lower_bound": [lower, lower]}),
            "metrics": pd.DataFrame({"relative_wealth": [1.1, 1.1]})}


def test_family_selection_cannot_upgrade_a_failed_candidate_or_chase_best_return():
    selected, _ = workflow.choose_family({"passed": _evidence(True, .01), "failed": _evidence(False, .5)})
    assert selected["variant"] == "passed"
    assert selected["historical_qualification_pass"]
    selected, _ = workflow.choose_family({"a": _evidence(False, .01), "b": _evidence(False, .02)})
    assert selected["variant"] == "b"
    assert not selected["historical_qualification_pass"]


def test_replay_cannot_use_a_model_fitted_for_a_future_year():
    with pytest.raises(ValueError, match="missing causal policy"):
        workflow.replay_models(_market(), {17: {2017: HoldingsModel(.1, .7, .3)}},
                               cost_bps=10., feature_columns=["feature"], seeds=(17,))
