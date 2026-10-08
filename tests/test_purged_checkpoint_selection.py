"""Synthetic chronology, checkpoint identity and model-selection contracts."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

import src.model.purged_checkpoint_selection as m
from src.data.benchmark_anchor_utility import (
    PRICE_FEATURES, build_anchor_dataset, prepare_training_rows, fit_feature_scaler, transform_features,
)
from src.model.constrained_net_excess import ConstrainedNetExcessSpec
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_to_dict


@pytest.fixture(scope="module")
def market():
    dates = pd.bdate_range("2013-01-02", "2023-01-10", name="datetime")
    t = np.arange(len(dates))
    frame = pd.DataFrame({name: np.sin(t * (.007 + j * .001)) for j, name in enumerate(PRICE_FEATURES)}, index=dates)
    days = pd.Series(dates, index=dates)
    frame["signal_date"] = days
    frame["trade_date"], frame["exit_date"] = days.shift(-1), days.shift(-2)
    def at(values, hours, minutes=0):
        return values.dt.tz_localize("America/New_York") + pd.Timedelta(hours=hours, minutes=minutes)
    frame["feature_available_at"] = at(days, 16)
    frame["decision_at"] = at(frame.trade_date, 9, 29)
    frame["return_available_at"] = at(frame.exit_date, 9, 30)
    frame["forecast_available_at"] = at(days.shift(-6), 9, 30)
    frame["auxiliary_available_at"] = at(days.shift(-5), 16)
    frame["soxl_return"] = .025 * np.sin(t * .19)
    frame["soxx_return"] = .008 * np.sin(t * .19)
    frame.loc[frame.exit_date.isna(), ["soxl_return", "soxx_return"]] = np.nan
    frame["forecast_available"] = 1.0
    frame["future_vol_prediction"] = .02 + .003 * np.sin(t * .07)
    frame["forecast_model_available_at"] = pd.Series([
        pd.Timestamp(f"{date.year}-01-01", tz="America/New_York") for date in dates], index=dates)
    return frame


@pytest.fixture
def training(market):
    dataset = build_anchor_dataset(market, as_of="2023-01-10")
    rows, audit = prepare_training_rows(dataset, 2016)
    scaler = fit_feature_scaler(rows, audit, "price")
    transformed = transform_features(rows, scaler, "price")
    fold = {"rows": rows, "audit": audit, "scaler": scaler, "features": transformed["features"],
        "returns": rows[["soxl_return", "soxx_return"]].to_numpy(float), "invalid_training": False}
    spec = ConstrainedNetExcessSpec()
    history = []
    # Two equal early improvements, a higher infeasible state, and a later best.
    for epoch in range(121):
        value = 0.0 if epoch < 3 else .1 if epoch < 30 else .2
        feasible = True
        if epoch == 10:
            value, feasible = .3, False
        model = BenchmarkAnchorPolicy(seed=17, anchor=True)
        with torch.no_grad():
            model.theta[0, 0] = value
        history.append({"epoch": epoch, "F": value, "feasible": feasible, "checkpoint": policy_to_dict(model)})
    fit = {"variant": "price", "year": 2016, "seed": 17, "failure": None, "qualification_eligible": True,
        "spec": asdict(spec), "training_rows": len(rows), "epochs": history,
        "training_feature_sha256": m._array_hash(fold["features"]),
        "training_return_sha256": m._array_hash(fold["returns"]), "selected_epoch": 30, "F": .2}
    return fold, fit, spec


def test_reconstructs_first_strict_feasible_incumbent_and_explicit_null(training, monkeypatch):
    fold, fit, spec = training
    def independent(model, features, returns, *, spec):
        return {"F": float(model.theta[0, 0].detach()), "feasible": True}
    monkeypatch.setattr(m, "evaluate_policy", independent)
    for cap, epoch, score in [(0, "anchor", 0.), (15, 3, .1), (30, 30, .2), (120, 30, .2)]:
        payload, audit = m.reconstruct_incumbent(fit, fold, budget=cap, spec=spec, variant="price", year=2016, seed=17)
        assert audit["selected_epoch"] == epoch
        assert audit["training_F"] == score and not audit["new_fit"]
        assert audit["checkpoint_sha256"] == payload["tensor_sha256"]
        if cap == 0:
            assert not payload["state_dict"]["theta"].any()


@pytest.mark.parametrize("corruption", ["feature", "checkpoint", "score", "row_dates", "missing_epoch"])
def test_reconstruction_fails_closed_on_changed_evidence(training, monkeypatch, corruption):
    fold, fit, spec = deepcopy(training)
    monkeypatch.setattr(m, "evaluate_policy", lambda model, *args, **kwargs:
        {"F": float(model.theta[0, 0].detach()), "feasible": True})
    if corruption == "feature":
        fold["features"][0, 0] += 1
    elif corruption == "checkpoint":
        fit["epochs"][4]["checkpoint"]["state_dict"]["theta"][0, 0] += .01
    elif corruption == "score":
        fit["epochs"][4]["F"] += .01
    elif corruption == "row_dates":
        fold["audit"]["eligible_dates"][-1] = "2099-01-01"
    else:
        fit["epochs"].pop()
    with pytest.raises(ValueError):
        m.reconstruct_incumbent(fit, fold, budget=15, spec=spec, variant="price", year=2016, seed=17)


def test_purges_actual_outcome_availability_across_year_boundary(market):
    rows, audit = m.purged_validation_frame(market, outer_year=2018)
    cutoff = pd.Timestamp("2018-01-01", tz="America/New_York")
    expected = market.loc[market.index.year.isin([2016, 2017]) & market.return_available_at.lt(cutoff)]
    pd.testing.assert_frame_equal(rows, expected)
    assert audit["purged_tail_rows"] == 2
    assert pd.Timestamp(audit["label_available_at_max"]) < cutoff
    changed = market.copy()
    changed.loc[rows.index[-1], "return_available_at"] = cutoff
    shorter, _ = m.purged_validation_frame(changed, outer_year=2018)
    assert rows.index[-1] not in shorter.index


@pytest.mark.parametrize("corruption", ["interior_gap", "missing_session", "interior_nonfinite", "missing_time", "naive_time"])
def test_does_not_silently_compress_validation_holdings_path(market, corruption):
    changed = market.copy()
    stamp = pd.Timestamp("2017-02-01")
    if corruption == "interior_gap":
        changed.loc[stamp, "return_available_at"] = pd.Timestamp("2018-01-03", tz="America/New_York")
    elif corruption == "missing_session":
        changed = changed.drop(index=stamp)
    elif corruption == "interior_nonfinite":
        changed.loc[stamp, "soxl_return"] = np.nan
    elif corruption == "missing_time":
        changed.loc[stamp, "return_available_at"] = pd.NaT
    else:
        changed["return_available_at"] = changed.return_available_at.dt.tz_localize(None)
    with pytest.raises(ValueError):
        m.purged_validation_frame(changed, outer_year=2018)


def test_future_values_cannot_change_validation_rows(market):
    before, audit = m.purged_validation_frame(market, outer_year=2018)
    changed = market.copy()
    future = changed.index.year >= 2018
    changed.loc[future, ["soxl_return", "soxx_return"]] = np.nan
    changed.loc[future, "future_vol_prediction"] = -999
    after, later_audit = m.purged_validation_frame(changed, outer_year=2018)
    pd.testing.assert_frame_equal(before, after)
    assert audit == later_audit


def _fake_selection_paths(monkeypatch, scores):
    seen = []
    def reference(rows, *, cost_bps):
        out = rows[["signal_date", "trade_date", "exit_date"]].reset_index(drop=True).copy()
        out["strategy_return"] = 0.0
        return out
    def replay(rows, annual, *, variant, cost_bps):
        seen.append((rows.copy(), annual["budget"], cost_bps))
        out = reference(rows, cost_bps=cost_bps)
        out["strategy_return"] = np.expm1(scores[annual["budget"]][cost_bps])
        for flag in ("observed_forecast_fallback", "invalid_training_fallback", "numerical_failure"):
            out[flag] = False
        return out
    monkeypatch.setattr(m, "benchmark_path", reference)
    monkeypatch.setattr(m, "replay_models", replay)
    bundles = {("price", budget): {"budget": budget, **{year: {"ensemble_models": [
        {"state_dict": {"theta": np.array([0. if budget == 0 else 1.])}}]}
        for year in (2016, 2017)}} for budget in m.BUDGETS}
    return bundles, seen


def test_worst_cost_selects_shared_cap_without_looking_at_outer_year(market, monkeypatch):
    scores = {0: {10.: 0, 25.: 0}, 15: {10.: .02, 25.: -.01},
              30: {10.: .001, 25.: .001}, 60: {10.: .001, 25.: .001}, 120: {10.: .003, 25.: -.001}}
    bundles, seen = _fake_selection_paths(monkeypatch, scores)
    chosen, table, audit = m.select_budget(market, bundles, variant="price", outer_year=2018)
    assert chosen == 30 and table.selected.sum() == 1  # identical 60 loses complexity tie
    assert audit["all_three_seeds_used"] and not audit["selection_uses_outer"]
    assert len(seen) == 10
    for rows, _, _ in seen:
        assert set(rows.index.year) == {2016, 2017}
        assert rows.return_available_at.lt(pd.Timestamp("2018-01-01", tz="America/New_York")).all()


@pytest.mark.parametrize("other_score", [-.001, 0.0])
def test_explicit_null_wins_all_negative_or_zero_validation(market, monkeypatch, other_score):
    scores = {b: {10.: other_score, 25.: other_score} for b in m.BUDGETS}
    scores[0] = {10.: 0.0, 25.: 0.0}
    bundles, _ = _fake_selection_paths(monkeypatch, scores)
    chosen, table, _ = m.select_budget(market, bundles, variant="price", outer_year=2018)
    assert chosen == 0
    assert table.loc[table.budget.eq(0), "selection_score"].iloc[0] == 0.0


def test_learned_cap_null_is_not_a_positive_candidate_due_to_roundoff(market, monkeypatch):
    scores = {budget: {10.: -0.001, 25.: -0.001} for budget in m.BUDGETS}
    scores[0] = {10.: 0.0, 25.: 0.0}
    scores[15] = {10.: 1e-16, 25.: 1e-16}
    bundles, _ = _fake_selection_paths(monkeypatch, scores)
    for year in (2016, 2017):
        bundles[("price", 15)][year]["ensemble_models"][0]["state_dict"]["theta"][:] = 0.0
    chosen, table, _ = m.select_budget(market, bundles, variant="price", outer_year=2018)
    assert chosen == 0
    assert table.loc[table.budget.eq(15), "selection_score"].iloc[0] == 0.0


def test_adaptive_bundles_keep_fixed_warmup_and_do_not_mutate_source():
    bundles = {("price", b): {year: {"budget": b, "year": year} for year in m.YEARS} for b in m.BUDGETS}
    selected = {year: 0 if year == 2018 else 30 for year in m.OUTER_YEARS}
    result = m.annual_bundles(bundles, variant="price", selected_by_year=selected)
    assert result[2016]["budget"] == result[2017]["budget"] == 120
    assert result[2018]["budget"] == 0
    result[2018]["budget"] = 999
    assert bundles[("price", 0)][2018]["budget"] == 0


def test_actual_null_replay_retains_two_year_own_book_and_matches_benchmark(market):
    dataset = build_anchor_dataset(market, as_of="2023-01-10")
    annual = {}
    for year in (2016, 2017):
        rows, audit = prepare_training_rows(dataset, year)
        scaler = fit_feature_scaler(rows, audit, "price")
        annual[year] = {"variant": "price", "model_year": year, "scaler": scaler,
            "scaler_sha256": scaler["scaler_sha256"], "invalid_training": False, "numerical_failure": False,
            "ensemble_models": [policy_to_dict(BenchmarkAnchorPolicy(seed=s, anchor=True)) for s in m.SEEDS],
            "ensemble_audit": {"anchor_selected": True}}
    valid, _ = m.purged_validation_frame(market, outer_year=2018)
    for cost in m.COSTS:
        path = m.replay_models(valid, annual, variant="price", cost_bps=cost)
        reference = m.benchmark_path(valid, cost_bps=cost)
        np.testing.assert_allclose(path.strategy_return, reference.strategy_return, atol=1e-14, rtol=0)
        boundary = np.flatnonzero(path.signal_date.dt.year.to_numpy() == 2017)[0]
        assert path.pretrade_soxx_weight.iloc[boundary] == 1
        assert abs(path.fees.iloc[boundary]) < 1e-15


def test_actual_selection_unchanged_by_outer_and_unmatured_outcomes_or_auxiliary_labels(market):
    dataset = build_anchor_dataset(market, as_of="2023-01-10")
    bundles = {}
    for budget in m.BUDGETS:
        annual = {}
        for year in (2016, 2017):
            training, audit = prepare_training_rows(dataset, year)
            scaler = fit_feature_scaler(training, audit, "price")
            payloads = []
            for seed in m.SEEDS:
                policy = BenchmarkAnchorPolicy(seed=seed, anchor=True)
                with torch.no_grad():
                    policy.theta[0, 0] = budget / 1200.
                payloads.append(policy_to_dict(policy))
            annual[year] = {"variant": "price", "model_year": year, "scaler": scaler,
                "scaler_sha256": scaler["scaler_sha256"], "invalid_training": False,
                "numerical_failure": False, "ensemble_models": payloads,
                "ensemble_audit": {"anchor_selected": budget == 0}}
        bundles[("price", budget)] = annual
    selected, scores, _ = m.select_budget(market, bundles, variant="price", outer_year=2018)
    changed = market.copy()
    outer = changed.index.year >= 2018
    cutoff = pd.Timestamp("2018-01-01", tz="America/New_York")
    immature = changed.index.year.isin([2016, 2017]) & changed.return_available_at.ge(cutoff)
    changed.loc[outer | immature, ["soxl_return", "soxx_return"]] = 1000.
    changed.loc[outer, list(PRICE_FEATURES)] = 9999.
    changed["forecast_available_at"] = pd.Timestamp("2099-01-01", tz="America/New_York")
    changed["auxiliary_available_at"] = pd.Timestamp("2099-01-01", tz="America/New_York")
    changed["tail_event_5d"] = 999.0
    later_selected, later_scores, _ = m.select_budget(changed, bundles, variant="price", outer_year=2018)
    assert later_selected == selected
    pd.testing.assert_frame_equal(later_scores, scores)
