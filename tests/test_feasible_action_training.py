"""Contrast tests for the feasible-action training domain.

These test the domain filter mechanics without claiming the mechanism helps:
filter=all must exactly reproduce the sealed control, and the mask must depend
only on decision-time holdings, never on labels or future rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import coherent_price_replay_experiment as coherent
from src.workflow import feasible_action_training as F
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def _inputs():
    frozen = coherent.load_frozen(ROOT)
    prepared = frozen["inputs/prepared_inputs.pkl"]
    return frozen, prepared


@pytest.fixture(scope="module")
def _common(_inputs):
    _, prepared = _inputs
    return dict(recovery_models=prepared["recovery_models"],
                source_spec=IntradayRecoveryDirectAllocationSpec(root=ROOT),
                selected_model=prepared["selected_model"])


@pytest.fixture(scope="module")
def _book_all(_inputs, _common):
    _, prepared = _inputs
    return F.train_feasible_action_book(prepared["frame"].copy(), domain="all",
                                        update_through_year=2022, cost_bps=10.0, **_common)


def test_domain_all_exactly_reproduces_sealed_control(_book_all):
    assert np.allclose(_book_all["replay"]["strategy_return"], _book_all["base_replay"]["strategy_return"],
                       rtol=0, atol=1e-12)
    assert np.allclose(_book_all["replay"]["ending_soxl_weight"], _book_all["base_replay"]["ending_soxl_weight"],
                       rtol=0, atol=1e-12)


def test_unknown_domain_is_blocked(_inputs, _common):
    _, prepared = _inputs
    with pytest.raises(ValueError, match="unknown training domain"):
        F.train_feasible_action_book(prepared["frame"].copy(), domain="magic",
                                     update_through_year=2022, cost_bps=10.0, **_common)


def test_feasibility_mask_matches_replay_hard_risk_violation(_inputs):
    _, prepared = _inputs
    frozen = coherent.load_frozen(ROOT)
    training = frozen["analysis/training_results.pkl"][2022]["training_rows"]
    replay = frozen["analysis/training_results.pkl"][2022]["replay"]
    spec = IntradayRecoveryDirectAllocationSpec(root=ROOT)
    feasible, _ = F.feasibility_mask(training, maximum_soxl_weight=spec.maximum_soxl_weight,
                                     minimum_cash_weight=spec.minimum_cash_weight)
    joined = training.assign(feasible=feasible).merge(replay[["signal_date", "hard_risk_violation"]],
                                                      on="signal_date", how="left")
    assert joined["hard_risk_violation"].notna().all()
    assert (joined["feasible"] == ~joined["hard_risk_violation"].astype(bool)).all()


def test_feasibility_mask_ignores_labels_and_future_return(_inputs):
    _, prepared = _inputs
    frozen = coherent.load_frozen(ROOT)
    training = frozen["analysis/training_results.pkl"][2022]["training_rows"].copy()
    spec = IntradayRecoveryDirectAllocationSpec(root=ROOT)
    base, _ = F.feasibility_mask(training, maximum_soxl_weight=spec.maximum_soxl_weight,
                                 minimum_cash_weight=spec.minimum_cash_weight)
    scrambled = training.copy()
    scrambled["execute_advantage_bps"] = np.random.default_rng(0).normal(size=len(scrambled))
    scrambled["execute_net_return"] = np.random.default_rng(1).normal(size=len(scrambled))
    after, _ = F.feasibility_mask(scrambled, maximum_soxl_weight=spec.maximum_soxl_weight,
                                  minimum_cash_weight=spec.minimum_cash_weight)
    assert (base == after).all()


def test_future_year_labels_do_not_change_past_fit(_inputs):
    _, prepared = _inputs
    frozen = coherent.load_frozen(ROOT)
    rows = frozen["analysis/training_results.pkl"][2022]["training_rows"].copy()
    spec = IntradayRecoveryDirectAllocationSpec(root=ROOT)
    feasible, _ = F.feasibility_mask(rows, maximum_soxl_weight=spec.maximum_soxl_weight,
                                     minimum_cash_weight=spec.minimum_cash_weight)
    rows["routing_eligible"] = feasible
    first = F._first_decision_by_year(prepared["frame"])
    base_models, _ = F._annual_models(rows, domain="feasible", update_through_year=2022,
                                      first_decision=first)
    perturbed = rows.copy()
    future = pd.to_datetime(perturbed["signal_date"]).dt.year >= 2020
    perturbed.loc[future, "execute_advantage_bps"] = (
        perturbed.loc[future, "execute_advantage_bps"] + 10_000.0)
    new_models, _ = F._annual_models(perturbed, domain="feasible", update_through_year=2022,
                                     first_decision=first)
    features = rows.loc[rows["routing_eligible"], list(F.ECONOMIC_FEATURES)].head(50)
    for year in (2017, 2018, 2019):
        assert base_models[year] is not None and new_models[year] is not None
        assert np.allclose(base_models[year].predict(features), new_models[year].predict(features), rtol=0, atol=1e-12)


def test_insufficient_feasible_rows_reports_fallback(_inputs):
    _, prepared = _inputs
    frozen = coherent.load_frozen(ROOT)
    rows = frozen["analysis/training_results.pkl"][2022]["training_rows"].copy()
    rows["routing_eligible"] = False  # nothing feasible -> fallback everywhere
    first = F._first_decision_by_year(prepared["frame"])
    _, diagnostics = F._annual_models(rows, domain="feasible", update_through_year=2022, first_decision=first)
    assert diagnostics
    assert all(d["fallback"] and not d["model_available"] for d in diagnostics)
