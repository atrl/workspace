from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.position_accounting import settle_target_session
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec, mark_to_market_weights,
)
from src.workflow.stage1_open_action_value import ACTION_FEATURES, prepare_on_policy_open, Stage1OpenActionValueSpec
from src.workflow.stage1_economic_action import (
    ECONOMIC_FEATURES, fee_affordable_target, prepare_economic_open, prepare_economic_recovery,
)


SPEC = IntradayRecoveryDirectAllocationSpec(root=Path("."))
SELECTED = {"action_beta": .6, "probability_threshold": .825, "minimum_source_alpha_probability": .25}


class Classifier:
    def __init__(self, probability):
        self.probability = probability
        self.features = None

    def predict_proba(self, features):
        self.features = features.copy()
        return np.array([[1 - self.probability, self.probability]])


class NetValue:
    def __init__(self, bps):
        self.bps = bps
        self.features = None

    def predict(self, features):
        self.features = features.copy()
        return np.array([self.bps])


def _row(soxl=.1, soxx=.5):
    return pd.Series({
        "soxl_weight": soxl, "soxx_weight": soxx, "cash_weight": 1 - soxl - soxx,
        "tail_probability": .1, "alpha_probability": .8,
        "early_ret_soxl": .02, "early_ret_soxx": .01,
        "early_range_soxl": .03, "early_range_soxx": .02,
        "early_from_low_soxl": .02, "early_from_low_soxx": .01,
        "early_volume_soxl": 1_000_000., "early_volume_soxx": 500_000.,
    })


def _open(row=None, **overrides):
    kwargs = dict(pretrade_weights=np.array([.11, .5, .39]), equity=1., peak=1.,
                  risk_state=0, risk_clock=0, history=[], recovery_spec=SPEC,
                  model=Classifier(.1), policy="rules_only", days_since_trade=0, cost_bps=10.)
    kwargs.update(overrides)
    return prepare_economic_open(_row() if row is None else row, **kwargs)


def test_soft_target_beta_is_not_the_independent_hard_limit():
    current = np.array([.11, .5, .39])
    opened = _open(pretrade_weights=current)
    assert opened["soft_beta_cap"] == pytest.approx(.8)
    assert opened["hard_beta_cap"] == pytest.approx(1.45)
    assert opened["holding_beta"] > opened["soft_beta_cap"]
    assert opened["hard_risk_violation"] is False
    assert opened["mandatory_derisk"] is False
    assert opened["execute_open_target"] is False
    np.testing.assert_array_equal(opened["open_weights"], current)
    np.testing.assert_array_equal(opened["execution_target_weights"], current)
    old = prepare_on_policy_open(
        _row(), pretrade_weights=current, equity=1., peak=1., risk_state=0, risk_clock=0,
        history=[], recovery_spec=SPEC, model=Classifier(.1), action_spec=Stage1OpenActionValueSpec(root=Path(".")),
    )
    assert old["execute_open_target"] is True


@pytest.mark.parametrize("policy,model", [("rules_only", Classifier(.0)), ("net_value", NetValue(-100))])
def test_real_soxl_and_cash_violations_override_models(policy, model):
    opened = _open(pretrade_weights=np.array([.31, .55, .14]), policy=policy, model=model)
    assert opened["execute_open_target"] is True
    assert opened["decision_reason"] == "hard_risk_repair"
    assert set(opened["hard_risk_violation_reasons"]) == {"maximum_soxl_weight", "minimum_cash_weight", "hard_beta_cap"}
    assert 3 * opened["desired_open_weights"][0] + opened["desired_open_weights"][1] <= opened["hard_beta_cap"] + 1e-12


def test_net_value_does_not_inherit_the_large_transition_override():
    current = np.array([.29, .54, .17])
    rules = _open(_row(.01, .2), pretrade_weights=current, model=Classifier(.0))
    net = _open(_row(.01, .2), pretrade_weights=current, model=NetValue(-1), policy="net_value")
    assert rules["large_transition"] and net["large_transition"]
    assert rules["decision_reason"] == "large_transition_execute_target"
    assert net["decision_reason"] == "nonpositive_net_advantage"
    assert net["execute_open_target"] is False
    assert net["execution_turnover"] == 0


@pytest.mark.parametrize("bps,execute", [(-1., False), (0., False), (.001, True)])
def test_net_value_uses_signed_bps_zero_threshold(bps, execute):
    result = _open(model=NetValue(bps), policy="net_value")
    assert result["execute_open_target"] is execute
    assert result["predicted_net_advantage_bps"] == bps


@pytest.mark.parametrize("model", [None, NetValue(np.nan), NetValue(np.inf)])
def test_unavailable_net_model_executes_the_existing_desired_target(model):
    result = _open(model=model, policy="net_value")
    assert result["execute_open_target"] is True
    np.testing.assert_allclose(result["open_weights"], result["desired_open_weights"])


def test_feature_contract_uses_advanced_clock_and_no_future_intraday_input():
    model = NetValue(-1)
    arguments = dict(pretrade_weights=np.array([0., .2, .8]), model=model, policy="net_value",
                     risk_state=2, risk_clock=4, equity=.9, peak=1., days_since_trade=7)
    row = _row()
    first = _open(row, **arguments)
    row["early_ret_soxl"], row["early_ret_soxx"] = -.9, .9
    second = _open(row, **arguments)
    assert first["features"] == second["features"]
    assert tuple(model.features.columns) == ECONOMIC_FEATURES
    assert len(ECONOMIC_FEATURES) == 25
    assert first["features"]["risk_clock"] == 5
    assert first["features"]["days_since_trade"] == 7
    assert first["hard_beta_cap"] == pytest.approx(.5 * 1.45)
    classifier = Classifier(.1)
    _open(model=classifier)
    assert tuple(classifier.features.columns) == ACTION_FEATURES


@pytest.mark.parametrize("cost", [10., 25.])
def test_fee_projection_fixes_boundary_violation_without_loosening_caps(cost):
    row = _row(.3, .55)
    first = _open(row, pretrade_weights=np.array([0., 0., 1.]), model=None,
                  policy="net_value", cost_bps=cost)
    legacy = settle_target_session([0, 0, 1], open_target_weights=first["desired_open_weights"], asset_returns=[0, 0, 0], cost_bps=cost)
    assert legacy["ending_weights"][0] > .3 + 1e-12
    assert legacy["ending_weights"][2] < .15 - 1e-12
    fixed = settle_target_session([0, 0, 1], open_target_weights=first["execution_target_weights"], asset_returns=[0, 0, 0], cost_bps=cost)
    assert first["execution_target_weights"].sum() == pytest.approx(1.)
    assert fixed["ending_weights"] == pytest.approx([.3, .55, .15], abs=1e-13)
    assert 3 * fixed["ending_weights"][0] + fixed["ending_weights"][1] <= 1.45 + 1e-12
    next_day = _open(row, pretrade_weights=fixed["ending_weights"], model=NetValue(-1),
                     policy="net_value", cost_bps=cost, equity=fixed["net_equity"], history=[fixed["net_return"]])
    assert next_day["hard_risk_violation"] is False
    assert next_day["execute_open_target"] is False


@pytest.mark.parametrize("current,desired", [([0, 0, 1], [.3, .55, .15]), ([.3, .55, .15], [0, .2, .8]), ([.1, .6, .3], [.2, .4, .4])])
def test_fee_projection_handles_buys_sells_and_rotations(current, desired):
    execution = fee_affordable_target(np.array(desired), pretrade_weights=np.array(current), cost_bps=25)
    settled = settle_target_session(current, open_target_weights=execution, asset_returns=[0, 0, 0], cost_bps=25)
    np.testing.assert_allclose(settled["ending_weights"], desired, rtol=0, atol=5e-13)


def test_wait_clock_recovery_exception_is_preserved_without_double_advance():
    opened = _open(equity=.85, model=NetValue(-100), policy="net_value")
    assert opened["hard_beta_cap"] == pytest.approx(.2)
    clock = opened["risk_clock"]
    recovery_model = Classifier(.99)
    at_cut = mark_to_market_weights(opened["open_weights"], np.array([.02, .01, 0]))
    recovered = prepare_economic_recovery(_row(), open_state=opened, recovery_spec=SPEC,
                                          model=recovery_model, selected_model=SELECTED,
                                          pretrade_weights=at_cut, cost_bps=10)
    assert recovered["recovery_action"] is True
    assert recovered["effective_recovery_beta"] == pytest.approx(.6)
    assert recovered["effective_recovery_beta"] > opened["hard_beta_cap"]
    assert recovery_model.features.iloc[0]["current_beta_cap"] == pytest.approx(opened["soft_beta_cap"])
    assert opened["risk_clock"] == clock


def test_emergency_recovery_is_clamped_and_can_abstain_when_no_room():
    opened = _open(equity=.75, model=None, policy="net_value")
    at_cut = mark_to_market_weights(opened["open_weights"], np.array([.02, .01, 0]))
    recovered = prepare_economic_recovery(_row(), open_state=opened, recovery_spec=SPEC,
                                          model=Classifier(.99), selected_model=SELECTED,
                                          pretrade_weights=at_cut, cost_bps=25)
    assert recovered["recovery_action"] is True
    assert recovered["recovery_clipped_by_tail_cap"] is True
    settled = settle_target_session(at_cut, open_target_weights=recovered["target_weights"], asset_returns=[0, 0, 0], cost_bps=25)
    assert 3 * settled["ending_weights"][0] + settled["ending_weights"][1] <= .3 + 1e-12
    no_room = _open(equity=.75, risk_state=3, risk_clock=0, model=None, policy="net_value")
    suppressed = prepare_economic_recovery(_row(), open_state=no_room, recovery_spec=SPEC,
                                           model=Classifier(.99), selected_model=SELECTED)
    assert suppressed["recovery_action"] is False
    assert suppressed["decision_reason"] == "tail_cap_leaves_no_recovery_room"


def test_recovery_does_not_sell_risk_retained_by_the_open_hold_decision():
    row = _row(.1, .15)
    opened = _open(row, pretrade_weights=np.array([.1, .2, .7]), equity=.9,
                   risk_state=2, risk_clock=4, model=NetValue(-100), policy="net_value")
    assert opened["execute_open_target"] is False
    assert opened["soft_beta_cap"] < .45 < opened["holding_beta"]
    actual_at_cut = mark_to_market_weights(opened["open_weights"], np.array([.02, .01, 0]))
    recovered = prepare_economic_recovery(row, open_state=opened, recovery_spec=SPEC,
                                          model=Classifier(.99), selected_model=SELECTED,
                                          pretrade_weights=actual_at_cut, cost_bps=10)
    assert recovered["recovery_action"] is False
    assert recovered["target_weights"] is None
    assert recovered["decision_reason"] == "no_incremental_recovery"
    assert recovered["holding_beta_at_recovery"] > recovered["requested_recovery_beta"]


def test_trailing_loss_ceiling_is_independent_of_soft_target():
    opened = _open(pretrade_weights=np.array([.1, .5, .4]), history=[-.02] * 4,
                   model=NetValue(-100), policy="net_value")
    assert opened["trailing_loss_limit_active"] is True
    assert opened["hard_beta_cap"] == pytest.approx(.6)
    assert opened["tail_beta_cap"] == pytest.approx(.6)
    assert opened["hard_risk_violation"] is True
    assert opened["execute_open_target"] is True


def test_drifted_raw_allocation_cannot_exceed_the_ramp_hard_ceiling():
    # The upstream 2% band can carry passive drift beyond .30 SOXL/.15 cash.
    # The old q*raw_beta soft cap can then exceed q*structural_max_beta.
    opened = _open(_row(.31, .545), pretrade_weights=np.array([.1, .2, .7]),
                   equity=.9, risk_state=2, risk_clock=4, model=None, policy="net_value")
    assert opened["current_beta_cap"] == pytest.approx(.5 * (3 * .31 + .545))
    assert opened["hard_beta_cap"] == pytest.approx(.5 * 1.45)
    assert opened["soft_cap_clipped_by_hard_limit"] is True
    target = opened["desired_open_weights"]
    assert 3 * target[0] + target[1] <= opened["hard_beta_cap"] + 1e-12
    settled = settle_target_session([.1, .2, .7], open_target_weights=opened["execution_target_weights"],
                                     asset_returns=[0, 0, 0], cost_bps=10)
    assert 3 * settled["ending_weights"][0] + settled["ending_weights"][1] <= opened["hard_beta_cap"] + 1e-12
