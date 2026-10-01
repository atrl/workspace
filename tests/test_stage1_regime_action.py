import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow import stage1_regime_action as regime


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


class _ExplodingModel:
    def predict(self, frame):
        raise RuntimeError("estimator failed")


def _observations(periods=400):
    t = np.arange(periods)
    return pd.DataFrame({"$close": 100 * np.exp(.0005 * t + .03 * np.sin(t / 8)),
                         "$volume": 100000 + 20000 * np.cos(t / 7)},
                        index=pd.bdate_range("2015-09-01", periods=periods))


def _frame(periods=275, start="2016-01-04"):
    dates = pd.bdate_range(start, periods=periods)
    observations = _observations(periods + 400)
    features = regime.compute_regime_features(observations)
    # Synthetic feature dates are assigned explicitly, without a stale-date join.
    features = features.iloc[62:62 + periods].copy()
    features.index = dates
    frame = pd.DataFrame({
        "signal_date": dates, "trade_date": dates + pd.offsets.BDay(1), "exit_date": dates + pd.offsets.BDay(2),
        "segment": "development_walk_forward", "soxl_return": np.sin(np.arange(periods)) * .003,
        "soxx_return": np.cos(np.arange(periods)) * .001, "soxl_weight": .2,
        "soxx_weight": .5, "cash_weight": .3, "tail_probability": .1, "alpha_probability": .6,
        "early_ret_soxl": np.nan, "early_ret_soxx": np.nan,
    })
    return regime.attach_regime_features(frame, features)


def _kwargs():
    return {"recovery_models": {}, "source_spec": IntradayRecoveryDirectAllocationSpec(root=Path(".")),
            "selected_model": {"action_beta": .6, "minimum_source_alpha_probability": .25, "probability_threshold": .825}}


def _open(row, *, model, spec=None, weights=None, equity=1., peak=1.):
    return regime.prepare_regime_open(row, pretrade_weights=np.array([0., 0., 1.]) if weights is None else weights,
        equity=equity, peak=peak, risk_state=0, risk_clock=0, history=[], recovery_spec=_kwargs()["source_spec"],
        model=model, spec=spec or regime.RegimeActionSpec("test"))


def test_probabilities_are_causal_finite_and_suffix_invariant():
    raw = _observations()
    first = regime.compute_regime_features(raw)
    altered = raw.copy()
    altered.iloc[220:, 0] *= np.linspace(1, 3, len(altered) - 220)
    second = regime.compute_regime_features(altered)
    pd.testing.assert_frame_equal(first.iloc[:220], second.iloc[:220])
    assert first.iloc[:62][list(regime.REGIME_PROBABILITIES)].isna().all(axis=1).all()
    mature = first.iloc[62:]
    regime._validate_regime(mature)
    np.testing.assert_allclose(mature[list(regime.REGIME_PROBABILITIES)].sum(axis=1), 1., atol=1e-12)
    assert len(first) == len(raw)


def test_exact_date_attachment_rejects_missing_warmup_and_duplicate_observations():
    features = regime.compute_regime_features(_observations()).iloc[62:]
    frame = pd.DataFrame({"signal_date": features.index[:3]})
    attached = regime.attach_regime_features(frame, features)
    assert len(attached) == 3
    with pytest.raises(ValueError, match="exact signal-date"):
        regime.attach_regime_features(frame, features.iloc[1:])
    with pytest.raises(ValueError, match="warmup"):
        regime.attach_regime_features(frame, regime.compute_regime_features(_observations()))
    with pytest.raises(ValueError, match="attachment"):
        regime.attach_regime_features(frame, pd.concat([features, features.iloc[:1]]))


def test_qlib_loader_uses_calendar_and_features_with_no_future_query(monkeypatch, tmp_path):
    import qlib

    raw = _observations()
    calls = []
    monkeypatch.setattr(qlib, "init", lambda **kw: calls.append(("init", kw)))

    def features(instruments, fields, **kw):
        calls.append(("features", {"instruments": instruments, "fields": fields, **kw}))
        result = raw.loc[kw["start_time"]:kw["end_time"], fields].copy()
        result.index = pd.MultiIndex.from_arrays([np.full(len(result), "SOXX"), result.index], names=["instrument", "datetime"])
        return result

    monkeypatch.setattr("qlib.data.D", SimpleNamespace(features=features,
        calendar=lambda **kw: raw.index[raw.index <= pd.Timestamp(kw["end_time"])].to_numpy()))
    result = regime.load_regime_features(tmp_path, raw.index[80], raw.index[100])
    assert len(result) == 21
    assert calls[-1][1]["end_time"] == raw.index[100]
    assert calls[-1][1]["fields"] == ["$close"]
    assert result.attrs["regime_source"]["source_api"] == "qlib.data.D.features"
    with pytest.raises(ValueError, match="62 prior"):
        regime.load_regime_features(tmp_path, raw.index[20], raw.index[100])


def test_price_only_scope_does_not_fill_missing_close():
    raw = _observations().drop(columns="$volume")
    regime._validate_regime(regime.compute_regime_features(raw).iloc[62:])
    raw.iloc[90, 0] = np.nan
    with pytest.raises(ValueError, match="invalid Qlib SOXX close"):
        regime.compute_regime_features(raw)


def test_missing_regime_features_fail_closed_even_with_no_model():
    row = _frame(1).iloc[0].drop("regime_momentum_60")
    with pytest.raises(ValueError, match="missing causal regime"):
        _open(row, model=None)
    opened = _open(row, model=None, spec=regime.RegimeActionSpec("ablation", use_regime=False))
    assert opened["execute_open_target"]


def test_hard_risk_repair_cannot_be_held_and_derisk_protection_is_explicit():
    row = _frame(1).iloc[0]
    negative = _ConstantModel(-1000.)
    repair = _open(row, model=negative, weights=np.array([.8, .1, .1]))
    assert repair["hard_risk_violation"] and repair["execute_open_target"]
    assert repair["decision_reason"] == "hard_risk_repair"
    protected = _open(row, model=negative, weights=np.array([.25, .5, .25]))
    assert protected["protected_derisk"] and protected["execute_open_target"]
    ordinary = _open(row, model=negative, spec=regime.RegimeActionSpec("unprotected", protect_derisk=False),
                     weights=np.array([.25, .5, .25]))
    assert not ordinary["hard_risk_violation"] and not ordinary["execute_open_target"]


@pytest.mark.parametrize("protect_derisk", [True, False])
def test_predictor_failure_cannot_suppress_hard_repair_under_either_policy(protect_derisk):
    row = _frame(1).iloc[0]
    spec = regime.RegimeActionSpec("hard_repair", protect_derisk=protect_derisk)
    expected = _open(row, model=_ConstantModel(-1000.), spec=spec, weights=np.array([.8, .1, .1]))
    result = _open(row, model=_ExplodingModel(), spec=spec, weights=np.array([.8, .1, .1]))
    assert result["hard_risk_violation"] and result["execute_open_target"]
    assert result["decision_reason"] == "hard_risk_repair"
    assert not result["prediction_evaluated"] and np.isnan(result["predicted_net_advantage_bps"])
    np.testing.assert_array_equal(result["execution_target_weights"], expected["execution_target_weights"])
    with pytest.raises(ValueError, match="missing causal regime"):
        _open(row.drop("regime_momentum_20"), model=_ExplodingModel(), spec=spec, weights=np.array([.8, .1, .1]))


def test_only_explicitly_protected_soft_cut_bypasses_prediction_failure():
    row = _frame(1).iloc[0]
    weights = np.array([.25, .5, .25])
    protected = _open(row, model=_ExplodingModel(), weights=weights)
    assert protected["execute_open_target"] and protected["protected_derisk"]
    assert not protected["prediction_evaluated"]
    soft_spec = regime.RegimeActionSpec("soft_beta", protect_derisk=False)
    held = _open(row, model=_ConstantModel(-1000.), spec=soft_spec, weights=weights)
    assert not held["hard_risk_violation"] and not held["protected_derisk"]
    assert not held["execute_open_target"] and held["prediction_evaluated"]
    np.testing.assert_array_equal(held["open_weights"], weights)
    with pytest.raises(RuntimeError, match="estimator failed"):
        _open(row, model=_ExplodingModel(), spec=soft_spec, weights=weights)


def test_unprotected_no_regime_control_reproduces_economic_own_policy_training():
    from src.workflow.stage1_economic_training import train_causal_economic_models

    frame = _frame()
    old = train_causal_economic_models(frame, **_kwargs())
    spec = regime.RegimeActionSpec("exact_control", use_regime=False, protect_derisk=False)
    new = regime.train_regime_models(frame, spec=spec, **_kwargs())
    for column in ("strategy_return", "turnover", "fees", "equity", "risk_state", "risk_clock", "ending_soxl_weight", "ending_soxx_weight"):
        np.testing.assert_array_equal(new["replay"][column], old["replay"][column])
    np.testing.assert_array_equal(new["models"][2017].named_steps["ridge"].coef_, old["models"][2017].named_steps["ridge"].coef_)
    np.testing.assert_array_equal(new["training_rows"].execute_advantage_bps, old["training_rows"].execute_advantage_bps)


def test_q_lib_dataset_only_accepts_strictly_mature_prior_year_labels():
    spec = regime.RegimeActionSpec("maturity")
    rows = pd.DataFrame([{name: 1. for name in regime.regime_feature_names(spec)} for _ in range(4)])
    rows["signal_date"] = pd.to_datetime(["2021-12-28", "2021-12-29", "2021-12-30", "2022-01-03"])
    rows["execute_advantage_bps"] = [1., 2., 3., 4.]
    rows["label_available_at"] = pd.to_datetime([
        "2021-12-31T14:31:00Z", "2022-01-01T05:00:00Z", "2022-01-03T14:31:00Z", "2021-12-31T14:31:00Z"], utc=True)
    dataset = regime.build_regime_dataset(rows, spec=spec)
    mature, cutoff = regime.mature_regime_rows(dataset, spec=spec, model_year=2022,
        first_decision_at=pd.Timestamp("2022-01-04T14:31:00Z"))
    assert cutoff == pd.Timestamp("2022-01-01T05:00:00Z")
    assert mature.execute_advantage_bps.tolist() == [1.]


def test_own_policy_hold_path_preserves_cash_and_cost_stress_changes_actual_path():
    frame = _frame(12)
    spec = regime.RegimeActionSpec("own_path", protect_derisk=False)
    held = regime.simulate_regime_policy(frame, models={2016: _ConstantModel(-1000.)}, spec=spec, **_kwargs())
    assert held.strategy_return.eq(0).all() and held.turnover.eq(0).all()
    assert held.days_since_trade.tolist() == list(range(12))
    first = regime.simulate_regime_policy(frame, models={}, spec=spec, cost_bps=10., **_kwargs())
    stressed = regime.simulate_regime_policy(frame, models={}, spec=spec, cost_bps=25., **_kwargs())
    assert stressed.fees.sum() > first.fees.sum()
    assert stressed.iloc[-1].equity < first.iloc[-1].equity
    assert first.iloc[0].fees > 0
    assert first.iloc[-1].ending_cash_weight != held.iloc[-1].ending_cash_weight


def test_training_future_suffix_cannot_change_prior_model_and_pickle_replay():
    frame = _frame()
    spec = regime.RegimeActionSpec("suffix", protect_derisk=False)
    first = regime.train_regime_models(frame, spec=spec, **_kwargs())
    altered = frame.copy()
    future = altered.signal_date.dt.year.eq(2017)
    altered.loc[future, ["soxl_return", "soxx_return"]] = [.2, .1]
    altered.loc[future, "regime_momentum_20"] = 9.
    second = regime.train_regime_models(altered, spec=spec, **_kwargs())
    assert first["models"][2016] is None
    np.testing.assert_allclose(first["models"][2017].named_steps["ridge"].coef_,
                               second["models"][2017].named_steps["ridge"].coef_, atol=1e-12)
    assert first["diagnostics"].iloc[0].training_sha256 == second["diagnostics"].iloc[0].training_sha256
    original = regime.simulate_regime_policy(frame, models=first["models"], spec=spec, **_kwargs())
    restored = regime.simulate_regime_policy(frame, models=pickle.loads(pickle.dumps(first["models"])), spec=spec, **_kwargs())
    pd.testing.assert_frame_equal(original, restored)
    pd.testing.assert_frame_equal(original, first["replay"])
    assert first["dataset_audit"]["training_state"] == "causal_own_policy_path"


def test_development_only_final_refit_needs_no_outer_observation():
    frame = _frame(12, start="2022-01-03")
    spec = regime.RegimeActionSpec("final_fit", minimum_training_rows=2)
    result = regime.train_regime_models(frame, spec=spec, update_through_year=2022, **_kwargs())
    assert result["final_model_year"] == 2023
    assert result["final_model"] is result["models"][2023]
    assert result["diagnostics"].iloc[-1].fit_trigger == "development_end_refit"
    assert result["diagnostics"].iloc[-1].training_rows == 11
    assert pd.Timestamp(result["diagnostics"].iloc[-1].label_available_at_max) < pd.Timestamp("2023-01-01", tz="America/New_York")
    assert result["training_rows"].iloc[-1].label_status == "pending_next_reset_target"
    wrong_spec = replace(spec, ridge_alpha=100.)
    with pytest.raises(ValueError, match="specification differ"):
        _open(frame.iloc[0], model=result["final_model"], spec=wrong_spec)


def test_histogram_model_has_bounded_deterministic_fit():
    spec = regime.RegimeActionSpec("hist", estimator="hist_gradient_boosting", max_iter=3, minimum_training_rows=2)
    result = regime.train_regime_models(_frame(10, start="2022-01-03"), spec=spec, update_through_year=2022, **_kwargs())
    assert result["final_model"].max_leaf_nodes == 7
    assert result["final_model"].random_state == 17
    with pytest.raises(ValueError, match="bounded"):
        regime.RegimeActionSpec("too_big", max_leaf_nodes=20)
