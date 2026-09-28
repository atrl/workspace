import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import stage1_runtime_release as release_module
from src.workflow.stage1_futu_sim import RuntimeState


class FixedProbability:
    def predict_proba(self, frame):
        return np.tile([.9, .1], (len(frame), 1))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def released(tmp_path, monkeypatch):
    allocation_card = tmp_path / "research" / "allocation"
    allocation_artifacts = allocation_card / "qlib_recorder" / "1" / "allocation_run" / "artifacts" / "allocation"
    (allocation_artifacts / "models").mkdir(parents=True)
    (allocation_artifacts / "selected_model.json").write_text(json.dumps({"candidate": "fixed"}))
    (allocation_artifacts / "models" / "fixed.pkl").write_bytes(b"frozen model identity fixture")
    artifacts = [allocation_artifacts / "selected_model.json", allocation_artifacts / "models" / "fixed.pkl"]
    bundle = {
        "experiment": "allocation", "recorder_id": "allocation_run",
        "training_window": ["2011-01-03", "2022-12-30"],
        "artifacts": [{"path": str(path.relative_to(allocation_card)), "sha256": _hash(path)} for path in artifacts],
    }
    (allocation_card / "frozen_bundle_manifest.json").write_text(json.dumps(bundle))
    (allocation_card / "summary.json").write_text(json.dumps({"experiment": "allocation", "recorder_id": "allocation_run"}))
    action_card = tmp_path / "research" / "action"
    recorder = action_card / "qlib_recorder" / "2" / "action_run"
    artifact_dir = recorder / "artifacts" / "action"
    artifact_dir.mkdir(parents=True)
    model_metadata = {}
    for name in ("frozen_open_action_value_model", "frozen_recovery_model"):
        path = artifact_dir / f"{name}.pkl"
        path.write_bytes(pickle.dumps(FixedProbability()))
        model_metadata[name] = {"artifact": str(path.relative_to(recorder)), "sha256": _hash(path),
                                "training_signal_end": "2022-12-31", "available": True}
    summary = {
        "experiment": "action", "recorder_id": "action_run", "recorder_local_dir": str(recorder),
        "development_gate_pass": True, "selection_uses_outer_test": False, "runtime_models": model_metadata,
        "source_stage1": {"validation_controller_state": release_module.ON_POLICY_STATE,
                          "allocation_source_experiment": "allocation", "allocation_source_recorder_id": "allocation_run",
                          "selected_model": {"action_beta": .6, "minimum_source_alpha_probability": .25, "probability_threshold": .825}},
        "selected_model": {"logistic_c": .1, "execute_probability_threshold": .5,
                           "maximum_skippable_turnover": .05, "mandatory_derisk_beta_delta": 0.0},
    }
    summary_path = action_card / "summary.json"
    summary_path.write_text(json.dumps(summary))
    release = {
        "version": 1, "release_id": "release_fixture", "execution_environment": "SIMULATE",
        "allocation": {"card_path": "research/allocation", "manifest_sha256": _hash(allocation_card / "frozen_bundle_manifest.json"), "recorder_id": "allocation_run"},
        "action": {"card_path": "research/action", "summary_sha256": _hash(summary_path), "recorder_id": "action_run"},
    }
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release))
    frame = pd.DataFrame({"soxl_weight": [.2], "soxx_weight": [.5], "cash_weight": [.3], "alpha_probability": [.5]},
                         index=pd.to_datetime(["2026-09-25"]))
    monkeypatch.setattr(release_module, "infer_frozen_allocation", lambda **_: (frame.copy(), {
        "market_audit": {"common_feature_end": "2026-09-25"}, "smoothing": {"status": "ready"},
    }))
    state = RuntimeState(equity=1.04, peak=1.08, previous=np.array([.2, .5, .3]), risk_state=1, risk_clock=2, history=[.01, -.02])
    kwargs = {"root": tmp_path, "release_path": release_path, "current_prices": {"SOXL": 100.0, "SOXX": 200.0},
              "include_current_intraday": False, "execution_date": "2026-09-28", "live_controller_state": state,
              "current_prices_as_of": "2026-09-28T09:31:00-04:00"}
    return kwargs, summary_path, release_path, artifact_dir


def _amend_summary(summary_path, release_path, update):
    summary = json.loads(summary_path.read_text())
    update(summary)
    summary_path.write_text(json.dumps(summary))
    release = json.loads(release_path.read_text())
    release["action"]["summary_sha256"] = _hash(summary_path)
    release_path.write_text(json.dumps(release))


@pytest.mark.parametrize("case", ["gate_failed", "outer_selection", "foreign_allocation", "shadow_controller"])
def test_unvalidated_or_mismatched_release_cannot_load_models(released, monkeypatch, case):
    kwargs, summary_path, release_path, _ = released
    def change(summary):
        if case == "gate_failed": summary["development_gate_pass"] = False
        elif case == "outer_selection": summary["selection_uses_outer_test"] = True
        elif case == "foreign_allocation": summary["source_stage1"]["allocation_source_recorder_id"] = "different"
        else: summary["source_stage1"]["validation_controller_state"] = "baseline_shadow"
    _amend_summary(summary_path, release_path, change)
    monkeypatch.setattr(release_module.pickle, "load", lambda _: pytest.fail("must not unpickle an unvalidated release"))
    with pytest.raises(ValueError, match="blocked:"):
        release_module.load_runtime_release(kwargs["root"], release_path)


def test_model_digest_is_checked_before_deserialization(released, monkeypatch):
    kwargs, _, release_path, artifact_dir = released
    (artifact_dir / "frozen_open_action_value_model.pkl").write_bytes(b"tampered pickle")
    monkeypatch.setattr(release_module.pickle, "load", lambda _: pytest.fail("must verify before unpickling"))
    with pytest.raises(ValueError, match="hash mismatch"):
        release_module.load_runtime_release(kwargs["root"], release_path)


def test_open_hold_preserves_actual_ledger_weights_and_saves_controller_context(released, monkeypatch):
    kwargs, _, _, _ = released
    def opened(row, **parameters):
        assert parameters["equity"] == 1.04
        assert parameters["history"] == [.01, -.02]
        return {"execute_open_target": False, "open_weights": parameters["pretrade_weights"],
                "drawdown": .04, "current_beta_cap": 1.2, "raw_beta": 1.1,
                "risk_state": 2, "risk_clock": 3, "history": parameters["history"]}
    monkeypatch.setattr(release_module, "prepare_on_policy_open", opened)
    result = release_module.build_release_target(**kwargs)
    assert result["decision"] == "hold"
    assert result["weights"] == {"SOXL": .2, "SOXX": .5, "cash": .3}
    context = result["decision_context"]
    assert context["controller"]["risk_clock"] == 3
    assert context["release_id"] == "release_fixture"
    assert context["quote_as_of"] == "2026-09-28T09:31:00-04:00"
    assert kwargs["live_controller_state"].risk_clock == 2


def test_recovery_reuses_open_controller_without_advancing_again(released, monkeypatch):
    kwargs, _, _, _ = released
    controller = {"drawdown": .08, "current_beta_cap": .3, "raw_beta": 1.1,
                  "risk_state": 2, "risk_clock": 3, "history": [.01, -.02]}
    context = {"controller": controller, "release_id": "release_fixture", "signal_date": "2026-09-25",
               "execution_date": "2026-09-28", "quote_as_of": kwargs["current_prices_as_of"]}
    monkeypatch.setattr(release_module, "prepare_on_policy_open", lambda *_a, **_kw: pytest.fail("must not advance risk state twice"))
    monkeypatch.setattr(release_module, "_with_intraday", lambda frame, **_: frame.assign(early_ret_soxx=.01))
    def recovered(row, **parameters):
        assert parameters["open_state"] == controller
        return {"recovery_action": True, "recovery_probability": .9, "target_weights": np.array([.1, .3, .6])}
    monkeypatch.setattr(release_module, "prepare_on_policy_recovery", recovered)
    result = release_module.build_release_target(**(kwargs | {"include_current_intraday": True, "open_decision_context": context}))
    assert result["decision"] == "rebalance" and result["recovery_action"] is True
    assert result["decision_context"] == context


def test_recovery_without_matching_open_context_is_blocked(released):
    kwargs, _, _, _ = released
    result = release_module.build_release_target(**(kwargs | {"include_current_intraday": True}))
    assert result["status"] == "blocked"
    assert result["reason"] == "missing_matching_open_decision_context"


def test_release_requires_actual_ledger_and_current_price_mark(released):
    kwargs, _, _, _ = released
    result = release_module.build_release_target(**(kwargs | {"live_controller_state": None}))
    assert result["reason"] == "live_paper_ledger_required"
    with pytest.raises(ValueError, match="stale"):
        release_module.build_release_target(**(kwargs | {"current_prices_as_of": "2026-09-25T09:31:00-04:00"}))


def _authorize_failed_trial(kwargs, summary_path, release_path):
    def fail_risk(summary):
        summary["development_gate_pass"] = False
        summary["development_gates"] = {"max_drawdown_not_worse": False, "outer_not_used_for_selection": True}
    _amend_summary(summary_path, release_path, fail_risk)
    release = json.loads(release_path.read_text())
    receipt = {
        "version": 1, "kind": "explicit_user_unpromoted_sim_trial", "execution_environment": "SIMULATE",
        "release_id": release["release_id"], "action_summary_sha256": release["action"]["summary_sha256"],
        "action_recorder_id": "action_run", "allocation_recorder_id": "allocation_run",
        "allocation_manifest_sha256": release["allocation"]["manifest_sha256"],
        "decision": "run_unpromoted_candidate_in_simulate_only",
        "user_instruction": "测试专用授权原文：已知回撤门未过，仅模拟试运行该新候选。",
        "authorized_at": "2020-01-01T10:00:00+08:00",
        "allowed_failed_development_gates": ["max_drawdown_not_worse"],
    }
    receipt_path = kwargs["root"] / "trial_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    release["paper_trial_authorization"] = {"receipt_path": "trial_receipt.json", "receipt_sha256": _hash(receipt_path)}
    release_path.write_text(json.dumps(release))
    return receipt_path


def test_exact_explicit_sim_receipt_allows_only_disclosed_research_gate_failure(released, monkeypatch):
    kwargs, summary_path, release_path, _ = released
    _authorize_failed_trial(kwargs, summary_path, release_path)
    loaded = release_module.load_runtime_release(kwargs["root"], release_path)
    assert loaded["paper_trial_authorization"]["allowed_failed_development_gates"] == ["max_drawdown_not_worse"]
    monkeypatch.setattr(release_module, "prepare_on_policy_open", lambda _row, **p: {
        "execute_open_target": False, "open_weights": p["pretrade_weights"], "drawdown": .04,
        "current_beta_cap": 1.2, "raw_beta": 1.1, "risk_state": 1, "risk_clock": 3, "history": p["history"],
    })
    target = release_module.build_release_target(**kwargs)
    assert target["status"] == "ready" and target["paper_trial_only"] is True
    assert target["development_gate_pass"] is False
    assert target["research_promotion_status"] == "not_promoted"


def test_old_generic_paper_authorization_is_not_a_gate_override(released):
    kwargs, summary_path, release_path, _ = released
    _amend_summary(summary_path, release_path, lambda s: s.update(development_gate_pass=False))
    release = json.loads(release_path.read_text())
    release["user_authorized_paper_trial"] = True
    release_path.write_text(json.dumps(release))
    with pytest.raises(ValueError, match="specific user paper-trial receipt"):
        release_module.load_runtime_release(kwargs["root"], release_path)


@pytest.mark.parametrize("case", ["changed_summary", "non_sim", "different_recorder", "missing_instruction", "future_receipt", "incomplete_gate_list", "scientific_gate"])
def test_trial_receipt_cannot_authorize_other_results_or_bypass_hard_gates(released, case):
    kwargs, summary_path, release_path, _ = released
    receipt_path = _authorize_failed_trial(kwargs, summary_path, release_path)
    receipt = json.loads(receipt_path.read_text())
    if case == "changed_summary":
        _amend_summary(summary_path, release_path, lambda s: s.update(new_evidence="changed"))
    elif case == "non_sim": receipt["execution_environment"] = "REAL"
    elif case == "different_recorder": receipt["allocation_recorder_id"] = "different"
    elif case == "missing_instruction": receipt["user_instruction"] = ""
    elif case == "future_receipt": receipt["authorized_at"] = "2099-01-01T10:00:00+08:00"
    elif case == "incomplete_gate_list": receipt["allowed_failed_development_gates"] = []
    else:
        _amend_summary(summary_path, release_path, lambda s: s["development_gates"].update(outer_not_used_for_selection=False))
        receipt["action_summary_sha256"] = _hash(summary_path)
        receipt["allowed_failed_development_gates"].append("outer_not_used_for_selection")
    receipt_path.write_text(json.dumps(receipt))
    release = json.loads(release_path.read_text())
    release["paper_trial_authorization"]["receipt_sha256"] = _hash(receipt_path)
    release_path.write_text(json.dumps(release))
    with pytest.raises(ValueError, match="blocked:"):
        release_module.load_runtime_release(kwargs["root"], release_path)
