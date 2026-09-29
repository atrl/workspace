import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.workflow import stage1_economic_shadow as shadow

REAL_PREPARE_OPEN = shadow._prepare_open


class FixedValueModel:
    def predict(self, frame):
        return np.full(len(frame), -1.0)


class FixedClassifier:
    def predict_proba(self, frame):
        return np.tile([0.9, 0.1], (len(frame), 1))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def entry(card, path):
    return {"path": str(path.relative_to(card)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def observation(tmp_path, monkeypatch):
    root = tmp_path
    source = root / shadow.SHADOW_SOURCE
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(shadow.__file__).read_bytes())
    policy_source = root / shadow.POLICY_SOURCE
    policy_source.write_text("# sealed fixture policy\n")
    allocation = root / "research/allocation"
    retained = allocation / "qlib_recorder/1/base/artifacts/allocation"
    selected = write_json(retained / "selected_model.json", {"candidate": "fixed"})
    allocation_model = retained / "models/fixed.pkl"
    allocation_model.parent.mkdir()
    allocation_model.write_bytes(pickle.dumps({"outer_refit": {}}))
    allocation_manifest = write_json(allocation / "frozen_bundle_manifest.json", {
        "experiment": "allocation", "recorder_id": "base",
        "artifacts": [entry(allocation, selected), entry(allocation, allocation_model)],
    })
    card = root / "research/economic"
    recorder = card / "qlib_recorder/2/candidate_run"
    model_path = recorder / "artifacts/models/value.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(pickle.dumps(FixedValueModel()))
    classifier = recorder / "artifacts/models/rules.pkl"
    classifier.write_bytes(pickle.dumps(FixedClassifier()))
    common = {"available": True, "training_signal_end": "2025-12-31",
              "label_available_at_max": "2026-01-02T14:30:00+00:00"}
    summary = write_json(card / "summary.json", {
        "experiment": "economic", "recorder_id": "candidate_run", "recorder_relative_dir": "qlib_recorder/2/candidate_run",
        "development_gate_pass": False,
        "allocation_source": {"card_path": "research/allocation", "recorder_id": "base", "manifest_sha256": shadow._sha(allocation_manifest)},
        "shadow_models": {
            "value": common | {"policy": "net_value", "artifact": "artifacts/models/value.pkl", "sha256": shadow._sha(model_path)},
            "rules": common | {"policy": "rules_only", "artifact": "artifacts/models/rules.pkl", "sha256": shadow._sha(classifier)},
        },
    })
    manifest = write_json(card / "frozen_bundle_manifest.json", {
        "experiment": "economic", "recorder_id": "candidate_run",
        "artifacts": [entry(card, summary), entry(card, model_path), entry(card, classifier)],
        "code_sha256": {shadow.POLICY_SOURCE: shadow._sha(policy_source), shadow.SHADOW_SOURCE: shadow._sha(source)},
    })
    ledger_path = write_json(root / "runtime/soxl/stage1_futu_sim/paper_ledger_v2.json", {
        "execution_environment": "SIMULATE", "initial_equity": 1000, "cash": 600,
        "positions": {"SOXL": 20, "SOXX": 10}, "prices": {"SOXL": 10, "SOXX": 20},
        "equity": 1000, "peak_equity": 1100,
        "settled_marks": [{"date": "2026-09-28", "equity": 1000, "drawdown": -0.09}],
        "fills": [{"create_time": "2026-09-28 09:31:00"}],
        "reconciliation": {"status": "matched", "as_of": "2026-09-29T09:31:00-04:00"},
    })
    state_path = write_json(ledger_path.parent / "execution_state_v2.json", {"pending_slots": {}, "completed_slots": {}})
    frame = pd.DataFrame([{"soxl_weight": .2, "soxx_weight": .2, "cash_weight": .6,
                           "tail_probability": .1, "alpha_probability": .8}], index=pd.to_datetime(["2026-09-28"]))
    monkeypatch.setattr(shadow, "infer_frozen_allocation", lambda **_kw: (frame, {"smoothing": {"status": "ready"}}))
    monkeypatch.setattr(shadow, "_calendar", lambda start, end: pd.date_range(start, end, freq="B"))
    def hold(_row, **kwargs):
        weights = kwargs["pretrade_weights"]
        return {"open_weights": weights, "execution_target_weights": weights,
                "desired_open_weights": np.array([.2, .2, .6]), "execute_open_target": False,
                "hard_beta_cap": 1.4, "soft_beta_cap": .8, "decision_reason": "negative_net_value",
                "predicted_net_advantage_bps": -1, "features": {"days_since_trade": kwargs["days_since_trade"]}}
    monkeypatch.setattr(shadow, "_prepare_open", hold)
    return dict(root=root, card_path="research/economic", candidate="value", expected_manifest_sha256=shadow._sha(manifest),
                ledger_path=ledger_path, execution_state_path=state_path, prices={"SOXL": 11, "SOXX": 22},
                quote_as_of="2026-09-29T09:31:20-04:00", execution_date="2026-09-29",
                now=datetime(2026, 9, 29, 9, 31, 30, tzinfo=ZoneInfo("America/New_York")))


def test_shadow_failed_candidate_marks_real_shares_without_mutating_account(observation):
    hashes = {key: shadow._sha(observation[key]) for key in ("ledger_path", "execution_state_path")}
    result = shadow.build_economic_open_shadow(**observation)
    assert result["account_equity_at_supplied_prices"] == 1040
    assert result["actual_pretrade_weights"] == pytest.approx([220 / 1040, 220 / 1040, 600 / 1040])
    assert result["integer_plan"]["target_quantities"] == {"SOXL": 20, "SOXX": 10}
    assert result["integer_plan"]["nonzero_changes"] == []
    assert result["orders_submitted"] == 0 and result["submit_supported"] is False
    assert result["own_state_simulation"] is False and result["research_promotion_status"] == "not_promoted"
    assert result["days_since_trade"] == 0  # Monday trade -> Tuesday decision has no completed HOLD session.
    for key, digest in hashes.items():
        assert shadow._sha(observation[key]) == digest


def test_classifier_candidate_uses_separate_read_only_policy(observation):
    result = shadow.build_economic_open_shadow(**(observation | {"candidate": "rules"}))
    assert result["policy"] == "rules_only"


def test_real_economic_policy_interface_keeps_legal_shares_on_negative_value(observation, monkeypatch):
    from src.workflow import stage1_economic_action
    policy_path = observation["root"] / shadow.POLICY_SOURCE
    policy_path.write_bytes(Path(stage1_economic_action.__file__).read_bytes())
    manifest_path = observation["root"] / observation["card_path"] / "frozen_bundle_manifest.json"
    manifest = shadow._read(manifest_path)
    manifest["code_sha256"][shadow.POLICY_SOURCE] = shadow._sha(policy_path)
    write_json(manifest_path, manifest)
    observation["expected_manifest_sha256"] = shadow._sha(manifest_path)
    monkeypatch.setattr(shadow, "_prepare_open", REAL_PREPARE_OPEN)
    result = shadow.build_economic_open_shadow(**observation)
    assert result["decision"] == "hold"
    assert result["integer_plan"]["target_quantities"] == {"SOXL": 20, "SOXX": 10}
    assert result["hard_beta_cap"] > result["soft_beta_cap"]


def test_stale_provider_calendar_cannot_certify_its_own_signal_as_fresh(observation, monkeypatch):
    frame = pd.DataFrame([{"soxl_weight": .2, "soxx_weight": .2, "cash_weight": .6}], index=pd.to_datetime(["2026-09-25"]))
    monkeypatch.setattr(shadow, "infer_frozen_allocation", lambda **_kw: (frame, {"smoothing": {"status": "ready"}}))
    monkeypatch.setattr(shadow, "_calendar", lambda *_a: pd.DatetimeIndex(["2026-09-25"]))
    with pytest.raises(ValueError, match="stale against the exchange calendar"):
        shadow.build_economic_open_shadow(**observation)


def test_model_and_source_tampering_fail_closed(observation):
    card = observation["root"] / observation["card_path"]
    (card / "qlib_recorder/2/candidate_run/artifacts/models/value.pkl").write_bytes(b"not the sealed model")
    with pytest.raises(ValueError, match="hash mismatch"):
        shadow.build_economic_open_shadow(**observation)


@pytest.mark.parametrize("case", ["stale_quote", "future_quote", "outside_window", "pending", "future_settlement", "future_fill"])
def test_forward_requires_causal_observations(observation, case):
    if case == "stale_quote":
        observation["now"] = observation["now"].replace(minute=35)
    elif case == "future_quote":
        observation["quote_as_of"] = "2026-09-29T09:33:20-04:00"
    elif case == "outside_window":
        observation["now"] = observation["now"].replace(minute=40)
        observation["quote_as_of"] = "2026-09-29T09:40:00-04:00"
    elif case == "pending":
        write_json(observation["execution_state_path"], {"pending_slots": {"x": {}}})
    elif case == "future_settlement":
        ledger = shadow._read(observation["ledger_path"])
        ledger["settled_marks"][0]["date"] = "2026-09-29"
        write_json(observation["ledger_path"], ledger)
    else:
        ledger = shadow._read(observation["ledger_path"])
        ledger["fills"][0]["create_time"] = "2026-09-29 09:31:45"
        write_json(observation["ledger_path"], ledger)
    with pytest.raises(ValueError, match="blocked"):
        shadow.build_economic_open_shadow(**observation)


def test_snapshot_cannot_consume_forward_dedup_slot(observation):
    first = shadow.run_once(**(observation | {"mode": "snapshot"}))
    second = shadow.run_once(**observation)
    repeated = shadow.run_once(**observation)
    assert first["observation_mode"] == "snapshot" and second["observation_mode"] == "forward"
    assert first["observation_path"] != second["observation_path"]
    assert repeated["status"] == "already_recorded"
    assert "stage1_economic_shadow/forward/" in second["observation_path"]


def test_integer_preview_uses_pre_fee_execution_target_and_checks_fee_scenarios(observation, monkeypatch):
    def execute(_row, **kwargs):
        return {"open_weights": np.array([.2, .2, .6]), "desired_open_weights": np.array([.2, .2, .6]),
                "execution_target_weights": np.array([.199, .199, .602]), "execute_open_target": True,
                "hard_beta_cap": 1.4, "soft_beta_cap": .8, "decision_reason": "positive_net_value", "features": {}}
    monkeypatch.setattr(shadow, "_prepare_open", execute)
    result = shadow.build_economic_open_shadow(**observation)
    plan = result["integer_plan"]
    assert plan["target_quantities"] == {"SOXL": 18, "SOXX": 9}
    assert set(plan["same_integer_plan_fee_scenarios"]) == {"10bps", "25bps"}
    assert plan["same_integer_plan_fee_scenarios"]["10bps"]["assumed_fee"] == pytest.approx(.044)
    assert plan["planning_cost_hard_limits_pass"]


def test_existing_sim_planner_creates_no_orders_when_floor_quantities_do_not_change(tmp_path, monkeypatch):
    # The sender is imported in this test only. The shadow module never imports it.
    from src.api.futu_sim import FutuSimulatedBroker
    broker = FutuSimulatedBroker()
    forbidden = SimpleNamespace(place_order=lambda **_kw: pytest.fail("must never submit unchanged quantities"))
    monkeypatch.setattr(broker, "_require_connected", lambda: (SimpleNamespace(), forbidden, 0))
    monkeypatch.setattr(broker, "account_snapshot", lambda: {"cash": 600, "positions": {"SOXL": 20, "SOXX": 10}})
    state = tmp_path / "execution.json"
    result = broker.rebalance(weights={"SOXL": .201, "SOXX": .202, "cash": .597},
                              prices={"SOXL": 10, "SOXX": 20}, portfolio_equity=1000,
                              signal_date="2026-09-28", slot="open", state_path=state, dry_run=True,
                              current_quantities={"SOXL": 20, "SOXX": 10})
    assert result["orders"] == [] and result["target_quantities"] == {"SOXL": 20, "SOXX": 10}
    assert not state.exists()
