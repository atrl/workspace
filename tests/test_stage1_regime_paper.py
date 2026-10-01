from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.backtest.position_accounting import settle_target_session
from src.workflow import stage1_regime_paper as paper
from src.workflow.stage1_economic_action import fee_affordable_target


def initial():
    return paper.new_paper_state(candidate_identity={"release": "sealed_candidate", "model_sha256": "a" * 64},
                                 initial_cash=1000, cost_bps=10)


def event(state, session="2026-10-01", slot="open", prices=None, **updates):
    time = "09:31:20" if slot == "open" else "10:31:20"
    value = {"session": session, "slot": slot, "observed_at": f"{session}T{time}-04:00",
             "quote_as_of": f"{session}T{time}-04:00", "prices": prices or {"SOXL": 10., "SOXX": 20.},
             "input_sha256": "b" * 64, "candidate_sha256": state["candidate_sha256"],
             "features": {"alpha_probability": .7}, "signal_date": "2026-09-30", "previous_session": "2026-09-30"}
    return value | updates


def buy(*, row, state, pretrade_weights):
    return {"execute_open_target": True, "execution_target_weights": np.array([.2, .5, .3]),
            "risk_state": 1, "risk_clock": state["risk_clock"] + 1,
            "history": state["history"], "decision_reason": "test_buy"}


def hold(*, row, state, pretrade_weights):
    return {"execute_open_target": False, "execution_target_weights": pretrade_weights,
            "risk_state": state["risk_state"], "risk_clock": state["risk_clock"] + 1,
            "history": state["history"], "decision_reason": "test_hold"}


def no_recovery(*, row, state, pretrade_weights, open_state):
    return {"recovery_action": False, "target_weights": None, "decision_reason": "no_recovery"}


def process(state, value, open_decider=buy, recovery_decider=no_recovery):
    return paper.process_paper_event(state, value, open_decider=open_decider, recovery_decider=recovery_decider)


def test_candidate_book_continues_own_fractional_positions_and_net_fees():
    state = initial()
    original = deepcopy(state)
    opened, receipt = process(state, event(state))
    assert state == original
    assert opened["positions"] == {"SOXL": 20., "SOXX": 25.}
    assert opened["cash"] == pytest.approx(299.3)
    assert receipt["trade"]["assumed_fee"] == pytest.approx(.7)
    recovered, _ = process(opened, event(opened, slot="recovery", prices={"SOXL": 11., "SOXX": 22.}))
    assert recovered["risk_clock"] == opened["risk_clock"] == 1
    assert recovered["positions"] == opened["positions"]
    next_open = event(recovered, session="2026-10-02", prices={"SOXL": 12., "SOXX": 24.},
                      signal_date="2026-10-01", previous_session="2026-10-01")
    final, receipt = process(recovered, next_open, open_decider=hold)
    assert final["history"] == pytest.approx([.1393])
    assert final["equity"] == pytest.approx(1139.3)
    assert final["peak"] == pytest.approx(1139.3)
    assert final["risk_clock"] == 2
    assert final["days_since_trade"] == 0
    assert receipt["own_state_simulation"] and receipt["orders_submitted"] == 0


def test_intraday_recovery_matches_shared_session_cash_accounting():
    state = initial()
    opened, _ = process(state, event(state))
    seen = {}

    def recovery(**kwargs):
        seen.update(kwargs)
        target = fee_affordable_target(np.array([.3, .5, .2]),
                                      pretrade_weights=kwargs["pretrade_weights"], cost_bps=10)
        return {"recovery_action": True, "target_weights": target}

    recovered, receipt = process(opened, event(opened, slot="recovery", prices={"SOXL": 11., "SOXX": 18.}),
                                  recovery_decider=recovery)
    assert seen["open_state"]["risk_clock"] == 1
    assert seen["open_state"]["open_weights"] == pytest.approx([200 / 999.3, 500 / 999.3, 299.3 / 999.3])
    target = receipt["decision"]["target_weights"]
    expected = settle_target_session([0., 0., 1.], open_target_weights=[.2, .5, .3],
        asset_returns=[.2, .1, 0.], cost_bps=10, recovery_target_weights=target, early_returns=[.1, -.1, 0.])
    next_event = event(recovered, session="2026-10-02", prices={"SOXL": 12., "SOXX": 22.},
                       signal_date="2026-10-01", previous_session="2026-10-01")
    final, _ = process(recovered, next_event, open_decider=hold)
    assert final["equity"] == pytest.approx(1000 * expected["net_equity"])
    assert final["history"] == pytest.approx([expected["net_return"]])


def test_no_trade_sessions_increment_only_when_completed():
    state = initial()
    opened, _ = process(state, event(state), open_decider=hold)
    assert opened["days_since_trade"] == 0
    recovered, _ = process(opened, event(opened, slot="recovery"))
    final, _ = process(recovered, event(recovered, session="2026-10-02", signal_date="2026-10-01",
                                       previous_session="2026-10-01"), open_decider=hold)
    assert final["days_since_trade"] == 1 and final["history"] == [0.0]


def test_identical_event_is_idempotent_changed_event_is_rejected():
    state = initial()
    value = event(state)
    first, _ = process(state, value)
    duplicate, receipt = process(first, value)
    assert duplicate == first and receipt["status"] == "already_recorded"
    with pytest.raises(ValueError, match="different inputs"):
        process(first, value | {"prices": {"SOXL": 10.1, "SOXX": 20.}})


def test_missing_recovery_and_skipped_open_block_without_hindsight():
    state = initial()
    opened, _ = process(state, event(state))
    with pytest.raises(ValueError, match="previous recovery observation is missing"):
        process(opened, event(opened, session="2026-10-02", signal_date="2026-10-01", previous_session="2026-10-01"))
    recovered, _ = process(opened, event(opened, slot="recovery"))
    with pytest.raises(ValueError, match="missing intervening open"):
        process(recovered, event(recovered, session="2026-10-05", signal_date="2026-10-02", previous_session="2026-10-02"))


def test_recovery_cannot_seed_a_book_or_advance_another_session():
    state = initial()
    with pytest.raises(ValueError, match="same-session open"):
        process(state, event(state, slot="recovery"))
    opened, _ = process(state, event(state))
    with pytest.raises(ValueError, match="same-session open"):
        process(opened, event(opened, session="2026-10-02", slot="recovery"))


@pytest.mark.parametrize("patch", [
    {"observed_at": "2026-10-01T09:40:00-04:00"},
    {"quote_as_of": "2026-10-01T09:31:26-04:00"},
    {"quote_as_of": "2026-10-01T09:29:00-04:00"},
    {"observed_at": "2026-10-01T09:31:20"},
    {"feature_available_at": "2026-10-01T10:30:00-04:00"},
    {"candidate_sha256": "c" * 64},
    {"input_sha256": "unsealed"},
    {"signal_date": "2026-10-01"},
    {"prices": {"SOXL": float("nan"), "SOXX": 20.}},
])
def test_unsealed_noncausal_or_stale_input_is_rejected_without_mutation(patch):
    state = initial()
    before = deepcopy(state)
    with pytest.raises(ValueError, match="blocked"):
        process(state, event(state, **patch))
    assert state == before


def test_evidence_failure_does_not_commit_state(tmp_path):
    state = initial()
    value = event(state)
    path = tmp_path / "independent/book.json"

    def fail(*args):
        raise RuntimeError("qlib recorder unavailable")

    with pytest.raises(RuntimeError, match="recorder unavailable"):
        paper.commit_paper_event(state_path=path, event=value, initial_state=state, open_decider=buy,
            recovery_decider=no_recovery, evidence_writer=fail, now=datetime.fromisoformat(value["observed_at"]))
    assert not path.exists()


def test_commit_is_atomic_idempotent_and_rejects_past_replay(tmp_path):
    state = initial()
    value = event(state)
    saved = []
    path = tmp_path / "book.json"
    kwargs = dict(state_path=path, event=value, initial_state=state, open_decider=buy,
        recovery_decider=no_recovery, evidence_writer=lambda *args: saved.append(args),
        now=datetime.fromisoformat(value["observed_at"]))
    paper.commit_paper_event(**kwargs)
    contents = path.read_bytes()
    receipt = paper.commit_paper_event(**kwargs)
    assert receipt["status"] == "already_recorded" and len(saved) == 1
    assert path.read_bytes() == contents and not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="accounting assumptions changed"):
        paper.commit_paper_event(**(kwargs | {"initial_state": state | {"cost_bps": 25.0}}))
    with pytest.raises(ValueError, match="stale or future event"):
        paper.commit_paper_event(**(kwargs | {"now": datetime.fromisoformat("2026-10-01T10:00:00-04:00")}))


def test_native_qlib_recorder_loads_only_hash_verified_release_objects(tmp_path):
    import qlib
    from qlib.workflow import R

    root = tmp_path
    source = root / "src/workflow/stage1_regime_paper.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(paper.__file__).read_bytes())
    uri = root / "qlib_recorder"
    provider = root / "provider"
    provider.mkdir()
    qlib.init(provider_uri=str(provider), region="us")
    with R.start(experiment_name="paper_loader_test", uri=str(uri)):
        recorder = R.get_recorder()
        recorder.save_objects(**{"model.pkl": {"frozen_model": "fixture"}})
        rid = recorder.id
        artifact = Path(recorder.get_local_dir()) / "artifacts/model.pkl"
    manifest = {"execution_mode": "INDEPENDENT_PAPER", "recorder_uri": "qlib_recorder",
        "experiment_name": "paper_loader_test", "recorder_id": rid,
        "code_sha256": {"src/workflow/stage1_regime_paper.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "objects": {"model": {"path": "model.pkl", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}}
    path = root / "release.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, objects = paper.load_sealed_recorder_objects(root=root, manifest_path=path,
                                                        expected_manifest_sha256=digest)
    assert loaded["recorder_id"] == rid and objects["model"] == {"frozen_model": "fixture"}
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="object hash mismatch"):
        paper.load_sealed_recorder_objects(root=root, manifest_path=path, expected_manifest_sha256=digest)
