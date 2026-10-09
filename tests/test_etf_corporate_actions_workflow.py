"""Synthetic failure tests; never run the registered market qualification."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from src.workflow import etf_corporate_actions as workflow


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def test_immutable_source_copy_is_independent_and_reuse_needs_every_file(tmp_path):
    staged = tmp_path / "staged"
    staged.write_bytes(b"original source\n")
    entry = {"staged_input": str(staged), "archive_relative_path": "issuer/source.bin",
             "sha256": sha(staged.read_bytes()), "bytes": staged.stat().st_size}
    contract, destination = {"files": [entry]}, tmp_path / "archive"
    first = workflow.copy_immutable_sources(contract, destination)
    assert first["reused"] is False
    target = destination / entry["archive_relative_path"]
    assert target.stat().st_ino != staged.stat().st_ino
    staged.unlink()
    assert workflow.copy_immutable_sources(contract, destination)["reused"] is True
    (destination / "unexpected").write_text("not registered")
    with pytest.raises(ValueError, match="immutable source archive differs"):
        workflow.copy_immutable_sources(contract, destination)
    assert target.read_bytes() == b"original source\n"


@pytest.mark.parametrize("bad", ["sha256", "bytes", "traversal"])
def test_invalid_source_inventory_writes_nothing(tmp_path, bad):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    item = {"staged_input": str(source), "archive_relative_path": "ok/body", "sha256": sha(b"data"), "bytes": 4}
    if bad == "traversal": item["archive_relative_path"] = "../escape"
    elif bad == "bytes": item["bytes"] = 3
    else: item["sha256"] = "0" * 64
    destination = tmp_path / "new_archive"
    with pytest.raises(ValueError):
        workflow.copy_immutable_sources({"files": [item]}, destination)
    assert not destination.exists()


def test_reference_comparison_uses_decimal_equivalence_without_replacing_tokens():
    columns = ["Record Date", "Ex-Date", "Payable Date", "Total Distribution", "Return of Capital"]
    parsed = pd.DataFrame([["2020-01-02", "2020-01-02", "2020-01-03", "0.123456789123456789", "0"]], columns=columns)
    before = parsed.copy(deep=True)
    reference = b"Record Date,Ex-Date,Payable Date,Total Distribution,Return of Capital\n2020-01-02,2020-01-02,2020-01-03,0.123456789123456789,0.0\n"
    assert workflow.compare_extraction_reference(parsed, reference, source_fields=columns)["amount_equivalence"] == "Decimal_exact"
    pd.testing.assert_frame_equal(parsed, before)
    with pytest.raises(ValueError, match="extraction reference differs"):
        workflow.compare_extraction_reference(parsed, reference.replace(b"123456789123456789", b"123456789123456780"), source_fields=columns)


def test_receipts_keep_failed_bodies_and_unknown_times(tmp_path):
    files = []
    for name, content, kind in [
        ("good.bin", b"issuer", "http_response_body"),
        ("bad.bin", b"Forbidden", "failed_http_response_body"),
        ("good.json", json.dumps({"status_code": 200, "sha256": sha(b"issuer"), "bytes": 6,
                                 "retrieved_at_utc": "2020-01-02T00:00:00Z"}).encode(), "http_receipt_partial"),
        ("bad.json", json.dumps({"status_code": 403, "body_sha256": sha(b"Forbidden"), "body_bytes": 9}).encode(), "failed_http_receipt")]:
        (tmp_path / name).write_bytes(content)
        files.append({"archive_relative_path": name, "sha256": sha(content), "bytes": len(content), "capture_kind": kind})
    result = workflow.audit_source_receipts({"source_archive": {"files": files}, "source_versions": {}}, tmp_path)
    assert result["failed_body_count"] == 1
    assert result["receipts"][0]["request_started_at_utc"] is None
    assert result["receipts"][0]["valid_response_body"] is True
    assert result["receipts"][1]["valid_response_body"] is False
    assert result["dom_exact_response_received_at_utc"] is None
    (tmp_path / "bad.json").write_text(json.dumps({"status_code": 200, "body_sha256": sha(b"Forbidden"), "body_bytes": 9}))
    with pytest.raises(ValueError, match="failed receipt"):
        workflow.audit_source_receipts({"source_archive": {"files": files}, "source_versions": {}}, tmp_path)


def test_completed_guard_precedes_plan_source_and_qlib_reads(tmp_path, monkeypatch):
    import qlib
    (tmp_path / "summary.json").write_text('{"existing":true}')
    monkeypatch.setattr(qlib, "init", lambda **kw: pytest.fail("unexpected qlib.init"))
    with pytest.raises(ValueError, match="completed card"):
        workflow.run_etf_corporate_actions_qualification(tmp_path / "absent.json", output_card=tmp_path,
            source_root=tmp_path, evidence_root=tmp_path, expected_plan_sha256="0" * 64)


def test_active_recorder_guard_precedes_plan_and_init(tmp_path, monkeypatch):
    import qlib
    from qlib.workflow import R
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    qlib.init(provider_uri=str(tmp_path / "unused"), region="us", kernels=1)
    with R.start(experiment_name="synthetic_event_guard", uri=str(tmp_path / "recorder")):
        rid = R.get_recorder().id
        monkeypatch.setattr(qlib, "init", lambda **kw: pytest.fail("active Recorder reset"))
        with pytest.raises(ValueError, match="before qlib.init"):
            workflow.run_etf_corporate_actions_qualification(tmp_path / "missing", output_card=tmp_path,
                source_root=tmp_path, evidence_root=tmp_path, expected_plan_sha256="0" * 64)
        assert R.get_recorder().id == rid


def test_wrong_plan_hash_precedes_init(tmp_path, monkeypatch):
    import qlib
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(workflow, "assert_run_available", lambda card: None)
    monkeypatch.setattr(qlib, "init", lambda **kw: pytest.fail("unexpected qlib.init"))
    with pytest.raises(ValueError, match="plan SHA-256"):
        workflow.run_etf_corporate_actions_qualification(plan, output_card=tmp_path,
            source_root=tmp_path, evidence_root=tmp_path, expected_plan_sha256="0" * 64)


def test_parent_reader_explicitly_pins_both_manifest_contracts(tmp_path, monkeypatch):
    refs, calls = {}, []
    for name, manifest in (("native", "verification_manifest.json"), ("legacy", "artifact_manifest.json")):
        card = tmp_path / name
        card.mkdir()
        (card / "summary.json").write_text("{}")
        refs[name] = {"card": name, "summary_sha256": sha(b"{}"), "recorder_id": name,
            "experiment_id": "1", "recorder_uri": str(card / "qlib_recorder"), "manifest_name": manifest,
            "manifest_sha256": "1" * 64, "artifact_sha256": {"data/one.pkl": "2" * 64, "data/two.pkl": "3" * 64}}
    def load(card, **kwargs):
        calls.append(kwargs)
        return {}, {"summary_sha256": sha(b"{}"), "experiment_id": "1", "recorder_uri": str(card / "qlib_recorder")}
    monkeypatch.setattr(workflow, "load_verified_recorder_objects", load)
    workflow.load_parent_bindings({"parent_references": refs}, tmp_path)
    assert [call["manifest_name"] for call in calls] == ["verification_manifest.json", "artifact_manifest.json"]
    assert all(set(call["artifacts"]) == set(call["expected_artifact_sha256"]) == {"data/one.pkl", "data/two.pkl"} for call in calls)
    assert all(call["expected_manifest_sha256"] == "1" * 64 for call in calls)


@pytest.mark.parametrize("change", [None, "paid", "amount", "historical", "split"])
def test_canonical_outputs_remain_bound_to_frozen_examples(change):
    cash = pd.DataFrame([
        {"instrument": "SOXL", "ex_date": "2020-01-02", "record_date": "2020-01-02",
         "pay_date": "2020-01-06", "original_total_decimal": "0.12000",
         "payable_status_at_market_as_of": "unpaid_receivable_not_spendable"},
        {"instrument": "SOXL", "ex_date": "2010-01-04", "record_date": "2010-01-05",
         "pay_date": "2010-01-06", "original_total_decimal": "0.010",
         "payable_status_at_market_as_of": "scheduled_pay_date_passed_not_actual_account_receipt"},
    ])
    split = {"instrument": "SOXL", "record_date": "2012-01-03", "payable_after_close_date": "2012-01-04",
             "effective_trading_date": "2012-01-05", "new_shares_per_old_share": 2}
    splits = pd.DataFrame([split])
    plan = {"market_as_of": "2020-01-03", "split_events": [split], "time_contract": {"terminal_unpaid_example": {
        "instrument": "SOXL", "ex_date": "2020-01-02", "pay_date": "2020-01-06",
        "amount_per_original_share_decimal": "0.12", "market_as_of": "2020-01-03"}}}
    historical = {"economic_ex_date": "2010-01-04", "record_date": "2010-01-05", "pay_date": "2010-01-06",
                  "original_share_rate_decimal": "0.01"}
    if change == "paid": cash.loc[0, "payable_status_at_market_as_of"] = "paid"
    elif change == "amount": cash.loc[0, "original_total_decimal"] = "0.121"
    elif change == "historical": cash.loc[1, "pay_date"] = "2010-01-07"
    elif change == "split": splits.loc[0, "new_shares_per_old_share"] = 3
    if change:
        with pytest.raises(ValueError): workflow.verify_registered_event_examples(cash, splits, plan, historical)
    else:
        assert all(workflow.verify_registered_event_examples(cash, splits, plan, historical).values())


@pytest.mark.parametrize("failure", [None, "artifact", "context_exit", "status", "readback", "input_changed"])
def test_completion_is_published_only_after_finished_and_all_checks(tmp_path, failure):
    local = tmp_path / "qlib_recorder/1/synthetic"
    artifact_root = local / "artifacts"
    artifact_root.mkdir(parents=True)
    for name in ("plan.json", "workflow.yaml", "qualification.json", "source_differences_summary.json", "conclusion.md"):
        (tmp_path / name).write_text("{}")
    state = {"active": False, "status": "RUNNING"}
    class Recorder:
        id, experiment_id = "synthetic", "1"
        client = SimpleNamespace(get_run=lambda rid: SimpleNamespace(info=SimpleNamespace(status=state["status"], run_id=rid)))
        def get_local_dir(self): return str(local)
        def save_objects(self, artifact_path, **kwargs):
            for name, value in kwargs.items():
                path = artifact_root / artifact_path / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(value))
        def load_object(self, name):
            assert not state["active"]
            value = pickle.loads((artifact_root / name).read_bytes())
            return {} if failure == "readback" else value
        def log_artifact(self, source, artifact_path):
            assert state["active"] and not (tmp_path / "summary.json").exists()
            if failure == "artifact": raise RuntimeError("synthetic artifact failure")
            target = artifact_root / artifact_path / Path(source).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
    @contextmanager
    def context():
        state["active"] = True
        try:
            yield
            if failure == "context_exit": raise RuntimeError("synthetic context failure")
            state["status"] = "FAILED" if failure == "status" else "FINISHED"
        finally:
            state["active"] = False
    def write(recorder, summary):
        recorder.save_objects(artifact_path="data", **{"events.pkl": ["synthetic event"]})
    def verify():
        assert state["status"] == "FINISHED" and not (tmp_path / "summary.json").exists()
        if failure == "input_changed": raise ValueError("changed source")
    def run():
        return workflow.archive_and_publish_completion(tmp_path, {"experiment": "synthetic"},
            recorder_context=context(), recorder_loader=Recorder, write_artifacts=write, verify_bound_inputs=verify)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            run()
        assert not (tmp_path / "summary.json").exists()
    else:
        summary = run()
        assert state["status"] == "FINISHED"
        assert json.loads((tmp_path / "summary.json").read_text()) == summary
        assert (tmp_path / "summary.json").read_bytes() == (artifact_root / "card/summary.json").read_bytes()
        manifest = json.loads((tmp_path / "verification_manifest.json").read_text())
        assert manifest["all_recorded_artifact_hashes_read_back"] is True
        for entry in manifest["entries"]:
            assert sha((tmp_path / entry["path"]).read_bytes()) == entry["sha256"]


def test_notebook_guards_before_explicit_initialization():
    root = Path(__file__).resolve().parents[1]
    card = root / "research/etf_corporate_actions_20261009"
    notebook = json.loads((card / (card.name + ".ipynb")).read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert code.index("assert_run_available(CARD)") < code.index("qlib.init(")
    assert sha((card / "plan.json").read_bytes()) in code
    assert "expected_plan_sha256=EXPECTED_PLAN_SHA256" in code
