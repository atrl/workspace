import json

from src.core import research_runtime as runtime


def test_completed_receipt_overrides_stale_pending_state():
    projection = runtime.project_task(
        [{"phase": "P1", "status": "completed"}, {"phase": "old_attempt", "status": "closed_failed"}],
        {"goal_state": "active", "status": "running", "current_phase": "P1",
         "phases": {"P1": "pending"}, "scientific_state": "not_promoted"},
    )
    assert projection.phases["P1"] == "completed"
    assert projection.current_phase is None
    assert any("stale run_state" in issue for issue in projection.issues)
    assert (projection.goal_state, projection.job_state, projection.scientific_state) == (
        "active", "unverified_running", "not_promoted")
    assert "process identity was not verified" in projection.issues[0]


def test_job_observation_detects_missing_and_pid_reuse(monkeypatch):
    monkeypatch.setattr(runtime, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(runtime, "process_start_identity", lambda _pid: "start-new")
    reused = runtime.observe_job(pid=123, start_identity="start-old")
    assert reused["status"] == "pid_reused"
    monkeypatch.setattr(runtime, "_pid_alive", lambda _pid: False)
    assert runtime.observe_job(pid=123, start_identity="start-old")["status"] == "missing"


def test_exit_receipt_requires_matching_pid_and_start_identity(monkeypatch):
    monkeypatch.setattr(runtime, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(runtime, "process_start_identity", lambda _pid: "start-current")
    mismatched = runtime.observe_job(pid=123, start_identity="start-current",
        exit_receipt={"pid": 456, "start_identity": "start-current", "exit_code": 0})
    assert mismatched["status"] == "running"
    assert mismatched["exit_receipt_status"] == "untrusted_mismatch"
    unbound = runtime.observe_job(pid=123, exit_receipt={"pid": 123,
        "start_identity": "start-current", "exit_code": 0})
    assert unbound["status"] == "running_unverified_identity"
    assert unbound["exit_receipt_status"] == "untrusted_missing_expected_identity"
    monkeypatch.setattr(runtime, "_pid_alive", lambda _pid: False)
    matched = runtime.observe_job(pid=123, start_identity="start-current",
        exit_receipt={"pid": 123, "start_identity": "start-current", "exit_code": 0})
    assert matched["status"] == "completed"
    assert matched["exit_receipt_status"] == "matched"


def test_usage_deduplicates_response_ids_and_uses_response_usage(tmp_path):
    rows = [
        {"type": "token_usage_record", "payload": {"response_id": "r1", "turn_id": "t",
         "usage": {"input_tokens": 100, "cached_input_tokens": 40,
                   "output_tokens": 10, "reasoning_output_tokens": 3}}},
        {"type": "token_usage_record", "payload": {"response_id": "r1", "turn_id": "t",
         "usage": {"input_tokens": 100, "output_tokens": 10}}},
        {"type": "token_usage_record", "payload": {"response_id": "r2", "turn_id": "t",
         "turn_token_usage": {"input_tokens": 150, "output_tokens": 15, "total_tokens": 165}}},
        {"type": "token_usage_record", "payload": {"response_id": "r3", "turn_id": "t",
         "usage": {"input_tokens": 30, "output_tokens": 6, "total_tokens": 36}}},
        {"type": "token_usage_record", "payload": {"response_id": "r4", "turn_id": "t",
         "turn_token_usage": {"input_tokens": 180, "output_tokens": 21, "total_tokens": 201}}},
    ]
    path = tmp_path / "usage.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = runtime.aggregate_usage(path)
    assert result["totals"]["input_tokens"] == 180
    assert result["totals"]["cached_input_tokens"] == 40
    assert result["totals"]["output_tokens"] == 21
    assert result["totals"]["reasoning_output_tokens"] == 3
    assert result["totals"]["total_tokens"] == 201
    assert result["responses"] == 4
    assert result["billing"] == "not_inferred"
    assert result["aggregation_mode"] == "full_scan"


def test_campaign_view_reads_phase_receipts_without_claiming_process_liveness(tmp_path):
    from argparse import Namespace
    from src.core.research_monitor import snapshot

    research = tmp_path / "research"
    card = research / "card"
    card.mkdir(parents=True)
    phases = card / "phases.json"
    phases.write_text(json.dumps({"receipts": [
        {"phase": "P1", "status": "completed"}, {"phase": "P2", "status": "implementation"}]}))
    campaign = research / "campaign.json"
    campaign.write_text(json.dumps({
        "current_phase": "stale_P1_pending",
        "goal_controller": {"observed_status": "blocked"},
        "current_task": {"phase_receipts": "research/card/phases.json",
                         "run_state": "research/card/run.json",
                         "job_handle": "research/card/job.json",
                         "execution_receipt": "research/card/exit.json",
                         "last_scientific_observation": "source_only"},
    }))
    args = Namespace(campaign=campaign, state=None, receipts=None, pid=None,
                     start_identity=None, exit_receipt=None, heartbeat=None,
                     stale_after=300, usage=None)
    before = snapshot(args)
    assert before["task"]["current_phase"] == "P2"
    assert before["job"]["status"] == "unknown"
    (card / "run.json").write_text(json.dumps({"phase": "P2", "status": "running"}))
    during = snapshot(args)
    assert during["task"]["job_state"] == "unverified_running"
    assert during["job"]["status"] == "unknown"
    assert args.state is None  # Watch iterations resolve fresh references.
    phases.write_text(json.dumps({"receipts": [
        {"phase": "P1", "status": "completed"}, {"phase": "P2", "status": "completed"}]}))
    assert snapshot(args)["task"]["current_phase"] is None
