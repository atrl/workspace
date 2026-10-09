import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import stage1_history_audit as audit
from src.workflow import stage1_history_audit as workflow

ROOT = Path(__file__).resolve().parents[1]


def _card(root, name, *, recorder_id, decision, outer_status=None, selection_uses_outer=False,
          with_recorder=False):
    card = root / "research" / name
    card.mkdir(parents=True)
    (card / "summary.json").write_text(json.dumps({
        "recorder_id": recorder_id, "decision": decision,
        "outer_status": outer_status, "selection_uses_outer_test": selection_uses_outer,
    }))
    (card / "qlib_recorder" / "exp" / (recorder_id if with_recorder else "other")).mkdir(parents=True)
    return card


def test_lineage_flags_missing_recorder_and_non_fresh_outer(tmp_path):
    _card(tmp_path, "a", recorder_id="r1", decision="promoted_x", with_recorder=False)
    _card(tmp_path, "b", recorder_id="r2", decision="promoted_y",
          outer_status="reused_diagnostic_not_fresh_unseen_test", with_recorder=True)
    # A separate, present production card keeps the lineage distinct from 'a'.
    _card(tmp_path, "prod", recorder_id="r3", decision="production", with_recorder=True)
    table = audit.audit_lineage(tmp_path, lineage=("a", "b"), production="prod")
    findings = audit.lineage_findings(table)
    assert findings["recorders_missing"] >= 1
    assert "a" in findings["missing_recorder_cards"]
    assert "b" in findings["outer_not_fresh_cards"]
    assert findings["all_promotions_have_recorder"] is False


def test_walk_forward_verdict_requires_everything(tmp_path):
    # A clean lineage with plenty of forward sessions passes.
    _card(tmp_path, "a", recorder_id="r1", decision="promoted_x", with_recorder=True)
    table = audit.audit_lineage(tmp_path, lineage=("a",), production="a")
    findings = audit.lineage_findings(table)
    ok = audit.walk_forward_verdict(findings, forward_sessions=60, required_forward_sessions=60)
    assert ok["passed_fresh_walk_forward"] is True
    # Too few forward sessions fails.
    short = audit.walk_forward_verdict(findings, forward_sessions=13, required_forward_sessions=60)
    assert short["passed_fresh_walk_forward"] is False
    assert any("forward sessions" in reason for reason in short["failed_reasons"])


def test_walk_forward_verdict_fails_on_outer_selection(tmp_path):
    _card(tmp_path, "a", recorder_id="r1", decision="promoted_x",
          selection_uses_outer=True, with_recorder=True)
    _card(tmp_path, "prod", recorder_id="r2", decision="production", with_recorder=True)
    table = audit.audit_lineage(tmp_path, lineage=("a",), production="prod")
    findings = audit.lineage_findings(table)
    verdict = audit.walk_forward_verdict(findings, forward_sessions=60, required_forward_sessions=60)
    assert verdict["passed_fresh_walk_forward"] is False
    assert any("outer window" in reason for reason in verdict["failed_reasons"])


def test_real_repo_lineage_has_not_passed_walk_forward():
    summary = workflow.run_stage1_history_audit(root=ROOT)
    assert summary["decision"] == "no_fresh_walk_forward_evidence"
    assert summary["walk_forward_verdict"]["passed_fresh_walk_forward"] is False
    # The production lineage is known to have missing recorders and reused outer data.
    assert summary["findings"]["recorders_missing"] >= 1
    assert summary["findings"]["all_promotions_have_recorder"] is False
    assert summary["forward_sessions"] < summary["required_forward_sessions"]


def test_managed_forward_sessions_counts_distinct_sessions(tmp_path):
    state = tmp_path / workflow.STAGE1_RUNTIME
    state.mkdir(parents=True)
    (state / "execution_state_v2.json").write_text(json.dumps({
        "completed_slots": {
            "2026-09-28:open": {"execution_session_date": "2026-09-28", "status": "filled_confirmed"},
            "2026-09-28:recovery": {"execution_session_date": "2026-09-28", "status": "no_action"},
            "2026-09-29:open": {"execution_session_date": "2026-09-29", "status": "hold_completed"},
        }
    }))
    assert workflow.managed_forward_sessions(tmp_path) == 2
