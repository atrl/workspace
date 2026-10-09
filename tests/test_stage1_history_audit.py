import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import stage1_history_audit as audit
from src.workflow import stage1_history_audit as workflow

ROOT = Path(__file__).resolve().parents[1]


def _card(root, name, *, recorder_id, decision, outer_status="fresh_unseen", selection_uses_outer=False,
          with_recorder=True, recorder_dir_empty=False):
    card = root / "research" / name
    card.mkdir(parents=True)
    (card / "summary.json").write_text(json.dumps({
        "recorder_id": recorder_id, "decision": decision,
        "outer_status": outer_status, "selection_uses_outer_test": selection_uses_outer,
    }))
    run = card / "qlib_recorder" / "exp" / recorder_id
    if with_recorder:
        run.mkdir(parents=True)
        if not recorder_dir_empty:
            (run / "artifacts").mkdir()
            (run / "summary.json").write_text("{}")
    return card


# ---------------------------------------------------------------------------
# A. Fail-closed lineage audit
# ---------------------------------------------------------------------------

def test_missing_recorder_never_grants_success(tmp_path):
    _card(tmp_path, "a", recorder_id="r1", decision="promoted_x", with_recorder=False)
    table = audit.audit_lineage(tmp_path, lineage=("a",), production="prod")
    findings = audit.lineage_findings(table)
    assert findings["all_promotions_have_recorder"] is False
    assert "a" in findings["missing_recorder_cards"]


def test_empty_recorder_directory_is_not_presence(tmp_path):
    _card(tmp_path, "a", recorder_id="r1", decision="promoted_x",
          with_recorder=True, recorder_dir_empty=True)
    table = audit.audit_lineage(tmp_path, lineage=("a",), production="prod")
    row = table.iloc[0]
    assert row["recorder_state"] == "empty_dir"
    assert bool(row["recorder_present"]) is False
    assert row["status"] == "empty_recorder"


def test_missing_card_is_flagged(tmp_path):
    # No cards at all: every entry is missing_card, never "clean".
    table = audit.audit_lineage(tmp_path, lineage=("a", "b"), production="prod")
    findings = audit.lineage_findings(table)
    assert findings["missing_card_cards"] == ["a", "b", "prod"]
    assert findings["all_cards_ok"] is False


def test_missing_fields_and_unknown_freshness_fail_closed(tmp_path):
    card = tmp_path / "research" / "a"
    card.mkdir(parents=True)
    # decision present but outer_status absent -> missing_field, freshness unknown.
    (card / "summary.json").write_text(json.dumps({"recorder_id": "r1", "decision": "promoted_x"}))
    table = audit.audit_lineage(tmp_path, lineage=("a",), production="prod")
    row = table.iloc[0]
    assert row["status"] == "missing_field"
    assert row["outer_is_fresh"] is None or pd.isna(row["outer_is_fresh"])
    findings = audit.lineage_findings(table)
    assert "a" in findings["outer_unknown_cards"]
    # Unknown freshness is treated as "used outer".
    assert findings["any_selection_used_outer"] is True


def test_empty_promotion_set_fails(tmp_path):
    _card(tmp_path, "a", recorder_id="r1", decision="diagnostic", with_recorder=True)
    _card(tmp_path, "prod", recorder_id="r2", decision="production", with_recorder=True)
    table = audit.audit_lineage(tmp_path, lineage=("a",), production="prod")
    findings = audit.lineage_findings(table)
    assert findings["empty_promotion_set"] is True
    assert findings["all_promotions_have_recorder"] is False


# ---------------------------------------------------------------------------
# A. Walk-forward verdict has NO success branch
# ---------------------------------------------------------------------------

def test_walk_forward_verdict_has_no_success_branch_any_input():
    perfect_findings = {
        "all_cards_ok": True, "empty_promotion_set": False,
        "any_selection_used_outer": False, "all_promotions_have_recorder": True,
    }
    perfect_evidence = {"reasons": [], "eligible_sessions": 9999, "raw_observed_count": 9999,
                        "settled_count": 9999, "required_sessions": 60, "status": "eligible",
                        "qualification": "needs_data"}
    verdict = audit.walk_forward_verdict(perfect_findings, forward_evidence=perfect_evidence)
    assert verdict["passed_fresh_walk_forward"] is False
    assert verdict["qualification"] == "needs_data"


def test_walk_forward_verdict_rejects_non_mapping_evidence():
    with pytest.raises(ValueError, match="diagnostic mapping"):
        audit.walk_forward_verdict({"all_cards_ok": True}, forward_evidence=60)


def test_forward_evidence_report_is_always_needs_data(tmp_path):
    # Even with a fully populated runtime, the diagnostic never claims eligibility.
    plan = {"minimum_complete_paired_sessions": 60, "minimum_sessions_per_regime": 15,
            "minimum_regime_transitions": 3}
    (tmp_path / "runtime/soxl/stage1_futu_sim").mkdir(parents=True)
    report = audit.forward_evidence_report(root=tmp_path, forward_plan=plan,
                                           contamination={"windows": []})
    assert report["qualification"] == "needs_data"
    assert report["eligible_sessions"] is None
    assert report["status"] == "needs_data"


def test_forward_evidence_report_rejects_non_plan():
    with pytest.raises(ValueError, match="forward plan"):
        audit.forward_evidence_report(root=Path("."), forward_plan="notadict",
                                      contamination={"windows": []})


# ---------------------------------------------------------------------------
# Real-repo assertions
# ---------------------------------------------------------------------------

def test_real_repo_active_release_reports_development_risk_failures():
    risk = audit.active_release_development_risk(ROOT)
    assert risk["status"] == "ok"
    assert risk["release_id"] == "stage1_hold_net_fees_20260928_paper_trial_v1"
    assert "max_drawdown_not_worse" in risk["failed_development_gates"]
    assert "cvar5_not_worse" in risk["failed_development_gates"]
    assert risk["development_gate_pass"] is False


def test_real_repo_forward_plan_comes_from_sealed_manifest():
    plan = audit.load_forward_plan(ROOT)
    assert plan["minimum_complete_paired_sessions"] == 60
    sealed = json.loads((ROOT / "production/stage1_regime_paper_20261001/release_manifest.json").read_text())
    assert plan == sealed["forward_plan"]


def test_workflow_writes_to_card_dir_and_never_passes(tmp_path):
    summary = workflow.run_stage1_history_audit(root=ROOT, card_dir=tmp_path)
    assert summary["decision"] == "no_fresh_walk_forward_evidence"
    assert summary["walk_forward_verdict"]["passed_fresh_walk_forward"] is False
    assert summary["forward_evidence"]["qualification"] == "needs_data"
    assert summary["active_release_development_risk"]["development_gate_pass"] is False
    # The workflow must not have written into the real repo card.
    assert (tmp_path / "summary.json").is_file()


def test_null_and_garbage_outer_status_are_unknown_not_fresh(tmp_path):
    for name, status in (("nullcard", None), ("garbagecard", "totally-unknown")):
        card = tmp_path / "research" / name
        card.mkdir(parents=True)
        (card / "summary.json").write_text(json.dumps({
            "recorder_id": f"r-{name}", "decision": "promoted_x",
            "outer_status": status, "selection_uses_outer_test": False,
        }))
        (card / "qlib_recorder" / "exp" / f"r-{name}" / "artifacts").mkdir(parents=True)
    table = audit.audit_lineage(tmp_path, lineage=("nullcard", "garbagecard"), production="prod")
    rows = {r["card"]: r for _, r in table.iterrows()}
    for name in ("nullcard", "garbagecard"):
        row = rows[name]
        assert row["outer_is_fresh"] is None or pd.isna(row["outer_is_fresh"])
        assert row["status"] == "unknown_freshness"
    findings = audit.lineage_findings(table)
    assert set(findings["unknown_freshness_cards"]) == {"nullcard", "garbagecard"}
    # Unknown freshness is treated as "used outer" and fails closed.
    assert findings["any_selection_used_outer"] is True


def test_explicit_known_fresh_status_is_accepted(tmp_path):
    _card(tmp_path, "a", recorder_id="r1", decision="promoted_x", outer_status="fresh_unseen_holdout")
    _card(tmp_path, "prod", recorder_id="r2", decision="production", outer_status="fresh_unseen_holdout")
    table = audit.audit_lineage(tmp_path, lineage=("a",), production="prod")
    row = table[table["card"] == "a"].iloc[0]
    assert bool(row["outer_is_fresh"]) is True
    assert row["status"] == "ok"


def test_forward_plan_is_labeled_declared_not_verified(tmp_path):
    summary = workflow.run_stage1_history_audit(root=ROOT, card_dir=tmp_path)
    assert summary["forward_plan_status"]["declared_plan"] is True
    assert summary["forward_plan_status"]["verified"] is False


def test_conclusion_risk_line_follows_verified_status(tmp_path):
    # When the verified loader cannot read the release, the conclusion must say
    # it cannot be verified, not assert a failure.
    summary = workflow.run_stage1_history_audit(root=ROOT, card_dir=tmp_path)
    if summary["active_release_development_risk"]["status"] != "ok":
        text = (tmp_path / "conclusion.md").read_text()
        assert "无法核验" in text
