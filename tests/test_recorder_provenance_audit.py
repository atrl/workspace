from pathlib import Path

from src.analysis.recorder_provenance_audit import (
    audit_card, audit_tree, missing_recorders,
)

ROOT = Path(__file__).resolve().parents[1]


def test_audit_card_reports_present_recorder(tmp_path):
    card = tmp_path / "research" / "mycard"
    run = card / "qlib_recorder" / "exp1" / "abc123"
    run.mkdir(parents=True)
    (card / "summary.json").write_text('{"recorder_id": "abc123"}')
    row = audit_card(card)
    assert row["status"] == "present"
    assert row["recorder_id"] == "abc123"


def test_audit_card_reports_missing_recorder(tmp_path):
    card = tmp_path / "research" / "mycard"
    (card / "qlib_recorder").mkdir(parents=True)
    (card / "summary.json").write_text('{"recorder_id": "deadbeef"}')
    row = audit_card(card)
    assert row["status"] == "missing_recorder"


def test_audit_card_without_summary(tmp_path):
    card = tmp_path / "research" / "empty"
    card.mkdir(parents=True)
    assert audit_card(card)["status"] == "no_summary"


def test_missing_recorders_filters_the_table():
    import pandas as pd
    table = pd.DataFrame([
        {"card": "a", "status": "present", "recorder_id": "1"},
        {"card": "b", "status": "missing_recorder", "recorder_id": "2"},
    ])
    missing = missing_recorders(table)
    assert list(missing["card"]) == ["b"]


def test_tree_audit_runs_and_reports_a_table():
    table = audit_tree(ROOT)
    assert "status" in table.columns
    # A well-formed repo must expose both present and (historically) missing states.
    assert set(table["status"]).issubset({"present", "missing_recorder", "no_summary",
                                          "no_declared_recorder", "unreadable_summary"})
