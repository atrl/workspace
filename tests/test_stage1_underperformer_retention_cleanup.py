from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_underperformer_retention_cleanup_v1"


def test_stage1_underperformer_cleanup_card_contract():
    required = {
        "stage1_underperformer_retention_cleanup_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "cleanup_decisions.csv",
        "delete_manifest.csv",
        "retained_ideas.md",
    }
    assert required.issubset(
        {path.name for path in CARD.iterdir() if path.is_file()}
    )

    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (
            CARD
            / "stage1_underperformer_retention_cleanup_v1.ipynb"
        ).read_text(encoding="utf-8")
    )
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert summary["replacement_promoted"] is False
    assert summary["selection_uses_outer_test"] is False
    assert summary["deleted_runnable_chains"] == 3
    assert summary["artifact_policy"] == {
        "policy": "slim_card_recorder_first",
        "workflow_config": "workflow.yaml",
        "recorder_only_artifacts": [],
    }
    assert summary["stage1_metrics_10bps"]["return_pct"] > 500.0
    assert "import qlib" in notebook_text
    assert "qlib.init" in notebook_text
    assert "D.calendar" in notebook_text
    assert "from qlib.workflow import R" in notebook_text
    assert "中文结论" in notebook_text


def test_cleanup_manifest_protects_stage1_and_removes_only_rejected_chains():
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    decisions = pd.read_csv(CARD / "cleanup_decisions.csv")
    manifest = pd.read_csv(CARD / "delete_manifest.csv")

    for relative_path in summary["protected_paths"]:
        assert (ROOT / relative_path).exists(), relative_path
    for relative_path in manifest["path"]:
        assert not (ROOT / relative_path).exists(), relative_path

    assert set(decisions["decision"]) == {
        "protected_current_baseline",
        "delete_runnable_keep_experience",
    }
    assert set(decisions.loc[decisions["decision"].str.startswith("delete"), "implementation"]) == {
        "cost_aware_target_tracking_v1",
        "stateful_direct_allocation_v1",
        "stage1_feature_recovery_ablation_v2",
    }
