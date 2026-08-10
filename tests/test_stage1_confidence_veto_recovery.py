from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.workflow.stage1_confidence_veto_recovery import (
    select_confidence_veto_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_confidence_veto_recovery_v1"


def test_selection_uses_development_and_prefers_action_year_stability():
    common = {
        "all_primary_gates_pass": True,
        "threshold_plateau_count": 3,
        "required_threshold_plateau_count": 3,
        "recovery_action_count": 20,
    }
    candidates = pd.DataFrame(
        [
            {
                **common,
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "candidate": "stable",
                "action_year_min_wealth_ratio": 0.99,
                "return_pct": 420.0,
            },
            {
                **common,
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "candidate": "higher_return_less_stable",
                "action_year_min_wealth_ratio": 0.98,
                "return_pct": 430.0,
            },
            {
                **common,
                "segment": "outer_reused_diagnostic",
                "selection_uses_outer_test": True,
                "candidate": "outer_leak",
                "action_year_min_wealth_ratio": 2.0,
                "return_pct": 900.0,
            },
        ]
    )

    selected = select_confidence_veto_candidate(candidates)

    assert selected["candidate"] == "stable"


def test_generated_card_declares_recorder_first_and_outer_is_diagnostic():
    if not (CARD / "summary.json").exists():
        return
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))

    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_role"] == "reused_diagnostic_only_not_model_selection"
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["frequency_definition"]["daily_09_31_rebalance"] == "unchanged"


def test_notebook_contains_executable_qlib_and_chinese_conclusion_cells():
    notebook = json.loads(
        (CARD / "stage1_confidence_veto_recovery_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "qlib.init" in source
    assert "D.calendar" in source
    assert "DatasetH" in source
    assert "run_stage1_confidence_veto_recovery_card" in source
    assert "中文结论" in source
