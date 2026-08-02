from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

import src.workflow as workflow_exports
from src.workflow.macro_state_factor_ablation import (
    BASE_GROUP,
    EXPERIMENT,
    MacroStateFactorAblationSpec,
    _macro_feature_groups,
    _select_feature_group,
    _write_notebook,
)


def test_macro_feature_groups_are_true_base_ablation_groups():
    frame = pd.DataFrame(
        {
            "alpha158_MA5": [1.0],
            "macro_dfii10_chg5": [0.1],
            "macro_dtwexbgs_ret5": [0.2],
            "macro_dgs2_chg5": [0.3],
        }
    )
    groups = _macro_feature_groups({BASE_GROUP: ["alpha158_MA5"]}, frame=frame)
    assert groups[BASE_GROUP] == ["alpha158_MA5"]
    assert groups["base_plus_real_rates"] == ["alpha158_MA5", "macro_dfii10_chg5"]
    assert groups["base_plus_broad_dollar"] == ["alpha158_MA5", "macro_dtwexbgs_ret5"]
    assert groups["base_plus_rate_vol"] == ["alpha158_MA5", "macro_dgs2_chg5"]


def test_macro_group_selection_is_validation_only_and_gate_aware():
    ablation = pd.DataFrame(
        [
            {
                "feature_group": BASE_GROUP,
                "validation_selection_objective_score": 1.0,
                "validation_return_pct": 2.0,
                "validation_risk_gate_pass": True,
                "test_return_pct": 100.0,
            },
            {
                "feature_group": "base_plus_real_rates",
                "validation_selection_objective_score": 2.0,
                "validation_return_pct": 3.0,
                "validation_risk_gate_pass": True,
                "test_return_pct": -100.0,
            },
            {
                "feature_group": "base_plus_rate_vol",
                "validation_selection_objective_score": 9.0,
                "validation_return_pct": 9.0,
                "validation_risk_gate_pass": False,
                "test_return_pct": 999.0,
            },
        ]
    )
    assert _select_feature_group(ablation) == "base_plus_real_rates"


def test_macro_workflow_export_and_notebook_contract(tmp_path: Path):
    assert workflow_exports.MacroStateFactorAblationSpec is MacroStateFactorAblationSpec
    summary = {
        "experiment": EXPERIMENT,
        "model_selection": {"selected_feature_group": BASE_GROUP},
        "macro_ablation": [
            {
                "feature_group": BASE_GROUP,
                "validation_selection_objective_score": 1.0,
                "test_return_pct": 2.0,
                "test_max_drawdown_pct": -1.0,
                "walk_forward_risk_all_pass": True,
                "walk_forward_return_all_pass": True,
                "walk_forward_min_return_pct": 0.5,
            }
        ],
        "promotion_decision": "not_promoted_validation_selected_baseline",
        "recorder_id": "recorder-test",
    }
    _write_notebook(tmp_path, summary)
    text = (tmp_path / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8")
    assert "from qlib.data.dataset import DatasetH" in text
    assert "run_macro_state_factor_ablation_card" in text
    assert "while not (ROOT / 'src').exists()" in text


def test_generated_macro_state_card_is_slim_validation_selected_and_recorder_backed():
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["model_selection"]["selection_uses_outer_test"] is False
    assert summary["model_selection"]["selected_feature_group"] == BASE_GROUP
    assert {row["feature_group"] for row in summary["macro_ablation"]} == {
        BASE_GROUP,
        "base_plus_real_rates",
        "base_plus_broad_dollar",
        "base_plus_rate_vol",
        "base_plus_all_macro_state",
    }
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["workflow_config"] == "workflow.yaml"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "positions": "artifacts/macro_state_factor_ablation/positions.csv",
        "walk_forward_positions": "artifacts/macro_state_factor_ablation/walk_forward_positions.csv",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    assert not (card / "positions.csv").exists()
    assert not (card / "walk_forward_positions.csv").exists()
    recorder_dir = root / summary["recorder_local_dir"]
    assert (recorder_dir / "artifacts" / "macro_state_factor_ablation" / "positions.csv").exists()
    assert (recorder_dir / "artifacts" / "macro_state_factor_ablation" / "walk_forward_positions.csv").exists()
