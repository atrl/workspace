from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_generalization_validation_v1"


def test_circular_block_bootstrap_is_deterministic_and_path_aware():
    from src.workflow.stage1_generalization_validation import (
        circular_block_bootstrap,
    )

    active = np.linspace(-0.01, 0.012, 100)
    first = circular_block_bootstrap(
        active,
        block_length=20,
        samples=100,
        seed=7,
    )
    second = circular_block_bootstrap(
        active,
        block_length=20,
        samples=100,
        seed=7,
    )

    assert np.array_equal(first, second)
    assert len(np.unique(first)) > 10
    assert np.all(first > 0.0)


def test_stage1_generalization_card_contract():
    required = {
        "stage1_generalization_validation_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "annual_metrics.csv",
        "cost_stress.csv",
        "rolling_window_metrics.csv",
        "rolling_summary.csv",
        "regime_metrics.csv",
        "parameter_stress.csv",
        "bootstrap_summary.csv",
        "cross_asset_proxy_metrics.csv",
        "matched_exposure_metrics.csv",
    }

    assert required.issubset({path.name for path in CARD.iterdir() if path.is_file()})
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    notebook = json.loads(
        (CARD / "stage1_generalization_validation_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert summary["selection_uses_outer_test"] is False
    assert summary["source_training_end"] == "2022-12-30"
    assert summary["clean_unseen_holdout_available"] is False
    assert summary["real_money_promotion"] == "blocked"
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["generalization_gates"]["annual_risk_pass_count"] == 4
    assert "import qlib" in notebook_text
    assert "qlib.init" in notebook_text
    assert "D.calendar" in notebook_text
    assert "from qlib.workflow import R" in notebook_text
    assert "中文结论" in notebook_text
