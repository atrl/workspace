from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_controller_aware_distillation_v1"


def test_controller_aware_distillation_card_contract():
    required = {
        "stage1_controller_aware_distillation_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "model_selection.csv",
        "backtest_benchmark_comparison.csv",
        "annual_metrics.csv",
        "risk_analysis.csv",
        "analysis_summary.json",
        "failure_modes.md",
    }
    assert required <= {path.name for path in CARD.iterdir()}
    assert not {
        "predictions.csv",
        "portfolio_report.csv",
        "positions.csv",
        "all_model_predictions.csv",
    } & {path.name for path in CARD.iterdir()}


def test_controller_aware_distillation_is_development_selected_and_safe():
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["selection_uses_outer_test"] is False
    assert summary["source_stage1"]["frozen"] is True
    assert summary["source_stage1"]["stage2_enabled"] is False
    assert summary["method"]["hard_constraints"]["maximum_soxl_weight"] == 0.30
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["promotion"] == "blocked"
    assert summary["promotion_gates"]["outer_used_for_selection"] is False
    assert summary["dataset_audit"]["dataset_class"].endswith("DatasetH")
    assert (
        summary["dataset_audit"]["segments"]["all"][1]
        <= "2022-12-31"
    )
    assert (
        summary["teacher_audit"]["rows"]
        == summary["dataset_audit"]["segment_rows"]["train_outer"]
    )
    assert "qlib.workflow.R" in summary["qlib_api"]
    recorder_dir = Path(summary["recorder_local_dir"])
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        assert not (recorder_dir / "artifacts" / filename).exists()
    selected = summary["development_selected_metrics_10bps"]
    assert selected["max_drawdown_pct"] >= -30.0
    assert selected["worst_rolling_5d_pct"] >= -12.0


def test_controller_aware_distillation_comparison_keeps_baseline():
    table = pd.read_csv(CARD / "model_selection.csv")
    assert {"stage1_baseline", "ridge_a100_s025_u015"} <= set(
        table["variant"]
    )
    assert not table["selection_uses_outer_test"].astype(bool).any()
    development = table[
        table["segment"].eq("development_walk_forward")
    ]
    assert development["variant"].nunique() >= 6
    assert development.loc[
        development["variant"].eq("stage1_baseline"), "selected"
    ].eq(False).all()


def test_controller_aware_distillation_notebook_is_qlib_native():
    notebook = json.loads(
        (
            CARD
            / "stage1_controller_aware_distillation_v1.ipynb"
        ).read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "import qlib" in text
    assert "qlib.init" in text
    assert "qlib.data import D" in text
    assert "qlib.workflow import R" in text
    assert "run_stage1_controller_aware_distillation_card" in text
    assert "conclusion.md" in text
