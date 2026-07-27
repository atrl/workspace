from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = (
    ROOT
    / "research"
    / "stage1_cash_funded_risk_budget_overlay_v1"
)


def test_cash_funded_overlay_card_contract():
    required = {
        "stage1_cash_funded_risk_budget_overlay_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "model_selection.csv",
        "backtest_benchmark_comparison.csv",
        "annual_metrics.csv",
        "risk_analysis.csv",
        "cash_availability.csv",
        "factor_diagnostics.csv",
        "momentum_failure.csv",
        "threshold_schedule.csv",
        "analysis_summary.json",
        "failure_modes.md",
    }
    assert required <= {path.name for path in CARD.iterdir()}
    assert not {
        "predictions.csv",
        "portfolio_report.csv",
        "positions.csv",
        "factor_label_frame.csv",
        "model_selection_full.csv",
    } & {path.name for path in CARD.iterdir()}


def test_cash_funded_overlay_is_frozen_safe_and_blocked():
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["selection_uses_outer_test"] is False
    assert summary["source_stage1"]["frozen"] is True
    assert summary["source_stage1"]["stage2_enabled"] is False
    assert summary["candidate_count"] == 6
    assert summary["method"]["capital_source"] == "cash_only"
    constraints = summary["method"]["hard_constraints"]
    assert constraints["minimum_cash_weight"] == 0.15
    assert constraints["source_maximum_soxl_weight"] == 0.30
    assert constraints["overlay_maximum_soxl_weight"] == 0.40
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    assert summary["promotion"] == "blocked"
    assert summary["selected_model"]["name"] == "soxl_010_pullback"
    assert summary["selection_gates"]["return_gate_pass"] is False
    assert (
        summary["outer_diagnostic_selected_metrics_10bps"][
            "return_delta_pct_points"
        ]
        < 0.0
    )
    assert summary["threshold_audit"]["outer_unique_threshold_rows"] == 1
    assert summary["dataset_audit"]["dataset_class"].endswith("DatasetH")
    assert "qlib.workflow.R" in summary["qlib_api"]
    recorder_dir = Path(summary["recorder_local_dir"])
    artifact_dir = (
        recorder_dir
        / "artifacts"
        / "stage1_cash_funded_risk_budget_overlay_v1"
    )
    assert (artifact_dir / "portfolio_report.csv").exists()
    assert (artifact_dir / "factor_label_frame.csv").exists()
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        assert not (recorder_dir / "artifacts" / filename).exists()


def test_cash_and_momentum_diagnostics_preserve_failure_boundary():
    cash = pd.read_csv(CARD / "cash_availability.csv")
    development = cash[
        cash["segment"].eq("development_walk_forward")
    ].iloc[0]
    outer = cash[
        cash["segment"].eq("outer_reused_diagnostic")
    ].iloc[0]
    assert development["rows"] == 69
    assert outer["rows"] == 176
    assert development["mean_cash_weight"] > 0.19
    assert outer["mean_cash_weight"] > 0.20

    factors = pd.read_csv(CARD / "factor_diagnostics.csv")
    relative = factors[
        factors["factor"].eq("relative_momentum20")
        & factors["segment"].eq("development_walk_forward")
    ].iloc[0]
    assert relative["spearman_vs_soxl_minus_soxx"] < 0.0
    assert (
        relative["high_third_mean_spread"]
        < relative["low_third_mean_spread"]
    )


def test_cash_funded_overlay_notebook_is_qlib_native():
    notebook = json.loads(
        (
            CARD
            / "stage1_cash_funded_risk_budget_overlay_v1.ipynb"
        ).read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "import qlib" in text
    assert "qlib.init" in text
    assert "qlib.data import D" in text
    assert "qlib.workflow import R" in text
    assert "run_stage1_cash_funded_risk_budget_overlay_card" in text
    assert "conclusion.md" in text
