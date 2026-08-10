from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "holdings_aware_pit_allocation_v1"


def _summary() -> dict:
    return json.loads((CARD / "summary.json").read_text(encoding="utf-8"))


def test_holdings_aware_card_is_slim_and_recorder_first():
    required = {
        "holdings_aware_pit_allocation_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "model_selection.csv",
        "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics.csv",
        "risk_analysis.csv",
        "training_summary.csv",
        "objective_gate_audit.csv",
        "stage1_integration_diagnostic.csv",
    }
    assert required <= {path.name for path in CARD.iterdir()}
    assert not {
        "predictions.csv",
        "positions.csv",
        "strategy_returns.csv",
    } & {path.name for path in CARD.iterdir()}
    assert _summary()["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )


def test_holdings_aware_training_fixes_actual_pretrade_accounting():
    summary = _summary()

    assert summary["architecture"]["turnover_accounting"] == (
        "target_vs_marked_to_market_actual_pretrade_holdings; "
        "SOXL_and_SOXX_order_notionals_charged_separately"
    )
    assert summary["frequency_objective"]["selection_uses_outer_test"] is False
    assert summary["development"]["selected"][
        "trade_day_reduction_vs_primary"
    ] >= 0.35


def test_holdings_aware_candidate_is_rejected_after_stage1_integration():
    summary = _summary()
    integration = pd.read_csv(CARD / "stage1_integration_diagnostic.csv")
    development = integration[
        integration["segment"].eq("development_walk_forward")
    ].iloc[0]

    assert summary["promotion_decision"] == (
        "not_promoted_pit_auxiliary_training_gates_failed"
    )
    assert summary["promotion"] == "blocked_pending_clean_forward"
    assert summary["stage1_integration_diagnostic"]["gate_pass"] is False
    assert summary["stage1_integration_diagnostic"]["deployment"] == (
        "rejected_not_wired_to_simulator"
    )
    assert development["trade_day_reduction"] >= 0.35
    assert development["terminal_wealth_ratio_vs_baseline"] < 0.80


def test_holdings_aware_notebook_uses_qlib_dataset_and_recorder():
    notebook = json.loads(
        (CARD / "holdings_aware_pit_allocation_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "import qlib" in text
    assert "qlib.init" in text
    assert "DatasetH" in text
    assert "from qlib.workflow import R" in text
    assert "run_holdings_aware_pit_allocation_card" in text
