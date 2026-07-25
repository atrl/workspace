from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.workflow.standalone_tqqq_qqq_cash_training import (
    CANDIDATES,
    FORBIDDEN_TRAINING_ASSETS,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "standalone_tqqq_qqq_cash_training_v1"


def test_standalone_candidates_never_enable_auxiliary_assets():
    assert all(candidate.auxiliary_weight == 0.0 for candidate in CANDIDATES)
    assert FORBIDDEN_TRAINING_ASSETS == ("SOXL", "SOXX")


def test_standalone_tqqq_card_contract_and_data_isolation():
    required = {
        "standalone_tqqq_qqq_cash_training_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "model_selection.csv",
        "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics.csv",
        "risk_analysis.csv",
        "training_data_audit.csv",
        "objective_gate_audit.csv",
        "analysis_summary.json",
        "failure_modes.md",
    }
    assert required.issubset(
        {path.name for path in CARD.iterdir() if path.is_file()}
    )
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (
            CARD / "standalone_tqqq_qqq_cash_training_v1.ipynb"
        ).read_text(encoding="utf-8")
    )
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert summary["provider_uri"].endswith("/data/qlib/tqqq")
    assert summary["training_assets"] == ["TQQQ", "QQQ"]
    assert summary["forbidden_training_assets"] == ["SOXL", "SOXX"]
    assert summary["forbidden_training_rows"] == 0
    assert summary["dataset_audit"]["instruments"] == ["TQQQ"]
    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_role"] == "report_only"
    assert (
        summary["artifact_policy"]["policy"]
        == "slim_card_recorder_first"
    )
    assert "import qlib" in notebook_text
    assert "DatasetH" in notebook_text
    assert "from qlib.workflow import R" in notebook_text
    assert "中文结论" in notebook_text

    diagnostics = pd.read_csv(CARD / "training_data_audit.csv")
    assert set(diagnostics["training_assets"]) == {"TQQQ"}
    assert set(diagnostics["underlying_data"]) == {"QQQ"}
    assert diagnostics["forbidden_training_rows"].eq(0).all()
    assert diagnostics["auxiliary_train_rows"].eq(0.0).all()
    assert not diagnostics["selection_uses_outer_test"].astype(bool).any()
    assert set(
        diagnostics.loc[
            diagnostics["segment"].eq("outer_diagnostic"),
            "train_end",
        ]
    ) == {"2022-12-30"}
