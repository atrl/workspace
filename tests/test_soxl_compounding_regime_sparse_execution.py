from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.workflow.soxl_compounding_regime_sparse_execution import (
    classify_compounding_regime,
    select_structural_challenger,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "soxl_compounding_regime_sparse_execution_v1"


def test_regime_classifier_is_structural_and_fail_closed():
    assert classify_compounding_regime(
        SimpleNamespace(
            ret20=0.12,
            eff20=0.45,
            autocorr20=0.05,
            rv20=0.30,
            drawdown60=-0.02,
        )
    ) == "trend"
    assert classify_compounding_regime(
        SimpleNamespace(
            ret20=-0.12,
            eff20=0.30,
            autocorr20=-0.20,
            rv20=0.50,
            drawdown60=-0.16,
        )
    ) == "crash"
    assert classify_compounding_regime(
        SimpleNamespace(
            ret20=float("nan"),
            eff20=0.30,
            autocorr20=0.0,
            rv20=0.30,
            drawdown60=-0.02,
        )
    ) == "mean_reverting"


def test_selection_never_uses_outer_and_can_reject_all_challengers():
    rows = pd.DataFrame(
        [
            {
                "profile": "confidence_veto_baseline",
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "all_gates_pass": False,
                "return_pct": 425.0,
                "open_trade_days": 1700,
            },
            {
                "profile": "failed_challenger",
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "all_gates_pass": False,
                "return_pct": 100.0,
                "open_trade_days": 300,
            },
            {
                "profile": "outer_leak",
                "segment": "outer_reused_diagnostic",
                "selection_uses_outer_test": True,
                "all_gates_pass": True,
                "return_pct": 999.0,
                "open_trade_days": 1,
            },
        ]
    )
    assert select_structural_challenger(rows) is None


def test_generated_card_is_recorder_first_and_keeps_baseline_on_rejection():
    if not (CARD / "summary.json").exists():
        return
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_role"] == "reused_diagnostic_only_not_model_selection"
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["retained_model"] == "confidence_veto_baseline"
    assert summary["production_change"] == "blocked"


def test_notebook_contains_executable_qlib_and_chinese_conclusion_cells():
    notebook_path = CARD / "soxl_compounding_regime_sparse_execution_v1.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "qlib.init" in source
    assert "D.calendar" in source
    assert "D.features" in source
    assert "DatasetH" in source
    assert "run_soxl_compounding_regime_sparse_execution_card" in source
    assert "中文结论" in source
