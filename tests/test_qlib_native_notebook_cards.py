from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _summary(card: Path) -> dict:
    return json.loads((card / "summary.json").read_text(encoding="utf-8"))


def test_research_card_has_executed_qlib_recorder_and_analysis_sidecars():
    card = ROOT / "research" / "qlib_native_reference_v1"
    summary = _summary(card)
    analysis = json.loads((card / "analysis_summary.json").read_text(encoding="utf-8"))
    notebook_text = (card / "qlib_native_reference_v1.ipynb").read_text(encoding="utf-8")

    assert summary["experiment"] == "qlib_native_reference_v1"
    assert "qlib.workflow.R.start" in summary["qlib_native_api"]
    assert "SignalRecord" in summary["qlib_native_api"]
    assert "SigAnaRecord" in summary["qlib_native_api"]
    assert analysis["source"] == "qlib_recorder"
    assert analysis["recorder_id"] == summary["recorder_id"]
    assert set(summary["sidecars"]) >= {
        "predictions",
        "signals",
        "portfolio_report",
        "analysis_summary",
        "regime_metrics",
        "window_metrics",
        "model_comparison",
        "qlib_portfolio_report",
        "risk_analysis",
        "explainability",
        "failure_modes",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "pred.pkl").exists()
    assert (recorder_dir / "artifacts" / "label.pkl").exists()
    assert (recorder_dir / "artifacts" / "sig_analysis" / "ic.pkl").exists()
    assert (recorder_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl").exists()
    assert (recorder_dir / "artifacts" / "portfolio_analysis" / "port_analysis_1day.pkl").exists()
    assert not (recorder_dir / "artifacts" / "code_status.txt").exists()
    assert summary["recorder_id"] in notebook_text
    assert "PortAnaRecord" in notebook_text
    assert "risk_analysis.csv" in notebook_text


def test_production_card_is_paper_sim_and_references_qlib_recorder():
    card = ROOT / "production" / "qlib_paper_sim_v1"
    summary = _summary(card)
    production_summary = json.loads((card / "production_summary.json").read_text(encoding="utf-8"))
    notebook_text = (card / "qlib_paper_sim_v1.ipynb").read_text(encoding="utf-8")

    assert summary["experiment"] == "qlib_paper_sim_v1"
    assert production_summary["execution_mode"] == "paper_sim"
    assert production_summary["real_broker_orders"] is False
    assert production_summary["recorder_id"] == summary["recorder_id"]
    assert (card / production_summary["order_intents"]).exists()
    assert (card / production_summary["positions"]).exists()
    assert "qlib.data.D.features" in production_summary["qlib_native_api"]
    assert "SignalRecord" in production_summary["qlib_native_api"]
    assert "PortAnaRecord" in production_summary["qlib_native_api"]
    assert summary["recorder_id"] in notebook_text
    assert "risk_analysis.csv" in notebook_text
