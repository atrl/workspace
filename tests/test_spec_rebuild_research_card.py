from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "spec_rebuild_foundation_v1"


def test_research_card_has_executable_notebook_sidecars_and_chinese_conclusion():
    notebook_path = CARD / "spec_rebuild_foundation_v1.ipynb"
    expected_sidecars = {
        "summary.json",
        "analysis_summary.json",
        "regime_metrics.csv",
        "window_metrics.csv",
        "model_comparison.csv",
        "failure_modes.md",
        "conclusion.md",
    }

    assert notebook_path.exists()
    assert expected_sidecars == {path.name for path in CARD.iterdir() if path.name != notebook_path.name}

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "中文结论" in markdown
    assert "qlib calendar" in markdown

    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "spec_rebuild_foundation_v1"
    assert summary["source"] == "qlib_calendar_plus_canonical_snapshot"
    assert "analysis_summary.json" in summary["artifacts"]
    assert "summary.json" not in summary["artifacts"]

    regime_metrics = pd.read_csv(CARD / "regime_metrics.csv")
    window_metrics = pd.read_csv(CARD / "window_metrics.csv")
    model_comparison = pd.read_csv(CARD / "model_comparison.csv")

    assert {
        "vix_regime",
        "trend_regime",
        "volume_regime",
        "drawdown_regime",
        "volatility_regime",
        "market_regime",
    }.issubset(regime_metrics.columns)
    assert {"window", "return_pct", "max_drawdown_pct", "turnover"}.issubset(window_metrics.columns)
    assert model_comparison.iloc[0]["model"] == "mean_return_smoke"
