from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "production" / "paper_sim_replay_v1"


def test_production_card_is_paper_sim_only_with_order_intent_ledger():
    notebook_path = CARD / "paper_sim_replay_v1.ipynb"
    expected_files = {
        "paper_sim_replay_v1.ipynb",
        "summary.json",
        "order_intents.csv",
        "positions.csv",
        "conclusion.md",
    }

    assert expected_files == {path.name for path in CARD.iterdir()}

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "中文结论" in markdown
    assert "paper/sim" in markdown
    assert "真实 broker" in markdown

    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    assert summary["strategy"] == "paper_sim_replay_v1"
    assert summary["execution_mode"] == "paper_sim"
    assert summary["real_broker_orders"] is False

    order_intents = pd.read_csv(CARD / "order_intents.csv")
    positions = pd.read_csv(CARD / "positions.csv")

    assert order_intents.to_dict("records") == [
        {"symbol": "SOXL", "target_percent": 0.25, "quantity": 50, "price": 50.0}
    ]
    assert positions.to_dict("records") == [{"symbol": "SOXL", "quantity": 50}]
