from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_model_matrix_dashboard(card_dir: str | Path, *, top_n: int = 10) -> dict[str, Any]:
    card = Path(card_dir)
    summary = json.loads((card / "analysis_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(card / "model_comparison.csv")
    regime_metrics = pd.read_csv(card / "regime_metrics.csv")
    window_metrics = pd.read_csv(card / "window_metrics.csv")
    return {
        "summary": summary,
        "top_models": comparison.head(top_n).to_dict(orient="records"),
        "regime_metrics": regime_metrics.to_dict(orient="records"),
        "window_metrics": window_metrics.to_dict(orient="records"),
        "artifacts": {
            "summary": str(card / "analysis_summary.json"),
            "model_comparison": str(card / "model_comparison.csv"),
            "regime_metrics": str(card / "regime_metrics.csv"),
            "window_metrics": str(card / "window_metrics.csv"),
        },
    }
