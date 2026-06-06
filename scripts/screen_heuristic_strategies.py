#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.heuristic_strategy_search import default_heuristic_config_grid, screen_heuristic_strategies


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen SOXL heuristic strategies against QuantBench-style gates")
    parser.add_argument("--panel", default="data/processed/panels/soxl_soxx_vix_full.csv")
    parser.add_argument("--output-dir", default="runtime/model_selection/card/heuristic_strategy_search_v1")
    parser.add_argument("--max-drawdown-pct", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    panel_path = ROOT / args.panel
    panel = pd.read_csv(panel_path)
    rows = screen_heuristic_strategies(
        panel,
        default_heuristic_config_grid(),
        max_drawdown_pct=args.max_drawdown_pct,
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "heuristic_strategy_screen.csv"
    summary_path = output_dir / "summary.json"
    rows.to_csv(csv_path, index=False)
    passing = rows.loc[rows["gate_pass"] == True]  # noqa: E712 - pandas scalar comparison
    summary = {
        "experiment": "heuristic_strategy_search_v1",
        "panel": args.panel,
        "configs": int(len(rows)),
        "passing_configs": int(len(passing)),
        "decision": "hard_gate_pass_found" if len(passing) else "hard_gate_no_heuristic_passes",
        "promotion_allowed": bool(len(passing)),
        "training_allowed_from_this_card": False,
        "max_drawdown_pct": float(args.max_drawdown_pct),
        "required_windows": ["test_2022", "test_2023_2024"],
        "best": rows.head(10).to_dict(orient="records"),
        "next_action": (
            "Add a stronger reentry/ensemble or rolling learner branch; simple causal momentum/VIX heuristics "
            "did not clear the 2023-2024 return plus drawdown conflict."
        ),
        "sidecars": ["summary.json", "heuristic_strategy_screen.csv"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "heuristic_strategy_search "
        f"configs={summary['configs']} passing={summary['passing_configs']} output={args.output_dir}"
    )
    return summary


if __name__ == "__main__":
    main()
