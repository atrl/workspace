#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.state_objective_budget_gap import write_budget_usage_gap_diagnostic


SOURCE_CARD = ROOT / "runtime/model_selection/card/state_objective_budget_usage_bc_official_replay_v1"
OUTPUT_DIR = ROOT / "runtime/model_selection/card/state_objective_budget_gap_diagnostic_v1"
RETENTION_GATE = ROOT / "runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json"


def _gate_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    return {
        "required_soxlbacktest_return_pct": float(summary["reference_return_pct"]),
        "required_best_return_pct": float(summary["required_best_return_pct"]),
        "max_walkforward_drawdown_pct": float(summary["max_walkforward_drawdown_pct"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Review structural gap after budget-usage BC official replay.")
    parser.add_argument("--source-card", type=Path, default=SOURCE_CARD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--retention-gate", type=Path, default=RETENTION_GATE)
    args = parser.parse_args()
    thresholds = _gate_thresholds(args.retention_gate)
    result = write_budget_usage_gap_diagnostic(
        source_card_dir=args.source_card,
        output_dir=args.output_dir,
        **thresholds,
    )
    print(
        "state_objective_budget_gap "
        f"decision={result.summary['decision']} "
        f"ppo_allowed={result.summary['ppo_allowed_from_current_target']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
