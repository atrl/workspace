#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.strict_retraining_candidate_queue import (  # noqa: E402
    derive_strict_retraining_candidate_queue,
    write_strict_retraining_candidate_queue,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else payload


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Build the strict retraining candidate queue from current walk-forward retention evidence.",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--strict-gate-summary",
        type=Path,
        default=ROOT / "runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json",
    )
    parser.add_argument(
        "--retention-gate-csv",
        type=Path,
        default=ROOT / "runtime/model_selection/card/model_retention_walkforward_gate_v1/retention_gate.csv",
    )
    parser.add_argument(
        "--final-contract-summary",
        type=Path,
        default=ROOT / "runtime/model_selection/card/final_walkforward_retention_contract_v1/summary.json",
    )
    parser.add_argument(
        "--candidate-filter-summary",
        type=Path,
        default=ROOT / "runtime/model_selection/card/strict_candidate_retention_filter_v1/summary.json",
    )
    parser.add_argument(
        "--soxlbacktest-summary",
        type=Path,
        default=ROOT / "runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/summary.json",
    )
    parser.add_argument(
        "--retraining-handoff-summary",
        type=Path,
        default=ROOT / "runtime/model_selection/card/final_walkforward_retraining_handoff_v1/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/model_selection/card/strict_retraining_candidate_queue_v1",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    retention_rows = pd.read_csv(args.retention_gate_csv).where(pd.notna, None).to_dict(orient="records")
    result = derive_strict_retraining_candidate_queue(
        strict_gate_summary=_summary(args.strict_gate_summary),
        retention_gate_rows=retention_rows,
        final_contract_summary=_summary(args.final_contract_summary),
        candidate_filter_summary=_summary(args.candidate_filter_summary),
        soxlbacktest_summary=_summary(args.soxlbacktest_summary),
        retraining_handoff_summary=_summary(args.retraining_handoff_summary),
    )
    sidecars = write_strict_retraining_candidate_queue(result, output_dir=args.output_dir)

    source_manifest = pd.DataFrame(
        [
            {
                "source_label": "strict_walkforward_gate_summary",
                "path": _relative(root, args.strict_gate_summary),
                "exists": args.strict_gate_summary.exists(),
                "role": "strict WF gate summary and thresholds",
            },
            {
                "source_label": "strict_walkforward_gate_rows",
                "path": _relative(root, args.retention_gate_csv),
                "exists": args.retention_gate_csv.exists(),
                "role": "per-profile current gate failures and gaps",
            },
            {
                "source_label": "final_walkforward_retention_contract",
                "path": _relative(root, args.final_contract_summary),
                "exists": args.final_contract_summary.exists(),
                "role": "final actor-retention contract",
            },
            {
                "source_label": "strict_candidate_retention_filter",
                "path": _relative(root, args.candidate_filter_summary),
                "exists": args.candidate_filter_summary.exists(),
                "role": "no-promotion actor retention filter",
            },
            {
                "source_label": "soxlbacktest_qlib_walkforward_baseline",
                "path": _relative(root, args.soxlbacktest_summary),
                "exists": args.soxlbacktest_summary.exists(),
                "role": "SOXLBacktest qlib buy/sell reference baseline",
            },
            {
                "source_label": "final_walkforward_retraining_handoff",
                "path": _relative(root, args.retraining_handoff_summary),
                "exists": args.retraining_handoff_summary.exists(),
                "role": "screen-only next training handoff",
            },
        ]
    )
    source_manifest_path = args.output_dir / "source_manifest.csv"
    source_manifest.to_csv(source_manifest_path, index=False)
    sidecar_paths = {
        **{key: _relative(root, path) for key, path in sidecars.items()},
        "source_manifest_csv": _relative(root, source_manifest_path),
    }
    result["source_manifest"] = source_manifest.to_dict(orient="records")
    result["sidecars"] = sidecar_paths
    summary_payload = {
        **result["summary"],
        "sidecars": sidecar_paths,
        "source_count": len(result["source_manifest"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    print(
        "strict_retraining_candidate_queue "
        f"decision={summary['decision']} "
        f"top={summary['top_candidate_id']} "
        f"queue={summary['queue_count']} "
        f"output={args.output_dir}"
    )
    return result


if __name__ == "__main__":
    main()
