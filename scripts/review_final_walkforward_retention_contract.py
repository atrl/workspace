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

from src.soxl_quant.app.final_walkforward_retention_contract import (  # noqa: E402
    derive_final_walkforward_retention_contract,
    load_latest_reference_baseline,
    write_final_walkforward_retention_contract,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Build the final walk-forward retention contract for SOXL actor retention.",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--strict-gate-path",
        type=Path,
        default=ROOT / "runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json",
    )
    parser.add_argument(
        "--candidate-filter-path",
        type=Path,
        default=ROOT / "runtime/model_selection/card/strict_candidate_retention_filter_v1/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/model_selection/card/final_walkforward_retention_contract_v1",
    )
    args = parser.parse_args(argv)

    strict_gate_payload = _load_json(args.strict_gate_path)
    candidate_filter_payload = _load_json(args.candidate_filter_path)
    latest_reference_baseline = load_latest_reference_baseline(args.root)
    result = derive_final_walkforward_retention_contract(
        strict_gate_payload=strict_gate_payload,
        strict_candidate_filter_payload=candidate_filter_payload,
        latest_reference_baseline=latest_reference_baseline,
    )
    sidecars = write_final_walkforward_retention_contract(result, output_dir=args.output_dir)

    source_manifest = pd.DataFrame(
        [
            {
                "source_label": "strict_walkforward_gate",
                "path": str(args.strict_gate_path.relative_to(args.root)),
                "exists": args.strict_gate_path.exists(),
                "role": "strict walk-forward drawdown and return gate rows",
            },
            {
                "source_label": "strict_candidate_retention_filter",
                "path": str(args.candidate_filter_path.relative_to(args.root)),
                "exists": args.candidate_filter_path.exists(),
                "role": "actor-retention allow/deny filter",
            },
            {
                "source_label": "latest_soxlbacktest_reference",
                "path": str(latest_reference_baseline["source_path"]),
                "exists": True,
                "role": "fresh SOXLBacktest buy/sell reference return and drawdown",
            },
        ]
    )
    source_manifest_path = args.output_dir / "source_manifest.csv"
    source_manifest.to_csv(source_manifest_path, index=False)
    result["source_manifest"] = source_manifest.to_dict(orient="records")
    result["sidecars"] = {
        **{key: str(path.relative_to(args.root)) for key, path in sidecars.items()},
        "source_manifest_csv": str(source_manifest_path.relative_to(args.root)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    print(
        "final_walkforward_retention_contract "
        f"decision={summary['decision']} "
        f"final_retained={summary['final_retained_count']} "
        f"output={args.output_dir}"
    )
    return result


if __name__ == "__main__":
    main()
