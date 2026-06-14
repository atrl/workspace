#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.model_retention_gate import (  # noqa: E402
    scan_model_retention_gates,
    write_registry_retention_updates,
    write_retention_gate_outputs,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Review replayable SOXL models against retention walk-forward gates.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1",
    )
    parser.add_argument("--max-walkforward-drawdown-pct", type=float, default=20.0)
    parser.add_argument("--best-return-ratio", type=float, default=0.95)
    parser.add_argument(
        "--apply-registry-retention",
        action="store_true",
        help=(
            "Apply the strict retention plan to the registry by setting every failed "
            "keep_actor row to false. Actor files are not deleted."
        ),
    )
    args = parser.parse_args(argv)

    result = scan_model_retention_gates(
        root=args.root,
        max_walkforward_drawdown_pct=args.max_walkforward_drawdown_pct,
        best_return_ratio=args.best_return_ratio,
    )
    write_retention_gate_outputs(result, output_dir=args.output_dir)
    changed: list[str] = []
    if args.apply_registry_retention:
        changed = write_registry_retention_updates(args.root, result)
    summary = result["summary"]
    print(
        "model_retention_gate "
        f"profiles={summary['profiles_scanned']} "
        f"passing={summary['passing_profile_count']} "
        f"disable_plan={summary.get('registry_keep_actor_disable_count', 0)} "
        f"registry_changed={len(changed)} "
        f"output={args.output_dir}"
    )
    return result


if __name__ == "__main__":
    main()
