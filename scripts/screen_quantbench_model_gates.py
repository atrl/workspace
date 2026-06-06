#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.quantbench_gates import scan_model_replays, write_scan_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen SOXL model replays against QuantBench-style drawdown/BuyHold gates")
    parser.add_argument("--max-drawdown-pct", type=float, default=20.0)
    parser.add_argument("--required-windows", default="test_2022,test_2023_2024")
    parser.add_argument(
        "--output-dir",
        default="runtime/model_selection/card/quantbench_model_gate_scan_v1",
        help="Runtime card directory for summary.json and model_gate_scan.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    required_windows = tuple(item.strip() for item in args.required_windows.split(",") if item.strip())
    result = scan_model_replays(root=ROOT, required_windows=required_windows, max_drawdown_pct=args.max_drawdown_pct)
    write_scan_outputs(result, output_dir=ROOT / args.output_dir)
    print(
        "quantbench_model_gate_scan "
        f"profiles={result['summary']['profiles_scanned']} "
        f"passing={result['summary']['passing_profiles']} "
        f"output={args.output_dir}"
    )
    return result


if __name__ == "__main__":
    main()
