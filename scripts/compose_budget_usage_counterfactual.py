#!/usr/bin/env python3
"""Compose and score a budget-usage counterfactual from two daily policy paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.budget_counterfactual import (  # noqa: E402
    build_budget_usage_counterfactual,
    write_counterfactual_artifacts,
)
from src.soxl_quant.data import build_dataset, init_qlib  # noqa: E402
from src.soxl_quant.settings import QLIB_DATA_PATH  # noqa: E402


def _default_label(path: Path) -> str:
    return path.parent.name or path.stem.replace("_soxl_daily", "")


def relative_output_label(path: Path, root: Path = ROOT) -> str:
    base = path if path.is_absolute() else root / path
    try:
        return base.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a budget-usage counterfactual from slow/fast daily CSVs")
    parser.add_argument("--slow", required=True, type=Path, help="Slow risk-mode daily CSV")
    parser.add_argument("--fast", required=True, type=Path, help="Fast usage daily CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSON/CSV/Markdown artifacts")
    parser.add_argument("--profile", default="", help="Output profile/card key")
    parser.add_argument("--slow-label", default="", help="Label for the slow path")
    parser.add_argument("--fast-label", default="", help="Label for the fast path")
    parser.add_argument("--composed-label", default="", help="Label for the composed path")
    parser.add_argument("--description", default="", help="Review description")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = args.profile or args.output_dir.name
    slow_label = args.slow_label or _default_label(args.slow)
    fast_label = args.fast_label or _default_label(args.fast)
    composed_label = args.composed_label or profile
    description = args.description or (
        f"Fixed-path budget-usage counterfactual: `{slow_label}` sets the slow risk mode "
        f"and `{fast_label}` is interpreted as usage inside that mode band."
    )

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    price = dataset.handler.price_frame.droplevel("instrument")["price"]

    result = build_budget_usage_counterfactual(
        pd.read_csv(args.slow),
        pd.read_csv(args.fast),
        price,
        slow_label=slow_label,
        fast_label=fast_label,
        composed_label=composed_label,
        profile=profile,
        description=description,
    )
    artifacts = write_counterfactual_artifacts(result, args.output_dir, price)
    composed = result.payload["variants"][composed_label]["test_proxy"]
    print(
        f"[budget_counterfactual] wrote {relative_output_label(args.output_dir)} "
        f"return={composed['return']:.1f}% sharpe={composed['sharpe']:.3f} "
        f"max_dd={composed['max_dd']:.1f}% avg_pos={composed['avg_pos']:.3f}"
    )
    for kind, path in artifacts.items():
        print(f"  {kind}: {relative_output_label(path)}")


if __name__ == "__main__":
    main()
