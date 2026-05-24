#!/usr/bin/env python3
"""Find poor OOS excess windows and explain them with feature deltas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.oos_diagnostics import run_oos_diagnostic  # noqa: E402


EXTERNAL_REFERENCES = [
    {
        "label": "Deflated Sharpe Ratio",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
        "note": "selection-bias and multiple-testing warning for repeated strategy trials",
    },
    {
        "label": "Probability of Backtest Overfitting",
        "url": "https://doi.org/10.21314/JCF.2016.322",
        "note": "walk-forward and OOS discipline motivation",
    },
    {
        "label": "Qlib GitHub",
        "url": "https://github.com/microsoft/qlib",
        "note": "keep diagnostics inside the qlib research-to-production workflow",
    },
    {
        "label": "Qlib arXiv",
        "url": "https://arxiv.org/abs/2009.11189",
        "note": "AI quant platform scope: data, learning, backtest, and deployment infrastructure",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-csv", required=True, type=Path, help="Daily replay CSV with strategy/buyhold NAV")
    parser.add_argument("--profile", required=True, help="Profile or diagnostic label")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--oos-start", default="2020-01-02")
    parser.add_argument("--oos-end", default="")
    parser.add_argument("--window-size", type=int, default=63)
    parser.add_argument("--top-windows", type=int, default=12)
    parser.add_argument("--top-features", type=int, default=30)
    parser.add_argument("--target-excess", type=float, default=30.0)
    parser.add_argument("--target-max-dd", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_oos_diagnostic(
        daily_csv=args.daily_csv,
        profile=args.profile,
        output_dir=args.output_dir,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        window_size=args.window_size,
        top_windows=args.top_windows,
        top_features=args.top_features,
        target_excess=args.target_excess,
        target_max_dd=args.target_max_dd,
        external_references=EXTERNAL_REFERENCES,
    )
    print(
        "oos_diagnostic "
        f"profile={args.profile} windows={manifest['windows']} features={manifest['features']} "
        f"oos={args.oos_start}..{manifest['oos_end']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
