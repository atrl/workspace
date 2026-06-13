#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.model_retention_frontier import (  # noqa: E402
    scan_model_retention_frontier,
    write_retention_frontier_outputs,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Rank SOXL model-selection evidence for the next retention training run.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime" / "model_selection" / "card" / "model_retention_frontier_v1",
    )
    parser.add_argument("--card", action="append", dest="cards", help="Limit scan to one model-selection card name.")
    args = parser.parse_args(argv)

    result = scan_model_retention_frontier(root=args.root, card_names=args.cards)
    write_retention_frontier_outputs(result, output_dir=args.output_dir)
    summary = result["summary"]
    print(
        "model_retention_frontier "
        f"rows={summary['rows_scanned']} "
        f"recommended={summary.get('recommended_next_source')} "
        f"output={args.output_dir}"
    )
    return result


if __name__ == "__main__":
    main()
