#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.dashboard_v2 import build_dashboard_v2_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dashboard v2 JSON data contract.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root that owns source dashboard/runtime data.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dashboard_data" / "v2")
    parser.add_argument(
        "--freshness-path",
        type=Path,
        help="Path to dashboard_data/model_replay_freshness.json; defaults under --root.",
    )
    parser.add_argument("--profile", action="append", dest="profiles", help="Profile to include; may be repeated.")
    parser.add_argument("--baseline", help="Baseline profile for comparison.")
    args = parser.parse_args()

    result = build_dashboard_v2_data(
        root=args.root,
        output_dir=args.output_dir,
        freshness_path=args.freshness_path or args.root / "dashboard_data" / "model_replay_freshness.json",
        selected_profiles=args.profiles,
        baseline_profile=args.baseline,
    )
    print(f"dashboard_v2_data={result.output_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"comparison={result.comparison_path}")
    print(f"models={len(result.model_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
