#!/usr/bin/env python3
from __future__ import annotations

"""Fetch optional Tushare Pro inputs into repo-owned raw external CSVs."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.core.tushare import fetch_us_daily_adj, tushare_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch optional Tushare Pro cached data")
    parser.add_argument("--api", choices=["us_daily_adj"], default="us_daily_adj")
    parser.add_argument("--symbols", nargs="+", default=["SOXX", "SOXL"])
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    parser.add_argument(
        "--allow-missing-token",
        action="store_true",
        help="exit successfully without fetching when the token environment variable is absent",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        msg = f"missing {args.token_env}; set it or pass --allow-missing-token for optional dry integration"
        if args.allow_missing_token:
            print(f"fetch_tushare_data: SKIP {msg}")
            return 0
        print(f"fetch_tushare_data: FAIL {msg}", file=sys.stderr)
        return 2

    paths: list[Path] = []
    for symbol in args.symbols:
        if args.api == "us_daily_adj":
            path = fetch_us_daily_adj(
                symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                token=token,
                output_path=tushare_csv_path("us_daily_adj", symbol),
            )
        else:  # pragma: no cover - argparse choices guard this.
            raise ValueError(args.api)
        paths.append(path)
        print(f"fetch_tushare_data: wrote {path.relative_to(ROOT)}")

    print(f"fetch_tushare_data: OK files={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
