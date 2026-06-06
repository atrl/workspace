#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.intraday_data import (
    FutuIntradayProvider,
    PolygonIntradayProvider,
    build_intraday_daily_features,
    write_intraday_surfaces,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize SOXL/SOXX intraday bars")
    parser.add_argument("--providers", default="futu,polygon", help="Comma-separated providers: futu,polygon")
    parser.add_argument("--symbols", default="SOXL,SOXX", help="Comma-separated symbols")
    parser.add_argument("--interval", default="1m", help="Bar interval, e.g. 1m")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--session", default="regular", choices=["regular", "extended", "all"])
    parser.add_argument("--dry-run", action="store_true", help="Fetch and summarize without writing files")
    parser.add_argument("--futu-host", default="127.0.0.1")
    parser.add_argument("--futu-port", type=int, default=11111)
    parser.add_argument("--futu-timeout-sec", type=float, default=20.0, help="Timeout for Futu SDK connect/request calls")
    return parser.parse_args(argv)


def _provider_for_name(name: str, args: argparse.Namespace):
    if name == "futu":
        return FutuIntradayProvider(
            host=args.futu_host,
            port=args.futu_port,
            log_home=Path(tempfile.gettempdir()) / "soxl_futu_sdk_home",
            connect_timeout_sec=args.futu_timeout_sec,
        )
    if name == "polygon":
        return PolygonIntradayProvider()
    raise ValueError(f"Unsupported provider: {name}")


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    symbols = [item.strip().upper().replace("US.", "") for item in args.symbols.split(",") if item.strip()]
    raw_frames = {}
    for provider_name in providers:
        provider = _provider_for_name(provider_name, args)
        for symbol in symbols:
            frame = provider.fetch(symbol, start=args.start, end=args.end, interval=args.interval)
            raw_frames[(provider_name, symbol)] = frame
            print(f"[intraday] provider={provider_name} symbol={symbol} rows={len(frame)}")
    canonical, paths = write_intraday_surfaces(
        raw_frames,
        root=ROOT,
        interval=args.interval,
        session=args.session,
        dry_run=args.dry_run,
    )
    feature_path = ROOT / "data" / "raw" / "external" / f"intraday_features_soxl_{args.interval}.csv"
    features = build_intraday_daily_features(canonical, output_path=None if args.dry_run else feature_path)
    print(f"[intraday] canonical rows={len(canonical)} features={len(features)} dry_run={args.dry_run}")
    return {
        "rows": int(len(canonical)),
        "feature_rows": int(len(features)),
        "paths": paths,
        "dry_run": bool(args.dry_run),
    }


if __name__ == "__main__":
    main()
