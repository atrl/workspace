#!/usr/bin/env python3
from __future__ import annotations

"""Fetch SOXL/SOXX intraday OHLCV into canonical repo-owned parquet files."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.intraday_data import (
    FutuIntradayProvider,
    PolygonIntradayProvider,
    canonical_intraday_path,
    intraday_feature_csv_path,
    intraday_raw_path,
    write_intraday_outputs,
)

SUPPORTED_PROVIDERS = ("futu", "polygon")
SUPPORTED_SYMBOLS = ("SOXL", "SOXX")
SUPPORTED_INTERVALS = ("1m",)


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_items(label: str, values: list[str], allowed: tuple[str, ...]) -> list[str]:
    normalized = [value.lower() if label == "providers" else value.upper() for value in values]
    invalid = sorted(set(normalized) - set(allowed))
    if invalid:
        raise ValueError(f"unknown {label}: {invalid}; allowed={list(allowed)}")
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Futu/Polygon 1-minute SOXL/SOXX data")
    parser.add_argument("--providers", default="futu,polygon", help="Comma-separated providers: futu,polygon")
    parser.add_argument("--symbols", default="SOXL,SOXX", help="Comma-separated symbols: SOXL,SOXX")
    parser.add_argument("--interval", default="1m", choices=SUPPORTED_INTERVALS)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--session", default="regular", choices=("regular", "extended"))
    parser.add_argument("--dry-run", action="store_true", help="Print plan without contacting providers or writing files")
    parser.add_argument("--output-root", type=Path, default=ROOT, help="Repository root for outputs")
    parser.add_argument("--futu-host", default="127.0.0.1")
    parser.add_argument("--futu-port", type=int, default=11111)
    return parser.parse_args(argv)


def _provider_instances(args: argparse.Namespace) -> dict[str, object]:
    return {
        "futu": FutuIntradayProvider(host=args.futu_host, port=args.futu_port),
        "polygon": PolygonIntradayProvider(),
    }


def _print_dry_run(providers: list[str], symbols: list[str], args: argparse.Namespace) -> None:
    print(
        "fetch_intraday_data: DRY-RUN "
        f"providers={providers} symbols={symbols} interval={args.interval} "
        f"start={args.start} end={args.end} session={args.session}"
    )
    for provider in providers:
        for symbol in symbols:
            path = intraday_raw_path(
                provider,
                symbol,
                args.interval,
                args.start,
                args.end,
                args.session,
                root=args.output_root,
            )
            print(f"- raw {provider}/{symbol}: {path.relative_to(args.output_root)}")
    print(f"- canonical: {canonical_intraday_path(root=args.output_root).relative_to(args.output_root)}")
    print(f"- features: {intraday_feature_csv_path(root=args.output_root).relative_to(args.output_root)}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        providers = _validate_items("providers", _comma_list(args.providers), SUPPORTED_PROVIDERS)
        symbols = _validate_items("symbols", _comma_list(args.symbols), SUPPORTED_SYMBOLS)
    except SystemExit as exc:
        return int(exc.code)
    except ValueError as exc:
        print(f"fetch_intraday_data: FAIL {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _print_dry_run(providers, symbols, args)
        return 0

    instances = _provider_instances(args)
    frames = {}
    for provider_name in providers:
        provider = instances[provider_name]
        for symbol in symbols:
            frame = provider.fetch(
                symbol,
                interval=args.interval,
                start=args.start,
                end=args.end,
                session=args.session,
            )
            frames[(provider_name, symbol)] = frame
            print(f"fetch_intraday_data: fetched provider={provider_name} symbol={symbol} rows={len(frame)}")

    paths = write_intraday_outputs(
        frames,
        interval=args.interval,
        start=args.start,
        end=args.end,
        session=args.session,
        root=args.output_root,
    )
    for label, path in sorted(paths.items()):
        print(f"fetch_intraday_data: wrote {label} {path.relative_to(args.output_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
