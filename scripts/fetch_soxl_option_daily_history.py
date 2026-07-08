from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import qlib
from qlib.data import D

from src.data.options import (
    build_option_snapshot_manifest,
    fetch_polygon_option_contracts,
    fetch_polygon_option_daily_bars,
    merge_option_daily_bars,
    option_ticker_date_bounds,
    select_daily_option_targets,
    write_option_daily_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SOXL option daily history from Polygon/Massive.")
    parser.add_argument("--underlying", default="SOXL")
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-07-02")
    parser.add_argument("--fetch-until", default=date.today().isoformat())
    parser.add_argument("--provider-uri", default="data/qlib/soxl")
    parser.add_argument("--output-dir", default="data/raw/options/polygon_soxl_daily")
    parser.add_argument("--min-dte", type=int, default=25)
    parser.add_argument("--max-dte", type=int, default=35)
    parser.add_argument("--target-dte", type=int, default=30)
    parser.add_argument("--targets", nargs="+", default=["atm_call", "atm_put"])
    parser.add_argument("--sleep-seconds", type=float, default=13.0)
    parser.add_argument("--max-contracts", type=int, default=0, help="Optional debug cap for unique option tickers.")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise SystemExit("POLYGON_API_KEY or MASSIVE_API_KEY is required")

    underlying = args.underlying.upper()
    qlib.init(provider_uri=str(Path(args.provider_uri).resolve()), region="us", kernels=1)
    spot = D.features([underlying], ["$open"], start_time=args.start, end_time=args.end, freq="day")
    if spot.empty:
        raise SystemExit(f"no qlib spot rows for {underlying} {args.start}..{args.end}")
    spot_frame = (
        spot.reset_index()
        .rename(columns={"datetime": "date", "$open": "underlying_open"})
        .assign(underlying=underlying)
    )
    spot_frame["date"] = pd.to_datetime(spot_frame["date"]).dt.date.astype(str)
    spot_frame = spot_frame[["date", "underlying", "underlying_open"]].dropna()

    expiration_start = (pd.Timestamp(args.start).date() + timedelta(days=args.min_dte)).isoformat()
    expiration_end = (pd.Timestamp(args.fetch_until).date() + timedelta(days=args.max_dte)).isoformat()
    contract_parts: list[pd.DataFrame] = []
    contract_audits: list[dict[str, object]] = []
    for expired in [True, False]:
        contracts, audit = fetch_polygon_option_contracts(
            underlying=underlying,
            api_key=api_key,
            expiration_start=expiration_start,
            expiration_end=expiration_end,
            expired=expired,
        )
        contract_parts.append(contracts)
        contract_audits.append(audit)
        time.sleep(args.sleep_seconds)
    contracts = (
        pd.concat([part for part in contract_parts if not part.empty], ignore_index=True)
        if any(not part.empty for part in contract_parts)
        else pd.DataFrame()
    )
    if contracts.empty:
        raise SystemExit(f"no Polygon option contracts fetched for {underlying}")
    contracts = contracts.sort_values(["expiration_date", "strike_price", "contract_type"]).drop_duplicates(
        ["option_ticker"], keep="last"
    )

    selections = select_daily_option_targets(
        contracts,
        spot_frame,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        target_dte=args.target_dte,
        target_specs=args.targets,
    )
    if selections.empty:
        raise SystemExit("no option targets selected")

    bounds = option_ticker_date_bounds(selections, fetch_until=args.fetch_until)
    if args.max_contracts and args.max_contracts > 0:
        bounds = bounds.head(args.max_contracts).copy()

    bar_parts: list[pd.DataFrame] = []
    bar_audits: list[dict[str, object]] = []
    last_checkpoint_count = 0
    for idx, row in enumerate(bounds.itertuples(index=False), start=1):
        bars, audit = fetch_polygon_option_daily_bars(
            row.option_ticker,
            underlying=underlying,
            api_key=api_key,
            start=row.start,
            end=row.end,
        )
        bar_parts.append(bars)
        bar_audits.append(audit)
        print(
            f"[{idx}/{len(bounds)}] {row.option_ticker} {row.start}..{row.end} "
            f"{audit['status']} rows={audit['row_count']}"
        )
        should_checkpoint = args.checkpoint_every > 0 and idx % args.checkpoint_every == 0
        if should_checkpoint:
            partial_bars = _merge_bar_parts(bar_parts)
            partial_manifest = build_option_snapshot_manifest(
                underlying=underlying,
                provider="polygon",
                requested_start=args.start,
                requested_end=args.end,
                fetch_until=args.fetch_until,
                target_specs=args.targets,
                min_dte=args.min_dte,
                max_dte=args.max_dte,
                contracts=contracts,
                selections=selections,
                bars=partial_bars,
                contract_audits=contract_audits,
                bar_audits=bar_audits,
            )
            write_option_daily_snapshot(
                output_dir=Path(args.output_dir),
                contracts=contracts,
                selections=selections,
                bars=partial_bars,
                manifest=partial_manifest,
            )
            last_checkpoint_count = idx
            print(f"checkpoint wrote {idx}/{len(bounds)} to {Path(args.output_dir).resolve()}")
        if idx < len(bounds):
            time.sleep(args.sleep_seconds)
    bars = _merge_bar_parts(bar_parts)
    manifest = build_option_snapshot_manifest(
        underlying=underlying,
        provider="polygon",
        requested_start=args.start,
        requested_end=args.end,
        fetch_until=args.fetch_until,
        target_specs=args.targets,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        contracts=contracts,
        selections=selections,
        bars=bars,
        contract_audits=contract_audits,
        bar_audits=bar_audits,
    )
    write_option_daily_snapshot(
        output_dir=Path(args.output_dir),
        contracts=contracts,
        selections=selections,
        bars=bars,
        manifest=manifest,
    )
    print(f"wrote {Path(args.output_dir).resolve()}")
    print(
        f"contracts={len(contracts)} selections={len(selections)} "
        f"unique_selected={selections['option_ticker'].nunique()} bars={len(bars)} "
        f"last_checkpoint={last_checkpoint_count}"
    )
    return 0


def _merge_bar_parts(bar_parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not any(not part.empty for part in bar_parts):
        return pd.DataFrame()
    return merge_option_daily_bars(pd.DataFrame(), pd.concat([part for part in bar_parts if not part.empty], ignore_index=True))


if __name__ == "__main__":
    raise SystemExit(main())
