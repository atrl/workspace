#!/usr/bin/env python3
"""增量更新 SOXL/SOXX/VIX 日数据到主 CSV 并刷新 qlib 二进制存储。

用法:
    python scripts/update_market_data.py [--dry-run] [--force-full]

环境依赖:
    yfinance (已包含在 requirements/qlib.txt 中)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import refresh_qlib_binary
from src.soxl_quant.settings import CSV_PATH, QLIB_DATA_PATH


def _fetch_yfinance(ticker: str, start: str | None = None, retries: int = 3) -> pd.DataFrame:
    """从 yfinance 拉取数据，返回标准列名的 DataFrame。"""
    import time
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # Use Ticker.history() instead of download() to avoid batch rate limits
            tk = yf.Ticker(ticker)
            if start:
                df = tk.history(start=start, auto_adjust=True)
            else:
                df = tk.history(period="max", auto_adjust=True)
            if df.empty:
                raise RuntimeError(f"No data returned for {ticker} from yfinance")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            # yfinance Ticker.history returns 'date' or 'index' column
            if "date" not in df.columns and "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})
            if "adj_close" in df.columns and "close" not in df.columns:
                df["close"] = df["adj_close"]
            return df[["date", "open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                sleep_sec = attempt * 3
                print(f"[yfinance] {ticker} attempt {attempt} failed: {exc}. Retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to fetch {ticker} from yfinance after {retries} attempts: {last_exc}")


def _load_master_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Master CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def update_master_csv(
    csv_path: Path = CSV_PATH,
    dry_run: bool = False,
    force_full: bool = False,
) -> dict:
    """增量更新主 CSV。

    Returns:
        {"before_rows": int, "after_rows": int, "new_rows": int, "last_date": str}
    """
    master = _load_master_csv(csv_path)
    before_rows = len(master)
    last_date = master["date"].max()

    if force_full:
        start = None
    else:
        # 拉取最后日期前 5 天起的数据，保证有足够 overlap 去重
        start_dt = last_date - timedelta(days=10)
        start = start_dt.strftime("%Y-%m-%d")

    print(f"[update] last_date={last_date.date()} fetch_start={start or 'max'}")

    soxl = _fetch_yfinance("SOXL", start=start)
    soxx = _fetch_yfinance("SOXX", start=start)
    vix = _fetch_yfinance("^VIX", start=start).drop(columns=["open", "high", "low", "volume"]).rename(
        columns={"close": "vix"}
    )

    soxl = soxl.rename(
        columns={
            "open": "soxl_open",
            "high": "soxl_high",
            "low": "soxl_low",
            "close": "soxl_close",
            "volume": "soxl_volume",
        }
    )
    soxx = soxx.rename(
        columns={
            "open": "soxx_open",
            "high": "soxx_high",
            "low": "soxx_low",
            "close": "soxx_close",
            "volume": "soxx_volume",
        }
    )

    merged = (
        soxl.merge(soxx, on="date", how="inner")
        .merge(vix, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if merged.empty:
        raise RuntimeError("No overlapping dates for SOXL/SOXX/VIX")

    # 合并策略：新数据与旧数据 concat 后按 date 去重，保留新数据
    master["date"] = pd.to_datetime(master["date"])
    merged["date"] = pd.to_datetime(merged["date"])

    combined = pd.concat([master, merged], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    after_rows = len(combined)
    new_rows = after_rows - before_rows
    new_last_date = combined["date"].max()

    print(
        f"[update] before={before_rows} after={after_rows} new_rows={new_rows} "
        f"last_date={new_last_date.date()}"
    )

    if dry_run:
        print("[update] dry-run: skip write")
    else:
        combined.to_csv(csv_path, index=False)
        print(f"[update] written {csv_path}")

    return {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "new_rows": new_rows,
        "last_date": new_last_date.strftime("%Y-%m-%d"),
        "dry_run": dry_run,
    }


def refresh_qlib(
    csv_path: Path = CSV_PATH,
    qlib_dir: Path = QLIB_DATA_PATH,
    dry_run: bool = False,
) -> None:
    if dry_run:
        print(f"[update] dry-run: skip refresh_qlib_binary({qlib_dir})")
        return
    refresh_qlib_binary(csv_path, qlib_dir, leveraged_etf="SOXL", underlying_etf="SOXX", leverage_ratio=3.0)
    print(f"[update] refreshed {qlib_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental market data updater for SOXL/SOXX/VIX")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    parser.add_argument("--force-full", action="store_true", help="Force full re-download (slow)")
    parser.add_argument("--skip-qlib", action="store_true", help="Skip qlib binary refresh")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    result = update_master_csv(dry_run=args.dry_run, force_full=args.force_full)
    if not args.skip_qlib:
        refresh_qlib(dry_run=args.dry_run)
    return result


if __name__ == "__main__":
    main()
