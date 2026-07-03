from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import requests

from .market_update import REPO_ROOT, _fetch_yahoo_chart_payload


BLACKROCK_SOXX_FUND_DOCUMENT_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
)
BLACKROCK_SOXX_FUND_DOCUMENT_PARAMS = {
    "appType": "PRODUCT_PAGE",
    "appSubType": "ISHARES",
    "targetSite": "us-ishares",
    "locale": "en_US",
    "portfolioId": "239705",
    "component": "fundDownload",
    "userType": "individual",
}
SOXX_PRODUCT_URL = "https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf"
DEFAULT_HISTORY_START = "2010-03-11"
DEFAULT_HISTORY_END = "2026-07-02"
HOLDINGS_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "asset_class",
    "market_value",
    "weight_pct",
    "notional_value",
    "quantity",
    "price",
    "location",
    "exchange",
    "currency",
    "fx_rate",
    "accrual_date",
    "as_of",
]
PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


@dataclass(frozen=True)
class SoxxConstituentPaths:
    root: Path = REPO_ROOT / "data" / "raw" / "soxx_constituents"

    @property
    def holdings_dir(self) -> Path:
        return self.root / "holdings"

    @property
    def daily_dir(self) -> Path:
        return self.root / "daily"

    @property
    def coverage_csv(self) -> Path:
        return self.root / "coverage.csv"

    @property
    def manifest_json(self) -> Path:
        return self.root / "manifest.json"

    @property
    def equity_universe_csv(self) -> Path:
        return self.root / "equity_universe.csv"


@dataclass(frozen=True)
class SoxxHoldingsSnapshot:
    as_of: str
    number_of_securities: int | None
    rows: pd.DataFrame
    source_url: str

    @property
    def equity_rows(self) -> pd.DataFrame:
        equities = self.rows.loc[self.rows["asset_class"].eq("Equity")].copy()
        equities = equities.sort_values(["weight_pct", "ticker"], ascending=[False, True]).reset_index(drop=True)
        return equities


def _spreadsheet_rows(workbook_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_text in re.findall(r"<ss:Row[^>]*>(.*?)</ss:Row>", workbook_text, flags=re.S):
        values: list[str] = []
        for cell_text in re.findall(r"<ss:Cell[^>]*>(.*?)</ss:Cell>", row_text, flags=re.S):
            match = re.search(r"<ss:Data[^>]*>(.*?)</ss:Data>", cell_text, flags=re.S)
            raw = match.group(1) if match else ""
            cleaned = re.sub(r"<[^>]+>", "", raw)
            values.append(html.unescape(cleaned).strip())
        if any(value != "" for value in values):
            rows.append(values)
    return rows


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "--"}:
        return None
    return float(text)


def _parse_as_of(rows: Sequence[Sequence[str]]) -> str:
    for row in rows:
        if len(row) >= 2 and row[0] == "Fund Holdings as of":
            return pd.Timestamp(row[1]).date().isoformat()
    raise ValueError("SOXX holdings workbook missing 'Fund Holdings as of' row")


def _parse_number_of_securities(rows: Sequence[Sequence[str]]) -> int | None:
    for row in rows:
        if len(row) >= 2 and row[0] == "Number of Securities (excluding cash and derivatives)":
            value = _parse_float(row[1])
            return int(value) if value is not None else None
    return None


def parse_soxx_holdings_workbook(workbook_text: str, *, source_url: str = SOXX_PRODUCT_URL) -> SoxxHoldingsSnapshot:
    rows = _spreadsheet_rows(workbook_text)
    as_of = _parse_as_of(rows)
    number_of_securities = _parse_number_of_securities(rows)
    header_index = None
    for index, row in enumerate(rows):
        if row[:6] == ["Ticker", "Name", "Sector", "Asset Class", "Market Value", "Weight (%)"]:
            header_index = index
            break
    if header_index is None:
        raise ValueError("SOXX holdings workbook missing holdings table header")

    header = rows[header_index]
    records: list[dict[str, Any]] = []
    for row in rows[header_index + 1 :]:
        if len(row) < 6:
            continue
        if row[0] in {"", "Ticker"}:
            continue
        record = {header[column_index]: row[column_index] if column_index < len(row) else "" for column_index in range(len(header))}
        if record.get("Asset Class") not in {"Equity", "Money Market", "Cash Collateral and Margins", "Futures", "Cash"}:
            if records:
                break
            continue
        records.append(
            {
                "ticker": record.get("Ticker", "").strip().upper(),
                "name": record.get("Name", "").strip(),
                "sector": record.get("Sector", "").strip(),
                "asset_class": record.get("Asset Class", "").strip(),
                "market_value": _parse_float(record.get("Market Value")),
                "weight_pct": _parse_float(record.get("Weight (%)")),
                "notional_value": _parse_float(record.get("Notional Value")),
                "quantity": _parse_float(record.get("Quantity")),
                "price": _parse_float(record.get("Price")),
                "location": record.get("Location", "").strip(),
                "exchange": record.get("Exchange", "").strip(),
                "currency": record.get("Currency", "").strip(),
                "fx_rate": _parse_float(record.get("FX Rate")),
                "accrual_date": record.get("Accrual Date", "").strip(),
                "as_of": as_of,
            }
        )

    holdings = pd.DataFrame(records, columns=HOLDINGS_COLUMNS)
    if holdings.empty:
        raise ValueError("SOXX holdings workbook produced no holding rows")
    return SoxxHoldingsSnapshot(
        as_of=as_of,
        number_of_securities=number_of_securities,
        rows=holdings,
        source_url=source_url,
    )


def fetch_soxx_holdings_snapshot(
    *,
    timeout: float = 30.0,
    request_get: Callable[..., requests.Response] = requests.get,
) -> SoxxHoldingsSnapshot:
    response = request_get(
        BLACKROCK_SOXX_FUND_DOCUMENT_URL,
        params=BLACKROCK_SOXX_FUND_DOCUMENT_PARAMS,
        timeout=timeout,
        headers={
            "Accept": "application/vnd.ms-excel,*/*",
            "Referer": SOXX_PRODUCT_URL,
            "User-Agent": "Mozilla/5.0 soxl-qlib-soxx-constituent-data",
        },
    )
    response.raise_for_status()
    return parse_soxx_holdings_workbook(response.text, source_url=response.url)


def normalize_yahoo_ohlcv_payload(symbol: str, payload: dict[str, Any]) -> pd.DataFrame:
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"yahoo chart error for {symbol}: {error}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"yahoo chart returned no result for {symbol}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_block = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj_block = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").date.astype(str),
            "open": quote_block.get("open", []),
            "high": quote_block.get("high", []),
            "low": quote_block.get("low", []),
            "close": quote_block.get("close", []),
            "adj_close": adj_block.get("adjclose", []),
            "volume": quote_block.get("volume", []),
        }
    )
    for column in PRICE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).drop_duplicates("date", keep="last")
    return frame.loc[:, PRICE_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_yahoo_daily_ohlcv(symbol: str, *, start: str, end: str) -> pd.DataFrame:
    return normalize_yahoo_ohlcv_payload(symbol, _fetch_yahoo_chart_payload(symbol, start=start, end=end))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _coverage_row(
    *,
    symbol: str,
    name: str,
    weight_pct: float | None,
    frame: pd.DataFrame | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "symbol": symbol,
            "name": name,
            "weight_pct": weight_pct,
            "status": status,
            "rows": 0,
            "start_date": None,
            "end_date": None,
            "missing_ohlcv_rows": None,
            "error": error,
        }
    missing = int(frame[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
    return {
        "symbol": symbol,
        "name": name,
        "weight_pct": weight_pct,
        "status": status,
        "rows": int(len(frame)),
        "start_date": str(frame["date"].min()),
        "end_date": str(frame["date"].max()),
        "missing_ohlcv_rows": missing,
        "error": error,
    }


def fetch_and_store_soxx_constituent_history(
    *,
    start: str = DEFAULT_HISTORY_START,
    end: str = DEFAULT_HISTORY_END,
    paths: SoxxConstituentPaths = SoxxConstituentPaths(),
    sleep_seconds: float = 0.15,
    holdings_snapshot: SoxxHoldingsSnapshot | None = None,
    price_fetcher: Callable[[str], pd.DataFrame] | None = None,
    symbols: Iterable[str] | None = None,
) -> dict[str, Any]:
    snapshot = holdings_snapshot or fetch_soxx_holdings_snapshot()
    holdings_path = paths.holdings_dir / f"soxx_holdings_{snapshot.as_of}.csv"
    _write_csv(snapshot.rows, holdings_path)

    equity = snapshot.equity_rows
    requested_symbols = [symbol.upper() for symbol in symbols] if symbols is not None else equity["ticker"].tolist()
    equity = equity.loc[equity["ticker"].isin(requested_symbols)].reset_index(drop=True)
    if equity.empty:
        raise ValueError("no SOXX equity constituents selected for history fetch")
    _write_csv(equity, paths.equity_universe_csv)

    fetcher = price_fetcher or (lambda ticker: fetch_yahoo_daily_ohlcv(ticker, start=start, end=end))
    coverage_rows: list[dict[str, Any]] = []
    price_files: dict[str, dict[str, Any]] = {}
    for index, row in equity.iterrows():
        symbol = str(row["ticker"])
        try:
            frame = fetcher(symbol)
            frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)
            if frame.empty:
                raise RuntimeError(f"{symbol} returned no daily rows in requested range")
            path = paths.daily_dir / f"{symbol}.csv"
            _write_csv(frame, path)
            coverage_rows.append(
                _coverage_row(
                    symbol=symbol,
                    name=str(row["name"]),
                    weight_pct=float(row["weight_pct"]) if pd.notna(row["weight_pct"]) else None,
                    frame=frame,
                    status="ok",
                )
            )
            price_files[symbol] = {"path": _manifest_path(path), "sha256": _sha256(path)}
        except Exception as exc:  # pragma: no cover - live provider failures are captured in coverage
            coverage_rows.append(
                _coverage_row(
                    symbol=symbol,
                    name=str(row["name"]),
                    weight_pct=float(row["weight_pct"]) if pd.notna(row["weight_pct"]) else None,
                    frame=None,
                    status="error",
                    error=str(exc),
                )
            )
        if sleep_seconds > 0 and index < len(equity) - 1:
            time.sleep(sleep_seconds)

    coverage = pd.DataFrame(coverage_rows).sort_values(["status", "weight_pct", "symbol"], ascending=[True, False, True])
    _write_csv(coverage, paths.coverage_csv)

    manifest = {
        "dataset": "soxx_current_constituent_daily_history",
        "source": {
            "holdings": "BlackRock iShares SOXX fund download",
            "holdings_url": snapshot.source_url,
            "product_url": SOXX_PRODUCT_URL,
            "prices": "Yahoo Finance chart API",
        },
        "as_of": snapshot.as_of,
        "requested_start": start,
        "requested_end": end,
        "number_of_securities_ex_cash_derivatives": snapshot.number_of_securities,
        "holding_rows": int(len(snapshot.rows)),
        "equity_constituents": int(len(snapshot.equity_rows)),
        "selected_symbols": requested_symbols,
        "successful_symbols": coverage.loc[coverage["status"].eq("ok"), "symbol"].tolist(),
        "error_symbols": coverage.loc[coverage["status"].ne("ok"), "symbol"].tolist(),
        "outputs": {
            "holdings": _manifest_path(holdings_path),
            "equity_universe": _manifest_path(paths.equity_universe_csv),
            "coverage": _manifest_path(paths.coverage_csv),
            "prices": price_files,
        },
        "hashes": {
            "holdings": _sha256(holdings_path),
            "equity_universe": _sha256(paths.equity_universe_csv),
            "coverage": _sha256(paths.coverage_csv),
        },
        "limitations": [
            "Current SOXX constituents backfilled through history are survivorship-biased unless paired with point-in-time holdings snapshots.",
            "Raw CSVs are input snapshots; qlib research must build and audit a qlib provider before training or backtesting.",
        ],
    }
    paths.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch current SOXX constituent holdings and daily OHLCV history.")
    parser.add_argument("--start", default=DEFAULT_HISTORY_START)
    parser.add_argument("--end", default=DEFAULT_HISTORY_END)
    parser.add_argument("--root", type=Path, default=SoxxConstituentPaths().root)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--symbols", default="", help="Optional comma-separated subset for diagnostics.")
    args = parser.parse_args(argv)

    selected_symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()] or None
    manifest = fetch_and_store_soxx_constituent_history(
        start=args.start,
        end=args.end,
        paths=SoxxConstituentPaths(root=args.root),
        sleep_seconds=args.sleep_seconds,
        symbols=selected_symbols,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
