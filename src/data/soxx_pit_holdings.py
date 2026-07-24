from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from .market_update import REPO_ROOT
from .soxx_constituents import (
    PRICE_COLUMNS,
    fetch_yahoo_daily_ohlcv,
)


SEC_SOXX_CIK = "0001100663"
SEC_SOXX_SERIES_ID = "S000004354"
SEC_SOXX_CONTRACT_ID = "C000012084"
SEC_SOXX_NPORT_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK=C000012084&type=NPORT-P"
    "&dateb=&owner=exclude&count=100&output=atom"
)
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_DEFAULT_USER_AGENT = "SOXL qlib research codex@openai.com"
PIT_HOLDINGS_COLUMNS = [
    "report_date",
    "filing_date",
    "accession",
    "series_id",
    "ticker",
    "cusip",
    "isin",
    "name",
    "country",
    "weight_pct",
    "market_value_usd",
]

# N-PORT identifies securities by CUSIP/ISIN rather than exchange ticker.  This
# table covers the complete 2019-present SOXX N-PORT union and preserves legacy
# symbols for securities subsequently acquired or renamed.
SOXX_NPORT_SECURITY_MAP: dict[str, str] = {
    "00215W100": "ASX",
    "007903107": "AMD",
    "032654105": "ADI",
    "038222105": "AMAT",
    "042068205": "ARM",
    "04626A103": "ALAB",
    "054540208": "ACLS",
    "11135F101": "AVGO",
    "114340102": "BRKS",
    "12571T100": "CCMP",
    "12709P103": "CCMP",
    "172755100": "CRUS",
    "225447101": "CREE",
    "29362U104": "ENTG",
    "45772F107": "IPHI",
    "458140100": "INTC",
    "482480100": "KLAC",
    "512807108": "LRCX",
    "512807306": "LRCX",
    "518415104": "LSCC",
    "55306N104": "MKSI",
    "55405Y100": "MTSI",
    "573874104": "MRVL",
    "57772K101": "MXIM",
    "595017104": "MCHP",
    "595112103": "MU",
    "609839105": "MPWR",
    "67066G104": "NVDA",
    "682189105": "ON",
    "683344105": "ONTO",
    "74736K101": "QRVO",
    "747525103": "QCOM",
    "750917106": "RMBS",
    "816850101": "SMTC",
    "826919102": "SLAB",
    "83088M102": "SWKS",
    "861012102": "STM",
    "87157D109": "SYNA",
    "874039100": "TSM",
    "880770102": "TER",
    "882508104": "TXN",
    "910873405": "UMC",
    "91347P105": "OLED",
    "977852102": "WOLF",
    "983919101": "XLNX",
}
SOXX_NPORT_ISIN_MAP: dict[str, str] = {
    "BMG5876H1051": "MRVL",
    "IL0010845571": "NVMI",
    "KYG254571055": "CRDO",
    "NL0009538784": "NXPI",
    "USN070592100": "ASML",
}


@dataclass(frozen=True)
class SecNportFiling:
    accession: str
    filing_date: str
    form: str

    @property
    def primary_xml_url(self) -> str:
        compact = self.accession.replace("-", "")
        return (
            f"{SEC_ARCHIVES_BASE_URL}/{int(SEC_SOXX_CIK)}/"
            f"{compact}/primary_doc.xml"
        )


@dataclass(frozen=True)
class SoxxPitHoldingsPaths:
    root: Path = REPO_ROOT / "data" / "raw" / "soxx_constituents"

    @property
    def history_csv(self) -> Path:
        return self.root / "pit_holdings_history.csv"

    @property
    def manifest_json(self) -> Path:
        return self.root / "pit_holdings_manifest.json"

    @property
    def daily_dir(self) -> Path:
        return self.root / "daily"

    @property
    def price_coverage_csv(self) -> Path:
        return self.root / "pit_price_coverage.csv"

    @property
    def price_manifest_json(self) -> Path:
        return self.root / "pit_price_manifest.json"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(node: ET.Element, name: str) -> str | None:
    child = next(
        (item for item in node.iter() if _local_name(item.tag) == name),
        None,
    )
    if child is None:
        return None
    return child.text


def _parse_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def parse_sec_soxx_nport_atom(atom_xml: str | bytes) -> list[SecNportFiling]:
    root = ET.fromstring(atom_xml)
    filings: list[SecNportFiling] = []
    for entry in (
        item for item in root.iter() if _local_name(item.tag) == "entry"
    ):
        values = {
            _local_name(item.tag): item.text
            for item in entry.iter()
            if item.text
        }
        accession = values.get("accession-number")
        filing_date = values.get("filing-date")
        form = values.get("filing-type")
        if not accession or not filing_date or not form:
            continue
        if form not in {"NPORT-P", "NPORT-P/A"}:
            continue
        filings.append(
            SecNportFiling(
                accession=accession,
                filing_date=pd.Timestamp(filing_date).date().isoformat(),
                form=form,
            )
        )
    if not filings:
        raise ValueError("SEC SOXX N-PORT feed produced no filings")
    return sorted(
        filings,
        key=lambda item: (item.filing_date, item.accession),
    )


def parse_sec_soxx_nport_primary_xml(
    primary_xml: str | bytes,
    *,
    filing: SecNportFiling,
) -> pd.DataFrame:
    root = ET.fromstring(primary_xml)
    series_id = next(
        (
            item.text
            for item in root.iter()
            if _local_name(item.tag) == "seriesId"
        ),
        None,
    )
    if series_id != SEC_SOXX_SERIES_ID:
        raise ValueError(
            f"SEC N-PORT series mismatch: expected {SEC_SOXX_SERIES_ID}, "
            f"got {series_id}"
        )
    report_date = next(
        (
            item.text
            for item in root.iter()
            if _local_name(item.tag) == "repPdDate"
        ),
        None,
    )
    if not report_date:
        raise ValueError("SEC SOXX N-PORT missing report date")

    rows: list[dict[str, Any]] = []
    for security in (
        item
        for item in root.iter()
        if _local_name(item.tag) == "invstOrSec"
    ):
        if _child_text(security, "assetCat") != "EC":
            continue
        cusip = (_child_text(security, "cusip") or "").strip().upper()
        isin_node = next(
            (
                item
                for item in security.iter()
                if _local_name(item.tag) == "isin"
            ),
            None,
        )
        isin = (
            isin_node.attrib.get("value", "").strip().upper()
            if isin_node is not None
            else ""
        )
        ticker = SOXX_NPORT_SECURITY_MAP.get(
            cusip,
            SOXX_NPORT_ISIN_MAP.get(isin),
        )
        if ticker is None:
            raise ValueError(
                "SEC SOXX N-PORT equity has unresolved identifier "
                f"CUSIP={cusip!r}, ISIN={isin!r} "
                f"({_child_text(security, 'name')!r})"
            )
        rows.append(
            {
                "report_date": pd.Timestamp(report_date)
                .date()
                .isoformat(),
                "filing_date": filing.filing_date,
                "accession": filing.accession,
                "series_id": series_id,
                "ticker": ticker,
                "cusip": cusip,
                "isin": isin,
                "name": (_child_text(security, "name") or "").strip(),
                "country": (
                    _child_text(security, "invCountry") or ""
                ).strip(),
                "weight_pct": _parse_float(
                    _child_text(security, "pctVal")
                ),
                "market_value_usd": _parse_float(
                    _child_text(security, "valUSD")
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=PIT_HOLDINGS_COLUMNS)
    if frame.empty:
        raise ValueError("SEC SOXX N-PORT produced no equity holdings")
    if frame["ticker"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["ticker"].duplicated(False), "ticker"].unique()
        )
        raise ValueError(
            f"SEC SOXX N-PORT produced duplicate tickers {duplicates}"
        )
    if frame["weight_pct"].isna().any():
        raise ValueError("SEC SOXX N-PORT equity weight is missing")
    return frame.sort_values(
        ["weight_pct", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)


def fetch_sec_soxx_nport_history(
    *,
    timeout: float = 30.0,
    sleep_seconds: float = 0.11,
    request_get: Callable[..., requests.Response] = requests.get,
    user_agent: str = SEC_DEFAULT_USER_AGENT,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    headers = {"User-Agent": user_agent, "Accept": "application/xml,text/xml"}
    response = request_get(
        SEC_SOXX_NPORT_ATOM_URL,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    filings = parse_sec_soxx_nport_atom(response.content)

    snapshots: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for index, filing in enumerate(filings):
        filing_response = request_get(
            filing.primary_xml_url,
            headers=headers,
            timeout=timeout,
        )
        filing_response.raise_for_status()
        snapshot = parse_sec_soxx_nport_primary_xml(
            filing_response.content,
            filing=filing,
        )
        snapshots.append(snapshot)
        source_rows.append(
            {
                "accession": filing.accession,
                "filing_date": filing.filing_date,
                "form": filing.form,
                "report_date": str(snapshot["report_date"].iloc[0]),
                "equity_count": int(len(snapshot)),
                "equity_weight_pct": float(snapshot["weight_pct"].sum()),
                "primary_xml_url": filing.primary_xml_url,
            }
        )
        if sleep_seconds > 0 and index < len(filings) - 1:
            time.sleep(sleep_seconds)

    history = (
        pd.concat(snapshots, ignore_index=True)
        .sort_values(["filing_date", "accession", "weight_pct", "ticker"])
        .reset_index(drop=True)
    )
    if history.duplicated(["accession", "ticker"]).any():
        raise ValueError("SEC SOXX N-PORT history contains duplicate rows")
    return history, source_rows


def build_pit_soxx_weight_panel(
    history: pd.DataFrame,
    *,
    calendar: Iterable[pd.Timestamp | str],
    availability_lag_days: int = 0,
) -> pd.DataFrame:
    missing = sorted(set(PIT_HOLDINGS_COLUMNS) - set(history.columns))
    if missing:
        raise ValueError(f"SOXX PIT holdings history missing columns {missing}")
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must be non-negative")

    trading_days = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize()
    trading_days = trading_days.sort_values().unique()
    if trading_days.empty:
        raise ValueError("SOXX PIT weight panel requires a calendar")

    snapshots = history.copy()
    snapshots["filing_date"] = pd.to_datetime(
        snapshots["filing_date"]
    ).dt.normalize()
    snapshots["available_date"] = snapshots["filing_date"] + pd.to_timedelta(
        availability_lag_days,
        unit="D",
    )
    snapshots["report_date"] = pd.to_datetime(
        snapshots["report_date"]
    ).dt.normalize()
    snapshots = snapshots.sort_values(
        ["available_date", "filing_date", "accession"]
    )

    rows: list[pd.DataFrame] = []
    filing_dates = sorted(snapshots["available_date"].unique())
    for index, available_date in enumerate(filing_dates):
        version = snapshots.loc[
            snapshots["available_date"].eq(available_date)
        ]
        # Multiple filings on one day are amendments or duplicate versions.
        # The lexicographically latest accession is the public state at close.
        accession = str(version["accession"].max())
        version = version.loc[version["accession"].eq(accession)].copy()
        next_available = (
            pd.Timestamp(filing_dates[index + 1])
            if index + 1 < len(filing_dates)
            else None
        )
        valid_dates = trading_days[trading_days >= pd.Timestamp(available_date)]
        if next_available is not None:
            valid_dates = valid_dates[valid_dates < next_available]
        if valid_dates.empty:
            continue
        repeated = version.loc[
            version.index.repeat(len(valid_dates))
        ].reset_index(drop=True)
        repeated["datetime"] = list(valid_dates) * len(version)
        rows.append(
            repeated[
                [
                    "datetime",
                    "ticker",
                    "weight_pct",
                    "report_date",
                    "filing_date",
                    "accession",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "datetime",
                "ticker",
                "weight_pct",
                "report_date",
                "filing_date",
                "accession",
            ]
        )
    panel = (
        pd.concat(rows, ignore_index=True)
        .rename(columns={"ticker": "instrument"})
        .sort_values(["datetime", "instrument"])
        .reset_index(drop=True)
    )
    if panel.duplicated(["datetime", "instrument"]).any():
        raise ValueError("SOXX PIT weight panel contains duplicate rows")
    if (
        pd.to_datetime(panel["datetime"])
        < pd.to_datetime(panel["filing_date"])
    ).any():
        raise ValueError("SOXX PIT weight panel used holdings before filing")
    return panel


def fetch_and_store_sec_soxx_nport_history(
    *,
    paths: SoxxPitHoldingsPaths = SoxxPitHoldingsPaths(),
    timeout: float = 30.0,
    sleep_seconds: float = 0.11,
    request_get: Callable[..., requests.Response] = requests.get,
    user_agent: str = SEC_DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    history, sources = fetch_sec_soxx_nport_history(
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        request_get=request_get,
        user_agent=user_agent,
    )
    paths.root.mkdir(parents=True, exist_ok=True)
    history.to_csv(paths.history_csv, index=False)
    digest = hashlib.sha256(paths.history_csv.read_bytes()).hexdigest()
    manifest = {
        "dataset": "soxx_sec_nport_point_in_time_holdings",
        "status": "ok",
        "source": "SEC EDGAR NPORT-P",
        "source_feed": SEC_SOXX_NPORT_ATOM_URL,
        "cik": SEC_SOXX_CIK,
        "series_id": SEC_SOXX_SERIES_ID,
        "contract_id": SEC_SOXX_CONTRACT_ID,
        "snapshot_count": int(
            history[["filing_date", "accession"]]
            .drop_duplicates()
            .shape[0]
        ),
        "report_start": str(history["report_date"].min()),
        "report_end": str(history["report_date"].max()),
        "filing_start": str(history["filing_date"].min()),
        "filing_end": str(history["filing_date"].max()),
        "instrument_count": int(history["ticker"].nunique()),
        "row_count": int(len(history)),
        "output": paths.history_csv.resolve().as_posix(),
        "sha256": digest,
        "availability_policy": (
            "holdings become model-eligible on or after SEC filing_date; "
            "report_date is never used as availability date"
        ),
        "limitations": [
            "SEC N-PORT coverage begins in 2019 and is periodic rather than daily.",
            "Delisted and renamed securities require legacy ticker price coverage.",
            "Historical holdings are training information only, not a tradable constituent rotation strategy.",
        ],
        "filings": sources,
    }
    paths.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_sec_soxx_nport_history(
    *, paths: SoxxPitHoldingsPaths = SoxxPitHoldingsPaths()
) -> pd.DataFrame:
    if not paths.history_csv.exists():
        raise FileNotFoundError(
            f"SOXX PIT holdings history missing: {paths.history_csv}"
        )
    history = pd.read_csv(paths.history_csv)
    missing = sorted(set(PIT_HOLDINGS_COLUMNS) - set(history.columns))
    if missing:
        raise ValueError(f"SOXX PIT holdings history missing columns {missing}")
    return history


def fetch_and_store_soxx_pit_price_history(
    *,
    paths: SoxxPitHoldingsPaths = SoxxPitHoldingsPaths(),
    start: str = "2010-03-11",
    end: str = "2026-07-02",
    sleep_seconds: float = 0.15,
    reuse_existing: bool = True,
    price_fetcher: Callable[[str], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    history = load_sec_soxx_nport_history(paths=paths)
    fetcher = price_fetcher or (
        lambda ticker: fetch_yahoo_daily_ohlcv(
            ticker,
            start=start,
            end=end,
        )
    )
    paths.daily_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    price_hashes: dict[str, str] = {}
    symbols = sorted(history["ticker"].astype(str).str.upper().unique())
    membership = (
        history.groupby("ticker", sort=True)
        .agg(
            first_report_date=("report_date", "min"),
            last_report_date=("report_date", "max"),
            snapshot_count=("accession", "nunique"),
        )
        .reset_index()
        .set_index("ticker")
    )
    for index, symbol in enumerate(symbols):
        path = paths.daily_dir / f"{symbol}.csv"
        status = "ok"
        error = ""
        try:
            if reuse_existing and path.exists():
                frame = pd.read_csv(path)
            else:
                frame = fetcher(symbol)
                frame = frame.loc[
                    frame["date"].astype(str).between(start, end)
                ].reset_index(drop=True)
                if frame.empty:
                    raise RuntimeError(
                        f"{symbol} returned no daily rows in requested range"
                    )
                frame.to_csv(path, index=False)
            missing = sorted(set(PRICE_COLUMNS) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"{symbol} daily history missing columns {missing}"
                )
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            first_price = frame["date"].min()
            last_price = frame["date"].max()
            first_member = pd.Timestamp(
                membership.loc[symbol, "first_report_date"]
            )
            last_member = pd.Timestamp(
                membership.loc[symbol, "last_report_date"]
            )
            overlap_start = max(first_price, first_member)
            overlap_end = min(last_price, last_member)
            if overlap_start > overlap_end:
                status = "insufficient_membership_overlap"
            price_hashes[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = int(len(frame))
            start_date = first_price.date().isoformat()
            end_date = last_price.date().isoformat()
        except Exception as exc:  # pragma: no cover - live provider failures
            status = "error"
            error = str(exc)
            rows = 0
            start_date = ""
            end_date = ""
        coverage_rows.append(
            {
                "symbol": symbol,
                "status": status,
                "rows": rows,
                "start_date": start_date,
                "end_date": end_date,
                "first_report_date": str(
                    membership.loc[symbol, "first_report_date"]
                ),
                "last_report_date": str(
                    membership.loc[symbol, "last_report_date"]
                ),
                "snapshot_count": int(
                    membership.loc[symbol, "snapshot_count"]
                ),
                "error": error,
            }
        )
        if sleep_seconds > 0 and index < len(symbols) - 1:
            time.sleep(sleep_seconds)

    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["status", "symbol"]
    )
    coverage.to_csv(paths.price_coverage_csv, index=False)
    usable = set(
        coverage.loc[coverage["status"].eq("ok"), "symbol"]
        .astype(str)
        .str.upper()
    )
    snapshot_coverage = (
        history.assign(
            usable=history["ticker"].astype(str).str.upper().isin(usable)
        )
        .groupby(["filing_date", "accession"], sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "usable_instruments": int(group["usable"].sum()),
                    "total_instruments": int(len(group)),
                    "usable_weight_pct": float(
                        group.loc[group["usable"], "weight_pct"].sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    manifest = {
        "dataset": "soxx_sec_nport_constituent_price_history",
        "status": "ok" if usable else "blocked_no_usable_prices",
        "source_holdings": paths.history_csv.resolve().as_posix(),
        "source_prices": (
            "existing canonical constituent CSVs plus Yahoo Finance chart API"
        ),
        "requested_start": start,
        "requested_end": end,
        "requested_instruments": len(symbols),
        "usable_instruments": len(usable),
        "error_instruments": sorted(
            coverage.loc[
                coverage["status"].ne("ok"), "symbol"
            ].tolist()
        ),
        "snapshot_usable_weight_pct_min": float(
            snapshot_coverage["usable_weight_pct"].min()
        ),
        "snapshot_usable_weight_pct_median": float(
            snapshot_coverage["usable_weight_pct"].median()
        ),
        "snapshot_usable_weight_pct_max": float(
            snapshot_coverage["usable_weight_pct"].max()
        ),
        "coverage_output": paths.price_coverage_csv.resolve().as_posix(),
        "price_hashes": price_hashes,
        "limitations": [
            "Yahoo does not retain every delisted ticker; unavailable historical members are excluded and weight coverage is reported.",
            "A constituent snapshot is eligible only after its SEC filing date.",
            "Single-stock 2x paths used by models must be derived and labeled synthetic unless an actual ETF history is present.",
        ],
    }
    paths.price_manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def validate_soxx_nport_security_map(
    history: pd.DataFrame,
    *,
    mapping: dict[str, str] = SOXX_NPORT_SECURITY_MAP,
) -> list[str]:
    cusips = {
        str(item).strip().upper()
        for item in history["cusip"].dropna().tolist()
    }
    return sorted(cusips - set(mapping))


def sec_soxx_nport_urls(
    filings: Sequence[SecNportFiling],
) -> list[str]:
    return [filing.primary_xml_url for filing in filings]


def normalized_security_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())
