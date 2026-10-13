"""Read original dated Fed G.17 supplement vintages, never revised-backfill."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


INDEX_URL = "https://www.federalreserve.gov/releases/g17/default.htm"
MONTHS = {name: number for number, name in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MONTH_PATTERN = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fetch(url, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    result = subprocess.run(["curl", "--http1.1", "--compressed", "--fail", "--silent", "--show-error",
        "--retry", "2", "--retry-all-errors", "--max-time", "30", url, "-o", str(temporary)], capture_output=True, text=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"needs_data: official source fetch failed ({result.returncode}) for {url}")
    os.replace(temporary, destination)
    return destination.read_bytes()


def parse_capacity_supplement(data: bytes, *, expected_date=None):
    text = data.decode("utf-8-sig")
    header = text[:1500]
    stamp = re.search(r"(" + MONTH_PATTERN + r")\s+(\d{1,2}),?\s+(20\d{2})", header, re.I)
    if not stamp or not re.search(r"9:15\s*a\.?m\.?", header, re.I):
        raise ValueError("needs_data: publication date or declared clock unknown")
    release = pd.Timestamp(f"{stamp[3]}-{MONTHS[stamp[1][:3].lower()]:02}-{int(stamp[2]):02}")
    if expected_date is not None and release != pd.Timestamp(expected_date):
        raise ValueError("blocked: dated source URL and document publication disagree")
    first = re.search(r"(?im)^\s*1?Table\s+2\s*$", text)
    last = re.search(r"(?im)^\s*1?Table\s+3\s*$", text[first.end():]) if first else None
    if not first or not last:
        raise ValueError("needs_data: complete source table2/table3 boundaries absent")
    section = text[first.end():first.end() + last.start()]
    if "Percent of capacity, seasonally adjusted" not in section:
        raise ValueError("blocked: source is not seasonally adjusted utilization")
    lines = section.splitlines()
    boundary = next(i for i, line in enumerate(lines) if re.match(r"\s*-{20,}", line))
    headings = lines[:boundary]
    monthly_heading = " ".join(line.split("|")[-1] for line in headings if "|" in line)
    years = [int(y) for y in re.findall(r"\b20\d{2}\b", monthly_heading)]
    month_tokens = re.findall(r"\b(" + MONTH_PATTERN + r")\.?", monthly_heading, re.I)
    mean_columns = set()
    for line in headings:
        for position, fragment in enumerate(line.split("|")):
            if re.search(r"\b(?:Ave\.?|Average)\b", fragment, re.I):
                mean_columns.add(position)
    if not years or len(month_tokens) < 2 or len(mean_columns) != 1:
        raise ValueError("needs_data: monthly year/month or average column ambiguous")
    row = [line for line in lines[boundary:] if len(line.split("|")) > 2 and line.split("|")[1].strip() in {"3344", "334412-9"}]
    if len(row) != 1:
        raise ValueError("needs_data: exactly one NAICS3344 utilization row required")
    pieces = row[0].split("|")
    code = pieces[1].strip()
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    values = [float(v) for v in re.findall(number_pattern, pieces[-1])]
    average_values = [float(v) for v in re.findall(number_pattern, pieces[next(iter(mean_columns))])]
    if len(values) != len(month_tokens) or not average_values or not np.isfinite(values).all() or not all(0 < v <= 100 for v in [*values, average_values[0]]):
        raise ValueError("needs_data: monthly source values incomplete or outside utilization units")
    months = [MONTHS[m[:3].lower()] for m in month_tokens]
    latest = pd.Timestamp(year=max(years), month=months[-1], day=1)
    periods = pd.period_range(end=latest, periods=len(values), freq="M").to_timestamp()
    if list(periods.month) != months or latest >= release.normalize():
        raise ValueError("blocked: monthly labels are not a valid pre-publication sequence")
    return {"release_date": str(release.date()), "declared_release_at": str((release + pd.Timedelta(hours=9, minutes=15)).tz_localize("America/New_York")),
        "observation_month": str(latest.date()), "capacity_level": values[-1],
        "capacity_change_1m": values[-1] - values[-2], "capacity_gap_to_average": values[-1] - average_values[0],
        "published_average": average_values[0], "published_months": [str(day.date()) for day in periods],
        "published_values": values, "source_naics": code, "source_scope_qualified": code == "3344",
        "source_units": "percent of capacity, seasonally adjusted",
        "raw_sha256": _sha(data), "intraday_first_receipt_verified": False}


def collect_capacity_vintages(*, directory, as_of, start="2010-09-01", maximum_documents=220, workers=4, progress=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "release_index.html"
    index = index_path.read_bytes() if index_path.is_file() else _fetch(INDEX_URL, index_path)
    if b"</html>" not in index.lower():
        raise ValueError("needs_data: source release index is incomplete")
    soup = BeautifulSoup(index, "html.parser")
    links = sorted({urljoin(INDEX_URL, a["href"]) for a in soup.select("a[href]") if re.search(r"(?:\d{8}|Current)/g17_sup\.txt$", a["href"], re.I)})
    links = [url for url in links if "Current/" in url or pd.Timestamp(start) <= pd.Timestamp(re.search(r"/(\d{8})/", url)[1]) <= pd.Timestamp(as_of)]
    if not links or len(links) > maximum_documents:
        raise ValueError("blocked: official source inventory exceeds frozen acquisition budget")
    def acquire(url):
        date = re.search(r"/(\d{8})/", url)
        name = date[1] if date else "Current"
        path = directory / (name + "_sup.txt")
        payload = path.read_bytes() if path.is_file() else _fetch(url, path)
        parsed = parse_capacity_supplement(payload, expected_date=date[1] if date else None)
        if pd.Timestamp(parsed["release_date"]) > pd.Timestamp(as_of):
            raise ValueError("blocked: current source document exceeds fixed as_of")
        return {"path": path.name, "url": url, "bytes": len(payload), "sha256": _sha(payload), "parsed": parsed}
    entries, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(acquire, url): url for url in links}
        for future in as_completed(pending):
            try: entries.append(future.result())
            except Exception as error:
                url = pending[future]
                date = re.search(r"/(\d{8})/", url)
                path = directory / ((date[1] if date else "Current") + "_sup.txt")
                failure = {"url": url, "reason": str(error)}
                if path.is_file():
                    failure.update(path=path.name, sha256=_sha(path.read_bytes()), bytes=path.stat().st_size)
                failures.append(failure)
            if progress: progress({"completed": len(entries) + len(failures), "expected": len(links), "qualified": len(entries), "failed": len(failures)})
    entries.sort(key=lambda e: e["parsed"]["release_date"])
    manifest = {"schema_version": "fed_g17_capacity_vintages_v1", "authority": "Federal Reserve original dated G17 supplement",
        "as_of": as_of, "captured_at": datetime.now(timezone.utc).isoformat(), "index_url": INDEX_URL,
        "index_sha256": _sha(index), "expected_documents": len(links), "entries": entries, "failures": failures,
        "source_complete": not failures, "latest_revised_fred_used": False, "source_is_trading_prices": False}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def capacity_information_state(manifest, calendar):
    """Carry only last released vintage, after the next actual native session."""
    calendar = pd.DatetimeIndex(calendar).normalize()
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("blocked: complete unique Qlib calendar required")
    events = []
    for entry in manifest["entries"]:
        event = dict(entry["parsed"])
        position = calendar.searchsorted(pd.Timestamp(event["release_date"]), side="right")
        event["available_at"] = calendar[position] + pd.Timedelta(hours=9, minutes=31) if position < len(calendar) else pd.NaT
        events.append(event)
    for failure in manifest["failures"]:
        date = re.search(r"/(\d{8})/", failure["url"])
        if date:
            position = calendar.searchsorted(pd.Timestamp(date[1]), side="right")
            events.append({"release_date": str(pd.Timestamp(date[1]).date()), "available_at": calendar[position] + pd.Timedelta(hours=9, minutes=31) if position < len(calendar) else pd.NaT})
    event_frame = pd.DataFrame(events).sort_values("release_date")
    rows = []
    for day in calendar:
        decision = day + pd.Timedelta(hours=9, minutes=46)
        usable = event_frame.loc[event_frame.available_at.notna() & event_frame.available_at.le(decision)]
        row = {"datetime": day, "known": 0., "capacity_level": np.nan, "capacity_change_1m": np.nan,
            "capacity_gap_to_average": np.nan, "available_at": pd.NaT, "release_date": pd.NaT, "observation_month": pd.NaT}
        if len(usable):
            last = usable.iloc[-1]
            row.update(available_at=last.available_at, release_date=pd.Timestamp(last.release_date), observation_month=pd.Timestamp(last.get("observation_month")))
            names = ["capacity_level", "capacity_change_1m", "capacity_gap_to_average"]
            scope = last.get("source_scope_qualified", False)
            if pd.notna(scope) and bool(scope) and all(pd.notna(last.get(name)) for name in names):
                row.update(known=1., **{name: float(last[name]) for name in names})
        rows.append(row)
    return pd.DataFrame(rows), event_frame
