"""TSMC published monthly-revenue PIT information data surface (no alpha, no models).

Collects as-filed SEC 6-K monthly revenue reports for CIK 0001046179, parses the
main monthly-revenue table (or the older press-release paragraph), binds each
period's FIRST as-filed version to a conservative availability session (the first
US session strictly AFTER the SEC acceptance date), and builds a qlib information
store (instrument ``TSMC_RELEASE``).

Zero models; the single factor is ``reported_yoy_acceleration_pct_points`` and it is
only defined when the current and prior consecutive periods are both qualified AND
the prior original source is already available at the current factor clock. Missing
or unsupported periods are ``needs_data`` and are never 0-filled.
"""
from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

CIK = "0001046179"
ISSUER = "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD"
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
ARCHIVE = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}"
RAW_DIR = "data/raw/external/tsmc_monthly_revenue_20261010_v1"
PROVIDER_URI = "data/qlib/tsmc_monthly_revenue_20261010_v1"
INSTRUMENT = "TSMC_RELEASE"
PERIOD_START = "2015-11"
PERIOD_END = "2026-08"
SOURCE_AS_OF = "2026-10-10"
USABLE_PRICE_CUTOFF = "2026-09-25"
FIELDS = ("reported_yoy_pct", "reported_mom_pct", "revenue_ntd_million",
          "revenue_precision_flag", "acceleration_pct_points", "event_available")
MONTHS = ("January February March April May June July August September October November December").split()
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
UA = "soxl-research-qlib/1.0 (public-filings research; SEC-compliant access)"


def expected_periods(start: str = PERIOD_START, end: str = PERIOD_END) -> list[str]:
    y, m = int(start[:4]), int(start[5:7])
    y1, m1 = int(end[:4]), int(end[5:7])
    out = []
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _next_period(period: str) -> str:
    y, m = int(period[:4]), int(period[5:7])
    m += 1
    if m == 13:
        y, m = y + 1, 1
    return f"{y:04d}-{m:02d}"


def _sha(data) -> str:
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- parsing
def _flat(raw: bytes) -> tuple[str, str]:
    encoding = "identity"
    if raw[:2] == b"\x1f\x8b":
        encoding = "gzip"
        raw = gzip.decompress(raw)
    text = html.unescape(raw.decode("utf-8", "replace"))
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text), encoding


def to_ntd_million(value, unit: str | None):
    if value is None or unit is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    u = unit.lower()
    if "million" in u:
        return float(value)
    if "billion" in u:
        return float(value) * 1000.0
    if "thousand" in u:
        return float(value) / 1000.0
    return None


def _num(token: str) -> float:
    return (-1 if token.strip().startswith("(") else 1) * float(token.strip().strip("()").replace(",", ""))


def parse_monthly_revenue(raw: bytes) -> dict | None:
    text, _ = _flat(raw)
    if not re.search(r"Revenue Report|Monthly Revenue|net revenues? for [A-Z][a-z]+ \d{4}", text):
        return None
    table = re.search(r"\(Unit\s*:\s*NT\$\s*([A-Za-z ]+?)\)(.*?)\bNet Revenue\b((?:\s+\(?-?[\d,.]+\)?){4,})", text)
    if table and re.search(r"Revenue Report|Monthly Revenue", text):
        values = [_num(x) for x in re.findall(r"\(?-?[\d,.]+\)?", table.group(3))]
        if len(values) >= 5:
            cur, prev, mom, same_prev_year, yoy = values[:5]
            header = table.group(2).strip()
            mk = re.search(r"Period\s+([A-Z][a-z]+)\s+(\d{4})", header)
            period = f"{int(mk.group(2)):04d}-{MONTH_NUM[mk.group(1)]:02d}" if mk else None
            return {"format": "main_table", "period": period, "unit": "NT$ " + table.group(1).strip(),
                    "current_ntd": cur, "prev_month_ntd": prev, "same_month_prev_year_ntd": same_prev_year,
                    "reported_mom": mom, "reported_yoy": yoy,
                    "calc_mom": round((cur / prev - 1) * 100, 2), "calc_yoy": round((cur / same_prev_year - 1) * 100, 2),
                    "header_span": table.group(2).strip()[:220], "row_span": table.group(0)[:320],
                    "unit_bound": "main_monthly_table"}
    para = re.search(r"revenues? for ([A-Z][a-z]+)\s+(\d{4}).{0,60}?approximately NT\$([\d.]+) (billion|million)", text)
    if not para:
        return None
    month, value, scale = para.group(1), float(para.group(3)), para.group(4)
    year = int(para.group(2))
    seg = text[para.start():para.start() + 430]
    clauses = re.findall(r"a[n]? (increase|decrease)(?: of)? ([\d.]+) percent (?:from|versus) ([A-Z][a-z]+)\s+(\d{4})", seg)
    def val(c):
        return float(("-" if c[0] == "decrease" else "") + c[1])
    return {"format": "paragraph", "unit": f"NT$ {scale}", "period": f"{year:04d}-{MONTH_NUM[month]:02d}",
            "current_approx": value, "scale": scale,
            "reported_mom": val(clauses[0]) if len(clauses) >= 1 else np.nan,
            "reported_yoy": val(clauses[1]) if len(clauses) >= 2 else np.nan,
            "mom_ref": f"{clauses[0][2]} {clauses[0][3]}" if len(clauses) >= 1 else None,
            "yoy_ref": f"{clauses[1][2]} {clauses[1][3]}" if len(clauses) >= 2 else None,
            "paragraph_span": seg[:360], "unit_bound": "press_release_paragraph"}


def table_numeric_gate(record: dict, *, tol: float = 0.15) -> bool:
    if record.get("format") == "main_table":
        return (abs(record["calc_mom"] - record["reported_mom"]) <= tol
                and abs(record["calc_yoy"] - record["reported_yoy"]) <= tol)
    return bool(record.get("reported_mom") is not None and record.get("reported_yoy") is not None
                and np.isfinite(record.get("reported_yoy", np.nan)))


# ---------------------------------------------------------------- clock
def next_us_open(after_utc, calendar_naive_et: pd.DatetimeIndex):
    acc = pd.Timestamp(after_utc)
    if acc.tz is None:
        raise ValueError("blocked: accepted time must be timezone-aware")
    et_day = acc.tz_convert("America/New_York").tz_localize(None).normalize()
    days = pd.DatetimeIndex(calendar_naive_et).normalize()
    later = sorted(d for d in days if d > et_day)
    return later[0] if later else None


def us_session_calendar(root: Path) -> tuple[pd.DatetimeIndex, dict]:
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str((root / "data/qlib/soxl").resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day")).normalize()
    frozen = {"uri": "data/qlib/soxl", "n_sessions": int(len(calendar)),
              "first": str(calendar[0].date()), "last": str(calendar[-1].date()),
              "sha256": hashlib.sha256("\n".join(d.strftime("%Y-%m-%d") for d in calendar).encode()).hexdigest()}
    return calendar, frozen


# ---------------------------------------------------------------- collection
def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "identity"})
    return s


def fetch_submissions(session, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "submissions.json"
    if not path.exists():
        r = session.get(SUBMISSIONS_URL, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
        time.sleep(1.0)
    return json.loads(path.read_text())


def enumerate_6k(submissions: dict) -> list[dict]:
    rec = submissions["filings"]["recent"]
    rows = [{"form": f, "filingDate": fd, "acceptanceDateTime": ad, "accessionNumber": a, "primaryDocument": d}
            for f, fd, ad, a, d in zip(rec["form"], rec["filingDate"], rec["acceptanceDateTime"],
                                       rec["accessionNumber"], rec["primaryDocument"])]
    return [r for r in rows if r["form"] in ("6-K", "6-K/A")]


def fetch_document(session, record: dict, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    accn, doc = record["accessionNumber"], record["primaryDocument"]
    path = dest / f"{accn}_{doc}"
    header_path = dest / f"{accn}_{doc}.headers.json"
    url = f"{ARCHIVE}/{accn.replace('-', '')}/{doc}"
    if not path.exists():
        r = session.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
        header_path.write_text(json.dumps({"url": url, "status": r.status_code,
                                           "headers": {k.lower(): v for k, v in r.headers.items()}}, indent=2))
        time.sleep(1.0)
    return {"path": str(path), "url": url, "raw_sha256": _sha(path.read_bytes())}


# ---------------------------------------------------------------- events
def build_events(records: list[dict], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per period (first as-filed version) with the PIT acceleration clock."""
    first: dict[str, dict] = {}
    revisions: dict[str, list] = {}
    for rec in records:
        parsed = rec.get("parsed")
        period = rec.get("period") or (parsed or {}).get("period")
        if not period or not parsed:
            continue
        entry = {"period": period, "accn": rec["accessionNumber"], "filingDate": rec["filingDate"],
                 "accepted_utc": rec["acceptanceDateTime"], "url": rec["url"], "raw_sha256": rec["raw_sha256"],
                 **{k: parsed.get(k) for k in ("format", "unit", "reported_mom", "reported_yoy",
                                               "current_ntd", "current_approx", "scale", "calc_mom", "calc_yoy")}}
        entry["revenue_ntd_million"] = to_ntd_million(
            entry["current_ntd"] if parsed["format"] == "main_table" else entry["current_approx"], entry["unit"])
        entry["precision_flag"] = 1.0 if parsed["format"] == "main_table" else 0.0
        prior = first.get(period)
        if prior is None or rec["acceptanceDateTime"] < prior["accepted_utc"]:
            if prior is not None:
                revisions.setdefault(period, []).append(prior)
            first[period] = entry
        else:
            revisions.setdefault(period, []).append(entry)

    rows = []
    for period in sorted(first):
        e = first[period]
        acc = pd.Timestamp(e["accepted_utc"])
        eligible = next_us_open(acc, calendar)
        rows.append({"period": period, "accn": e["accn"], "url": e["url"], "raw_sha256": e["raw_sha256"],
                     "accepted_utc": acc.isoformat(), "format": e["format"], "unit": e["unit"],
                     "reported_mom_pct": e.get("reported_mom"), "reported_yoy_pct": e.get("reported_yoy"),
                     "revenue_ntd_million": e["revenue_ntd_million"], "revenue_precision_flag": e["precision_flag"],
                     "eligible_session": (eligible.strftime("%Y-%m-%d") if eligible is not None else None),
                     "revisions": len(revisions.get(period, [])),
                     "numeric_gate_pass": table_numeric_gate(e)})
    frame = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    frame["admitted_to_scope"] = frame["period"].isin(set(expected_periods()))

    # Acceleration with an availability clock: prior source must be available at the
    # current factor's clock; the factor becomes usable at the LATER of the two clocks.
    by_period = frame.set_index("period").to_dict("index")
    accel, factor_avail, pair_accn = [], [], []
    for period in frame["period"]:
        value = np.nan
        prior_period = _prev_period(period)
        fclock = None
        pair = None
        if prior_period in by_period:
            cur, pri = by_period[period], by_period[prior_period]
            if cur["numeric_gate_pass"] and pri["numeric_gate_pass"] and _next_period(prior_period) == period:
                cur_clock = pd.Timestamp(cur["accepted_utc"])
                pri_clock = pd.Timestamp(pri["accepted_utc"])
                fclock = max(cur_clock, pri_clock)
                value = float(cur["reported_yoy_pct"] - pri["reported_yoy_pct"])
                pair = [pri["accn"], cur["accn"]]
        accel.append(value); factor_avail.append(fclock); pair_accn.append(pair)
    frame["acceleration_pct_points"] = accel
    frame["factor_clock_utc"] = [c.isoformat() if c is not None else None for c in factor_avail]
    frame["factor_source_accns"] = pair_accn
    frame["factor_eligible_session"] = [
        (next_us_open(c, calendar).strftime("%Y-%m-%d") if (c is not None and next_us_open(c, calendar) is not None) else None)
        for c in factor_avail]
    return frame


def _prev_period(period: str) -> str:
    y, m = int(period[:4]), int(period[5:7])
    m -= 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def build_events_from_raw(root: Path, calendar: pd.DatetimeIndex):
    """Re-parse the approved raw docs (source of truth) into events + a docs manifest.

    Values come from the actual document bytes, not a cached parquet; each doc's
    SHA/URL/accepted/period are bound so the workflow can fail closed on any drift.
    """
    raw = root / RAW_DIR
    docs = raw / "docs"
    submissions = json.loads((raw / "submissions.json").read_text())
    rec = submissions["filings"]["recent"]
    meta = {a: (fd, ad, prim) for (fd, ad, a, prim) in zip(rec["filingDate"], rec["acceptanceDateTime"],
                                                          rec["accessionNumber"], rec["primaryDocument"])}
    records, docs_manifest = [], []
    for path in sorted(docs.glob("*.htm")):
        accn = path.name.split("_")[0]
        body = path.read_bytes()
        parsed = parse_monthly_revenue(body)
        if not parsed:
            continue
        fd, ad, prim = meta.get(accn, (None, None, None))
        if ad is None:
            raise ValueError(f"blocked: {accn} not found in official submissions metadata")
        url = f"{ARCHIVE}/{accn.replace('-', '')}/{prim}"
        records.append({"accessionNumber": accn, "filingDate": fd, "acceptanceDateTime": ad, "url": url,
                        "raw_sha256": _sha(body), "parsed": parsed, "period": parsed.get("period")})
        docs_manifest.append({"accn": accn, "period": parsed.get("period"), "sha256": _sha(body),
                              "accepted_utc": ad, "url": url, "unit": parsed.get("unit"),
                              "format": parsed.get("format")})
    # Validate ALL admitted versions (including revisions) before first-version selection.
    assert_source_asof_records(records)
    frame = build_events(records, calendar)
    (raw / "docs_manifest.json").write_text(json.dumps(docs_manifest, indent=2, ensure_ascii=False))
    return frame, docs_manifest, records


def coverage(frame: pd.DataFrame, *, start: str = PERIOD_START, end: str = PERIOD_END) -> dict:
    expected = expected_periods(start, end)
    present = set(frame["period"]) if not frame.empty else set()
    missing = [p for p in expected if p not in present]
    extra = sorted(p for p in present if p not in set(expected))
    scope = frame[frame["period"].isin(expected)] if not frame.empty else frame
    unqualified = [r["period"] for _, r in scope.iterrows() if not bool(r["numeric_gate_pass"])] if not scope.empty else []
    return {"expected_periods": expected, "expected": len(expected), "present": len(present & set(expected)),
            "not_admitted_to_scope": extra, "missing": missing,
            "numeric_gate_failures": unqualified, "complete": not missing and not unqualified}


# ---------------------------------------------------------------- qlib store
def assert_source_asof_records(records: list[dict], *, source_as_of: str = SOURCE_AS_OF) -> None:
    """Every admitted source record (including a future revision) must precede source_as_of."""
    boundary = pd.Timestamp(source_as_of, tz="UTC")
    for r in records:
        acc = pd.Timestamp(r["acceptanceDateTime"])
        if acc.tz is None:
            raise ValueError(f"blocked: {r.get('accessionNumber')} acceptance is not timezone-aware")
        if acc > boundary:
            raise ValueError(f"blocked: {r.get('accessionNumber')} accepted {acc.isoformat()} beyond {source_as_of}")


def assert_source_asof(frame: pd.DataFrame, *, source_as_of: str = SOURCE_AS_OF) -> None:
    boundary = pd.Timestamp(source_as_of, tz="UTC")
    for _, r in frame.iterrows():
        acc = pd.Timestamp(r["accepted_utc"])
        if acc.tz is None:
            raise ValueError(f"blocked: {r['period']} acceptance is not timezone-aware")
        if acc > boundary:
            raise ValueError(f"blocked: {r['period']} accepted at {acc.isoformat()} beyond source_as_of {source_as_of}")


def reference_asof_state(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Expected daily asof state (for readback verification). No fitting."""
    events = frame[frame["period"].isin(set(expected_periods()))].dropna(subset=["eligible_session"]).copy()
    events["elig"] = pd.to_datetime(events["eligible_session"]).dt.normalize()
    cal = pd.DatetimeIndex(calendar).normalize()
    cal = cal[(cal >= events["elig"].min()) & (cal <= pd.Timestamp(USABLE_PRICE_CUTOFF))]
    by_period = {r["period"]: r for _, r in events.iterrows()}
    rows = []
    for day in cal:
        asof = {p: r for p, r in by_period.items() if r["elig"] <= day}
        state = {f: np.nan for f in FIELDS}
        state["event_available"] = 0.0
        if asof:
            latest_period = max(asof)
            latest = asof[latest_period]
            state["reported_yoy_pct"] = float(latest["reported_yoy_pct"])
            state["reported_mom_pct"] = float(latest["reported_mom_pct"])
            state["revenue_ntd_million"] = float(latest["revenue_ntd_million"])
            state["revenue_precision_flag"] = float(latest["revenue_precision_flag"])
            if day in set(events["elig"]):
                state["event_available"] = 1.0
            prev = _prev_period(latest_period)
            if (prev in asof and _next_period(prev) == latest_period
                    and bool(latest["numeric_gate_pass"]) and bool(asof[prev]["numeric_gate_pass"])
                    and np.isfinite(latest["reported_yoy_pct"]) and np.isfinite(asof[prev]["reported_yoy_pct"])):
                state["acceleration_pct_points"] = float(latest["reported_yoy_pct"] - asof[prev]["reported_yoy_pct"])
        rows.append({"datetime": day, **state})
    return pd.DataFrame(rows)


def build_qlib_store(root: Path, frame: pd.DataFrame, calendar: pd.DatetimeIndex, *,
                     provider_uri: str | Path | None = None) -> dict:
    from src.data.qlib_native import build_qlib_store_from_frame

    assert_source_asof(frame)
    store = reference_asof_state(frame, calendar)
    frame_out = pd.DataFrame({"instrument": INSTRUMENT, "datetime": store["datetime"]})
    for field in FIELDS:
        frame_out[field] = store[field].astype(float).to_numpy()
    uri = Path(provider_uri) if provider_uri is not None else (root / PROVIDER_URI)
    manifest = build_qlib_store_from_frame(frame_out, provider_uri=uri, freq="day", as_of=SOURCE_AS_OF,
                                           source="tsmc_monthly_revenue_20261010_v1",
                                           field_map={f: f for f in FIELDS})
    manifest.update({"instrument": INSTRUMENT, "information_instrument": True, "not_trading_prices": True,
                     "factor": "reported_yoy_acceleration_pct_points", "source_as_of": SOURCE_AS_OF,
                     "usable_price_cutoff": USABLE_PRICE_CUTOFF,
                     "calendar_slice": [str(store["datetime"].min().date()), str(store["datetime"].max().date())],
                     "revision_contract": "first_as_filed_only (original first values immutable; no latest-asof revision feature)",
                     "carry_policy": "daily asof state: latest known financial period with eligible_session<=day; factor only when prior period also known and eligible"})
    (uri / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def readback(root: Path, *, provider_uri: str | Path | None = None) -> pd.DataFrame:
    import qlib
    from qlib.data import D
    uri = Path(provider_uri) if provider_uri is not None else (root / PROVIDER_URI)
    qlib.init(provider_uri=str(uri), region="us", kernels=1)
    return D.features([INSTRUMENT], [f"${f}" for f in FIELDS], freq="day").reset_index()


PARSER_ORACLE = {
    "2015-12": {"reported_yoy": -16.1, "reported_mom": -8.0, "format": "paragraph"},
    "2016-01": {"reported_yoy": -18.7, "reported_mom": 21.4, "format": "paragraph"},
    "2016-02": {"reported_yoy": -4.9, "reported_mom": -16.0, "format": "paragraph"},
    "2019-12": {"reported_yoy": 15.0, "reported_mom": -4.2, "format": "paragraph"},
    "2020-01": {"reported_yoy": 32.8, "reported_mom": 0.4, "format": "paragraph"},
    "2020-02": {"reported_yoy": 53.4, "reported_mom": -9.9, "format": "paragraph"},
    "2023-12": {"reported_yoy": -8.4, "reported_mom": -14.4, "format": "main_table"},
    "2024-01": {"reported_yoy": 7.9, "reported_mom": 22.4, "format": "main_table"},
    "2024-02": {"reported_yoy": 11.3, "reported_mom": -15.8, "format": "main_table"},
    "2024-12": {"reported_yoy": 57.8, "reported_mom": 0.8, "format": "main_table"},
    "2025-01": {"reported_yoy": 35.9, "reported_mom": 5.4, "format": "main_table"},
    "2025-02": {"reported_yoy": 43.1, "reported_mom": -11.3, "format": "main_table"},
}


def verify_parser_oracle(frame: pd.DataFrame, *, tol: float = 0.05) -> dict:
    """Verify the 12 known samples against the parsed events (real, executed)."""
    by_period = {r["period"]: r for _, r in frame.iterrows()}
    results, passed = {}, 0
    for period, expected in PARSER_ORACLE.items():
        row = by_period.get(period)
        if row is None:
            results[period] = {"pass": False, "reason": "missing"}
            continue
        ok = (abs(float(row["reported_yoy_pct"]) - expected["reported_yoy"]) <= tol
              and abs(float(row["reported_mom_pct"]) - expected["reported_mom"]) <= tol)
        results[period] = {"pass": bool(ok), "reported_yoy": float(row["reported_yoy_pct"]),
                           "reported_mom": float(row["reported_mom_pct"])}
        passed += int(ok)
    return {"expected": len(PARSER_ORACLE), "passed": passed, "all_pass": passed == len(PARSER_ORACLE),
            "per_period": results}
