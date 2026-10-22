"""Parse one original G.17 Table 1 semiconductor industrial-production vintage."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from src.data.fed_capacity_vintages import MONTHS, MONTH_PATTERN


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def parse_ip_supplement(data: bytes, *, expected_date: str,
                        observation_month: str, monthly_columns: int = 6) -> dict:
    if monthly_columns not in (6, 7):
        raise ValueError("blocked: unsupported original G17 month-column count")
    text = data.decode("utf-8-sig")
    stamp = re.search(r"(" + MONTH_PATTERN + r")\s+(\d{1,2}),?\s+(20\d{2})", text[:1500], re.I)
    if not stamp or not re.search(r"9:15\s*a\.?m\.?", text[:1500], re.I):
        raise ValueError("needs_data: original G17 release date or clock missing")
    release = pd.Timestamp(f"{stamp[3]}-{MONTHS[stamp[1][:3].lower()]:02}-{int(stamp[2]):02}")
    if release != pd.Timestamp(expected_date):
        raise ValueError("blocked: raw original and inventory release dates differ")
    expected_month = pd.Timestamp(observation_month).to_period("M")
    if expected_month.end_time >= release:
        raise ValueError("blocked: observation month is not earlier than release")

    lines = text.splitlines()
    candidates = []
    for i, line in enumerate(lines):
        if "Semiconductor and other electronic" not in line:
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            pieces = lines[j].split("|")
            if "component" not in lines[j] or len(pieces) != 5 or pieces[1].strip() != "3344":
                continue
            prefix = "\n".join(lines[:i])
            if prefix.rfind("Table 1") <= prefix.rfind("Table 2"):
                continue
            candidates.append((i, j, pieces))
    if len(candidates) != 1:
        raise ValueError("needs_data: Table 1 NAICS3344 row is missing or ambiguous")
    i, j, pieces = candidates[0]
    heading = lines[max(0, i - 30):i]
    bases = re.findall(r"Index,\s*(\d{4})\s*=\s*100", "\n".join(heading))
    monthly = [line for line in heading if "NAICS" in line and "Proportion" in line]
    if len(bases) != 1 or len(monthly) != 1 or len(monthly[0].split("|")) < 5:
        raise ValueError("needs_data: G17 IP base year or six-month heading ambiguous")
    header = monthly[0].split("|")[3]
    month_tokens = re.findall(r"\b(" + MONTH_PATTERN + r")\.?", header, re.I)
    months = [MONTHS[m[:3].lower()] for m in month_tokens]
    values = [float(x) for x in re.findall(NUMBER, pieces[3])]
    expected = pd.period_range(end=expected_month, periods=monthly_columns, freq="M")
    if (len(months) != monthly_columns or len(values) != monthly_columns or list(expected.month) != months
            or not np.isfinite(values).all() or not all(v > 0 for v in values)):
        raise ValueError("needs_data: original six seasonally adjusted IP months incomplete")
    return {
        "release_date": str(release.date()),
        "declared_release_at": str((release + pd.Timedelta(hours=9, minutes=15)).tz_localize("America/New_York")),
        "observation_month": str(expected_month.to_timestamp().date()),
        "index_base_year": int(bases[0]),
        "published_months": [str(p.to_timestamp().date()) for p in expected],
        "published_SA_indices": values,
        "ip_latest_index": values[-1],
        "ip_prior_index_same_vintage": values[-2],
        "ip_mom_pct_same_vintage": float(100 * (values[-1] / values[-2] - 1)),
        "NAICS": "3344",
        "table": "Table 1 Industrial Production: Industry Subtotals and Individual Series",
        "raw_sha256": hashlib.sha256(data).hexdigest(),
        "layout_blank_lines": j - i - 1,
        "historical_first_receipt_qualified": False,
    }
