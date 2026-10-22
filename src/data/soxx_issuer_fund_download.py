"""Read the exact iShares SOXX fund-download SpreadsheetML history."""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd


_SS = "{urn:schemas-microsoft-com:office:spreadsheet}"
_BARE_AMPERSAND = re.compile(r"&(?!#(?:\d+|x[0-9A-Fa-f]+);|amp;|lt;|gt;|quot;|apos;)")
_HEADERS = ["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding"]
_HEADERS_WITH_INDEX = ["As Of", "Index Level", "NAV per Share",
                       "Ex-Dividends", "Shares Outstanding"]


def read_historical(source: Path) -> tuple[pd.DataFrame, dict]:
    """Parse the issuer's daily table while preserving the original file bytes.

    The published .xls is SpreadsheetML with bare ampersands in unrelated
    hyperlink/text fields. Escape only those invalid XML characters in memory.
    The caller keeps and hashes the unmodified downloaded file.
    """
    raw = Path(source).read_bytes().decode("utf-8-sig").lstrip("\ufeff")
    escaped, replacements = _BARE_AMPERSAND.subn("&amp;", raw)
    workbook = ElementTree.fromstring(escaped)
    sheets = [sheet for sheet in workbook.findall(_SS + "Worksheet")
              if sheet.get(_SS + "Name") == "Historical"]
    if len(sheets) != 1:
        raise ValueError("blocked: expected exactly one issuer Historical worksheet")
    rows = [[cell.text for cell in row.findall(".//" + _SS + "Data")]
            for row in sheets[0].findall(".//" + _SS + "Row")]
    if (not rows or rows[0] not in (_HEADERS, _HEADERS_WITH_INDEX)
            or any(len(row) != len(rows[0]) for row in rows[1:])):
        raise ValueError("blocked: issuer Historical worksheet schema changed")
    names = (["as_of", "index_level", "nav_per_share", "ex_dividends",
              "shares_outstanding"] if rows[0] == _HEADERS_WITH_INDEX else
             ["as_of", "nav_per_share", "ex_dividends", "shares_outstanding"])
    frame = pd.DataFrame(rows[1:], columns=names)
    frame["as_of"] = pd.to_datetime(frame.as_of, format="%b %d, %Y", errors="raise")
    for field in ("nav_per_share", "shares_outstanding"):
        if not frame[field].loc[~frame[field].eq("--")].map(
                lambda value: bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value))).all():
            raise ValueError("blocked: issuer Historical numeric field changed")
        frame[field] = pd.to_numeric(frame[field].replace("--", pd.NA), errors="raise")
    if (frame.as_of.duplicated().any() or frame.as_of.isna().any()
            or frame.nav_per_share.le(0).any() or frame.shares_outstanding.le(0).any()):
        raise ValueError("blocked: issuer Historical dates or positive numeric fields invalid")
    return frame.drop(columns=["index_level"], errors="ignore").set_index("as_of").sort_index(), {
        "schema": rows[0], "daily_calendar_rows": len(frame),
        "index_level_column_present": rows[0] == _HEADERS_WITH_INDEX,
        "historical_missing_nav_rows": int(frame.nav_per_share.isna().sum()),
        "historical_missing_shares_rows": int(frame.shares_outstanding.isna().sum()),
        "invalid_xml_ampersands_escaped_in_memory": replacements,
        "raw_source_modified": False,
    }
