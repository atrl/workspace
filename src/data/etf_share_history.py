"""Issuer SpreadsheetML share history; diagnostic availability is explicit."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def parse_ishares_history(raw):
    text = raw.decode("utf-8-sig")
    sheets = re.findall(r'<(?:ss:)?Worksheet\b[^>]*\bss:Name=["\']Historical["\'][^>]*>.*?</(?:ss:)?Worksheet>', text, re.S)
    if len(sheets) != 1:
        raise ValueError("expected one Historical worksheet")
    sheet = ET.fromstring('<root xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">' + sheets[0] + '</root>')
    ns = "{urn:schemas-microsoft-com:office:spreadsheet}"
    rows = []
    for row in sheet.iter():
        if row.tag.split("}")[-1] != "Row":
            continue
        cells = []
        for cell in row:
            if cell.tag.split("}")[-1] != "Cell":
                continue
            if any(key in cell.attrib for key in [ns + "Formula", ns + "MergeAcross", ns + "MergeDown"]):
                raise ValueError("unsupported formula or merged history cell")
            index = int(cell.attrib.get(ns + "Index", len(cells) + 1))
            if index != len(cells) + 1:
                raise ValueError("noncontiguous history cells")
            values = [item.text or "" for item in cell if item.tag.split("}")[-1] == "Data"]
            if len(values) != 1:
                raise ValueError("expected one Data value per cell")
            cells.append(values[0])
        rows.append(cells)
    if rows[0] != ["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding"] or any(len(row) != 4 for row in rows):
        raise ValueError("issuer history schema changed")
    frame = pd.DataFrame(rows[1:], columns=["date", "nav", "ex_dividend", "shares"])
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    for column in ["nav", "ex_dividend", "shares"]:
        frame[column] = pd.to_numeric(frame[column].str.replace(",", "").replace("--", np.nan), errors="raise")
    if frame.date.duplicated().any() or (frame[["nav", "shares"]].le(0) & frame[["nav", "shares"]].notna()).any().any():
        raise ValueError("duplicate dates or nonpositive issuer NAV/shares")
    return frame.set_index("date").sort_index()


def share_information(history, calendar, lag_sessions=2):
    """No forward fill, unknown is unknown; lag is an assumption, not a PIT certificate."""
    if lag_sessions < 1:
        raise ValueError("same-session end-of-day shares are unavailable at morning decision")
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("calendar must be unique and increasing")
    aligned = history.reindex(calendar)
    growth = aligned.shares.div(aligned.shares.shift(1)).sub(1)
    result = pd.DataFrame({"share_change_1d": growth,
        "share_change_5d": aligned.shares.div(aligned.shares.shift(5)).sub(1),
        "absolute_share_change_5d": growth.abs().rolling(5, min_periods=5).mean()}, index=calendar)
    # Cumulative change must not bridge missing daily share observations.
    result.loc[aligned.shares.rolling(6, min_periods=6).count().ne(6), "share_change_5d"] = np.nan
    result = result.shift(lag_sessions)
    result["source_date"] = pd.Series(calendar, index=calendar).shift(lag_sessions)
    result["assumed_available_at"] = calendar + pd.Timedelta(hours=9, minutes=31)
    result["known"] = result.iloc[:, :3].notna().all(axis=1).astype(float)
    return result
