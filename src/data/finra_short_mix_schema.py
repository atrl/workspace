"""Strict CNMS source parsing; exact received share quantities remain strings.

The returned source panel is a derived numeric cache, not the exact-data
authority or a historical publication-time certification.  CNMS ShortVolume
already includes ShortExemptVolume; the two quantities must never be added.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, localcontext
import hashlib
import re

import numpy as np
import pandas as pd


HEADER = ("Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market")
SYMBOLS = ("SOXL", "SOXX")
QUANTITIES = ("ShortVolume", "ShortExemptVolume", "TotalVolume")
CNMS_MARKETS = frozenset({"B", "Q", "N", "D"})
CACHE_FIELDS = ("short_volume", "short_exempt_volume", "total_volume", "short_ratio", "exempt_ratio", "source_ready")
_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]{1,6})?\Z", re.ASCII)
_COUNT = re.compile(r"[0-9]+\Z", re.ASCII)
_MARKETS = re.compile(r"[A-Z](?:,[A-Z])*\Z", re.ASCII)


class CNMSSchemaError(ValueError):
    """A retained raw file cannot qualify under the declared CNMS schema."""


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{8}", value, re.ASCII):
        raise CNMSSchemaError("trade date must be exactly YYYYMMDD")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise CNMSSchemaError("invalid calendar trade date: " + value) from error
    if parsed.strftime("%Y%m%d") != value:
        raise CNMSSchemaError("noncanonical trade date: " + value)
    return parsed


def _quantity(value, *, field, line):
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise CNMSSchemaError(f"line {line}: {field} must be an unsigned decimal with at most six decimal places")
    return Decimal(value)


def _validate_row(fields, *, expected_date, line):
    if len(fields) != len(HEADER):
        raise CNMSSchemaError(f"line {line}: expected six pipe-delimited fields")
    row = dict(zip(HEADER, fields))
    if row["Date"] != expected_date:
        raise CNMSSchemaError(f"line {line}: data date differs from requested {expected_date}")
    symbol = row["Symbol"]
    if not symbol or any(character.isspace() or not character.isprintable() for character in symbol):
        raise CNMSSchemaError(f"line {line}: empty or malformed Symbol")
    short, exempt, total = [_quantity(row[field], field=field, line=line) for field in QUANTITIES]
    if not Decimal(0) <= exempt <= short <= total:
        raise CNMSSchemaError(f"line {line}: require 0 <= ShortExemptVolume <= ShortVolume <= TotalVolume")
    market = row["Market"]
    codes = market.split(",")
    if _MARKETS.fullmatch(market) is None or len(codes) != len(set(codes)):
        raise CNMSSchemaError(f"line {line}: malformed or repeated Market codes")
    unknown = set(codes) - CNMS_MARKETS
    if unknown:
        raise CNMSSchemaError(f"line {line}: CNMS scope anomaly, unsupported Market codes {sorted(unknown)}")
    return row


def parse_cnms_file(body: bytes, expected_date: str) -> dict:
    """Validate every record before selecting exact SOXL/SOXX symbol matches.

    All three quantities retain their original decimal spelling, including
    trailing zeros.  Invalid non-target rows also reject the file; absence of
    a target symbol remains missing rather than a zero-volume observation.
    """
    _date(expected_date)
    if not isinstance(body, bytes) or not body:
        raise CNMSSchemaError("CNMS body must be nonempty bytes")
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError as error:
        raise CNMSSchemaError("CNMS source contains non-ASCII bytes") from error
    lines = text.splitlines()
    if len(lines) < 2 or lines[0] != "|".join(HEADER):
        raise CNMSSchemaError("CNMS header differs from the frozen six-field schema")
    if _COUNT.fullmatch(lines[-1]) is None:
        raise CNMSSchemaError("CNMS footer must be the integer data-record count")
    record_count = len(lines) - 2
    if int(lines[-1]) != record_count:
        raise CNMSSchemaError("CNMS footer record count differs from all data rows")
    selected, symbols, markets = {}, set(), set()
    decimal_rows = max_places = 0
    for line_number, line in enumerate(lines[1:-1], start=2):
        row = _validate_row(line.split("|"), expected_date=expected_date, line=line_number)
        symbol = row["Symbol"]
        if symbol in symbols:
            raise CNMSSchemaError(f"line {line_number}: duplicate Date/Symbol {expected_date}/{symbol}")
        symbols.add(symbol)
        markets.update(row["Market"].split(","))
        places = [len(row[field].partition(".")[2]) for field in QUANTITIES]
        max_places = max(max_places, *places)
        decimal_rows += int(any(places))
        if symbol in SYMBOLS:
            selected[symbol] = row
    return {
        "trade_date": expected_date,
        "rows": [selected[symbol] for symbol in SYMBOLS if symbol in selected],
        "record_count": record_count,
        "missing_symbols": [symbol for symbol in SYMBOLS if symbol not in selected],
        "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
        "header": list(HEADER), "footer": lines[-1],
        "schema": {
            "name": "finra_cnms_six_field_decimal_v1", "maximum_decimal_places": 6,
            "maximum_observed_decimal_places": max_places, "decimal_rows": decimal_rows,
            "unique_symbols": len(symbols), "market_codes_observed": sorted(markets),
            "short_volume_includes_short_exempt_volume": True,
            "exact_quantity_storage": "original_decimal_strings",
            "historical_PIT_qualified": False,
        },
    }


def source_frame(parsed_by_date) -> pd.DataFrame:
    """Materialize two source rows per requested date; None means failed intake.

    Source quantities stay exact in the parsed objects.  These float64 values
    are only an intermediate cache for the Qlib float32 store; the workflow
    must report readback quantization error against the exact source values.
    No prices, return labels, availability inference or fitted features enter
    this function.  Ratios are undefined for missing or zero total volume.
    """
    records = []
    for date in sorted(parsed_by_date):
        stamp = _date(date)
        parsed = parsed_by_date[date]
        by_symbol = {}
        if parsed is not None:
            if parsed.get("trade_date") != date:
                raise CNMSSchemaError("parsed file identity differs from requested source-panel date")
            for source in parsed["rows"]:
                row = _validate_row([source[name] for name in HEADER], expected_date=date, line="selected")
                symbol = row["Symbol"]
                if symbol not in SYMBOLS or symbol in by_symbol:
                    raise CNMSSchemaError("unexpected or duplicate selected source symbol")
                by_symbol[symbol] = row
            missing = [symbol for symbol in SYMBOLS if symbol not in by_symbol]
            if parsed.get("missing_symbols") != missing:
                raise CNMSSchemaError("selected rows and declared missing symbols disagree")
        for symbol in SYMBOLS:
            values = dict.fromkeys(CACHE_FIELDS, np.nan)
            values["source_ready"] = 0.
            if symbol in by_symbol:
                row = by_symbol[symbol]
                short, exempt, total = [Decimal(row[field]) for field in QUANTITIES]
                values.update(zip(CACHE_FIELDS[:3], map(float, (short, exempt, total))))
                if not np.isfinite([values[name] for name in CACHE_FIELDS[:3]]).all():
                    raise CNMSSchemaError("exact source quantity cannot be represented in the derived numeric cache")
                if total > 0:
                    with localcontext() as context:
                        context.prec = 38
                        values.update(short_ratio=float(short / total), exempt_ratio=float(exempt / total), source_ready=1.)
            records.append({"datetime": pd.Timestamp(stamp), "instrument": symbol, **values})
    return pd.DataFrame(records, columns=["datetime", "instrument", *CACHE_FIELDS])
