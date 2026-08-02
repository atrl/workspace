from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .qlib_native import QlibProviderConfig, build_qlib_store_from_frame, init_qlib_provider


MACRO_STATE_INSTRUMENT = "MACRO_STATE"
MACRO_STATE_PROVIDER_NAME = "macro_state_v1"
FRED_GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_USER_AGENT = "Mozilla/5.0 soxl-qlib-macro-state"
TREASURY_XML_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
GOVSPENDING_FRED_EXPORT_URL = "https://govspending.org/api/export/fred/{series_id}.csv"
FRED_MACRO_STATE_SERIES: dict[str, dict[str, Any]] = {
    "dgs2": {
        "series_id": "DGS2",
        "description": "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity",
        "availability_lag_trading_days": 1,
    },
    "dgs10": {
        "series_id": "DGS10",
        "description": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
        "availability_lag_trading_days": 1,
    },
    "dfii10": {
        "series_id": "DFII10",
        "description": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Inflation-Indexed",
        "availability_lag_trading_days": 1,
    },
    "t5yifr": {
        "series_id": "T5YIFR",
        "description": "5-Year, 5-Year Forward Inflation Expectation Rate",
        "availability_lag_trading_days": 1,
    },
    "dtwexbgs": {
        "series_id": "DTWEXBGS",
        "description": "Nominal Broad U.S. Dollar Index",
        "availability_lag_trading_days": 2,
    },
}
MACRO_STATE_SERIES: dict[str, dict[str, Any]] = {
    "dgs2": {
        "series_id": "UST_NOMINAL_PAR_2Y",
        "target_concept": "DGS2",
        "description": "U.S. Treasury daily nominal par yield at 2 years; proxy for the DGS2 concept",
        "availability_lag_trading_days": 1,
    },
    "dgs10": {
        "series_id": "UST_NOMINAL_PAR_10Y",
        "target_concept": "DGS10",
        "description": "U.S. Treasury daily nominal par yield at 10 years; proxy for the DGS10 concept",
        "availability_lag_trading_days": 1,
    },
    "dfii10": {
        "series_id": "UST_REAL_PAR_10Y",
        "target_concept": "DFII10",
        "description": "U.S. Treasury daily real par yield at 10 years; proxy for the DFII10 concept",
        "availability_lag_trading_days": 1,
    },
    "t5y5y_inflation_proxy": {
        "series_id": "UST_PAR_NOMINAL_REAL_5Y10Y_DERIVED",
        "target_concept": "T5YIFR",
        "description": "Approximate 5Y5Y inflation expectation from Treasury nominal and real par curves",
        "availability_lag_trading_days": 1,
    },
    "dtwexbgs": {
        "series_id": "DTWEXBGS",
        "description": "Nominal Broad U.S. Dollar Index",
        "availability_lag_trading_days": 2,
    },
}
MACRO_STATE_FIELDS = tuple(MACRO_STATE_SERIES)
STATE_SUFFIXES = (
    "source_observed",
    "is_carried",
    "staleness_days",
    "publication_lag_days",
    "observation_epoch_days",
)


def fetch_fred_macro_state_series(
    *,
    start: str,
    end: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch required no-auth FRED series and fail closed if any series is unavailable."""

    client = session or requests.Session()
    rows: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for field, metadata in FRED_MACRO_STATE_SERIES.items():
        series_id = str(metadata["series_id"])
        try:
            response = client.get(
                FRED_GRAPH_CSV_URL,
                params={"id": series_id, "cosd": start, "coed": end},
                timeout=30,
                headers={"User-Agent": FRED_USER_AGENT},
            )
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            date_col = "DATE" if "DATE" in frame.columns else "observation_date"
            if date_col not in frame or series_id not in frame:
                raise ValueError(f"unexpected FRED columns: {frame.columns.tolist()}")
            parsed = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(frame[date_col], errors="coerce").dt.normalize(),
                    "field": field,
                    "value": pd.to_numeric(frame[series_id], errors="coerce"),
                    "series_id": series_id,
                    "source": "fred_graph_csv",
                }
            ).dropna(subset=["datetime", "value"])
            parsed = parsed.loc[
                parsed["datetime"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
            ]
            if parsed.empty:
                raise ValueError("no observations in requested range")
            rows.append(parsed)
        except Exception as exc:  # pragma: no cover - exercised through injected session tests
            errors[field] = str(exc)
    if errors:
        raise RuntimeError(f"blocked: required FRED macro-state series unavailable: {errors}")
    return pd.concat(rows, ignore_index=True).sort_values(["field", "datetime"]).reset_index(drop=True)


def _treasury_xml_rows(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for properties in root.findall(".//{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}properties"):
        row = {child.tag.rsplit("}", 1)[-1]: child.text for child in properties}
        rows.append(row)
    return pd.DataFrame(rows)


def _fetch_treasury_curve_year(kind: str, year: int) -> pd.DataFrame:
    response = requests.get(
        TREASURY_XML_URL,
        params={"data": kind, "field_tdr_date_value": str(year)},
        timeout=30,
        headers={"User-Agent": FRED_USER_AGENT},
    )
    response.raise_for_status()
    frame = _treasury_xml_rows(response.text)
    if frame.empty or "NEW_DATE" not in frame:
        raise ValueError(f"Treasury {kind} returned no rows for {year}")
    frame["datetime"] = pd.to_datetime(frame["NEW_DATE"], errors="coerce").dt.normalize()
    return frame


def fetch_treasury_macro_state_series(*, start: str, end: str) -> pd.DataFrame:
    """Fetch official nominal/real Treasury curves and derive a transparent 5Y5Y proxy."""

    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    tasks = [(kind, year) for year in years for kind in ("daily_treasury_yield_curve", "daily_treasury_real_yield_curve")]
    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
        fetched = list(executor.map(lambda item: _fetch_treasury_curve_year(*item), tasks))
    nominal = pd.concat(fetched[0::2], ignore_index=True)
    real = pd.concat(fetched[1::2], ignore_index=True)
    nominal = nominal[["datetime", "BC_2YEAR", "BC_5YEAR", "BC_10YEAR"]].rename(
        columns={"BC_2YEAR": "dgs2", "BC_5YEAR": "nominal5", "BC_10YEAR": "dgs10"}
    )
    real = real[["datetime", "TC_5YEAR", "TC_10YEAR"]].rename(
        columns={"TC_5YEAR": "real5", "TC_10YEAR": "dfii10"}
    )
    curve = nominal.merge(real, on="datetime", how="inner")
    for column in ["dgs2", "nominal5", "dgs10", "real5", "dfii10"]:
        curve[column] = pd.to_numeric(curve[column], errors="coerce")
    curve["t5y5y_inflation_proxy"] = 2.0 * (curve["dgs10"] - curve["dfii10"]) - (
        curve["nominal5"] - curve["real5"]
    )
    curve = curve.loc[curve["datetime"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")]
    rows: list[pd.DataFrame] = []
    source_ids = {
        "dgs2": "UST_NOMINAL_PAR_2Y",
        "dgs10": "UST_NOMINAL_PAR_10Y",
        "dfii10": "UST_REAL_PAR_10Y",
        "t5y5y_inflation_proxy": "UST_PAR_NOMINAL_REAL_5Y10Y_DERIVED",
    }
    for field, series_id in source_ids.items():
        part = curve[["datetime", field]].rename(columns={field: "value"}).dropna()
        part["field"] = field
        part["series_id"] = series_id
        part["source"] = "us_treasury_daily_interest_rate_xml"
        rows.append(part)
    if any(part.empty for part in rows):
        raise RuntimeError("blocked: required Treasury macro-state curve is incomplete")
    return pd.concat(rows, ignore_index=True)


def fetch_broad_dollar_series(*, start: str, end: str) -> pd.DataFrame:
    response = requests.get(
        GOVSPENDING_FRED_EXPORT_URL.format(series_id="DTWEXBGS"),
        timeout=30,
        headers={"User-Agent": FRED_USER_AGENT},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), comment="#")
    if not {"date", "DTWEXBGS"}.issubset(frame.columns):
        raise RuntimeError("blocked: broad-dollar mirror returned unexpected schema")
    output = pd.DataFrame(
        {
            "datetime": pd.to_datetime(frame["date"], errors="coerce").dt.normalize(),
            "field": "dtwexbgs",
            "value": pd.to_numeric(frame["DTWEXBGS"], errors="coerce"),
            "series_id": "DTWEXBGS",
            "source": "govspending_fred_export",
        }
    ).dropna(subset=["datetime", "value"])
    output = output.loc[output["datetime"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")]
    if output.empty:
        raise RuntimeError("blocked: broad-dollar mirror has no observations in requested range")
    return output


def fetch_macro_state_series(*, start: str, end: str) -> pd.DataFrame:
    treasury = fetch_treasury_macro_state_series(start=start, end=end)
    dollar = fetch_broad_dollar_series(start=start, end=end)
    return pd.concat([treasury, dollar], ignore_index=True).sort_values(["field", "datetime"]).reset_index(drop=True)


def build_macro_state_daily_frame(
    observations: pd.DataFrame,
    *,
    calendar: Iterable[str | pd.Timestamp],
) -> pd.DataFrame:
    """Map observations to their conservative model-availability trading dates."""

    required = {"datetime", "field", "value"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"macro-state observations missing columns: {missing}")
    calendar_index = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().sort_values().unique()
    if len(calendar_index) == 0:
        raise ValueError("macro-state calendar cannot be empty")
    raw = observations.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce").dt.normalize()
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["datetime", "field", "value"])

    output = pd.DataFrame({"datetime": calendar_index, "instrument": MACRO_STATE_INSTRUMENT})
    epoch = pd.Timestamp("1970-01-01")
    for field, metadata in MACRO_STATE_SERIES.items():
        lag = int(metadata["availability_lag_trading_days"])
        field_raw = raw.loc[raw["field"].eq(field)].sort_values("datetime")
        if field_raw.empty:
            raise ValueError(f"macro-state required field has no observations: {field}")
        available_values: dict[pd.Timestamp, float] = {}
        available_observed: dict[pd.Timestamp, pd.Timestamp] = {}
        for row in field_raw.itertuples(index=False):
            first_index = int(calendar_index.searchsorted(pd.Timestamp(row.datetime), side="left"))
            available_index = first_index + lag
            if available_index >= len(calendar_index):
                continue
            available_date = pd.Timestamp(calendar_index[available_index])
            available_values[available_date] = float(row.value)
            available_observed[available_date] = pd.Timestamp(row.datetime)

        observed_value = pd.Series(np.nan, index=calendar_index, dtype="float64")
        observed_date = pd.Series(pd.NaT, index=calendar_index, dtype="datetime64[ns]")
        if available_values:
            dates = pd.DatetimeIndex(sorted(available_values))
            observed_value.loc[dates] = [available_values[pd.Timestamp(date)] for date in dates]
            observed_date.loc[dates] = [available_observed[pd.Timestamp(date)] for date in dates]
        source_observed = observed_value.notna().astype("float64")
        carried_value = observed_value.ffill()
        carried_observed_date = observed_date.ffill()
        available_date_series = pd.Series(pd.NaT, index=calendar_index, dtype="datetime64[ns]")
        available_date_series.loc[source_observed.eq(1.0)] = calendar_index[source_observed.eq(1.0)]
        carried_available_date = available_date_series.ffill()
        output[field] = carried_value.to_numpy()
        output[f"{field}_source_observed"] = source_observed.to_numpy()
        output[f"{field}_is_carried"] = (
            carried_value.notna() & source_observed.eq(0.0)
        ).astype("float64").to_numpy()
        output[f"{field}_staleness_days"] = (
            pd.Series(calendar_index, index=calendar_index) - carried_available_date
        ).dt.days.astype("float64").to_numpy()
        output[f"{field}_publication_lag_days"] = (
            carried_available_date - carried_observed_date
        ).dt.days.astype("float64").to_numpy()
        output[f"{field}_observation_epoch_days"] = (
            (carried_observed_date - epoch).dt.days.astype("float64")
        ).to_numpy()
    return output.replace([np.inf, -np.inf], np.nan)


def _soxl_calendar(root: Path, *, start: str, end: str) -> list[str]:
    config = QlibProviderConfig(provider_uri=root / "data" / "qlib" / "soxl", region="us", market="all")
    init_qlib_provider(config)
    from qlib.data import D

    return [pd.Timestamp(value).date().isoformat() for value in D.calendar(start, end, freq="day")]


def build_macro_state_qlib_store_from_frame(
    frame: pd.DataFrame,
    *,
    provider_uri: Path,
    as_of: str,
) -> dict[str, Any]:
    return build_qlib_store_from_frame(
        frame,
        provider_uri=provider_uri,
        market="all",
        region="us",
        freq="day",
        instrument_col="instrument",
        datetime_col="datetime",
        as_of=as_of,
        source="fred_macro_state_release_mapped_daily_state",
    )


def build_macro_state_qlib_store(
    *,
    root: Path,
    start: str,
    end: str,
    provider_uri: Path | None = None,
    raw_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    provider_uri = provider_uri or root / "data" / "qlib" / MACRO_STATE_PROVIDER_NAME
    raw_dir = raw_dir or root / "data" / "raw" / "external" / "macro_state"
    raw_dir.mkdir(parents=True, exist_ok=True)
    observations = fetch_macro_state_series(start=start, end=end)
    calendar = _soxl_calendar(root, start=start, end=end)
    daily = build_macro_state_daily_frame(observations, calendar=calendar)
    observations.to_csv(raw_dir / "macro_state_observations.csv", index=False)
    daily.to_csv(raw_dir / "macro_state_daily.csv", index=False)
    manifest = build_macro_state_qlib_store_from_frame(daily, provider_uri=provider_uri, as_of=end)
    raw_manifest = {
        "source": "us_treasury_xml_plus_govspending_fred_export",
        "source_urls": [TREASURY_XML_URL, GOVSPENDING_FRED_EXPORT_URL.format(series_id="DTWEXBGS")],
        "series": MACRO_STATE_SERIES,
        "requested_start": start,
        "requested_end": end,
        "calendar_start": min(calendar),
        "calendar_end": max(calendar),
        "observation_count": int(len(observations)),
        "row_count": int(len(daily)),
        "availability_policy": "observation_date_plus_conservative_trading_day_lag",
        "limitations": [
            "Treasury curve history is current-vintage and is not a revision-vintage archive.",
            "The 5Y5Y inflation feature is a transparent approximation derived from Treasury 5Y/10Y nominal and real par yields, not FRED T5YIFR.",
            "Daily Treasury and derived inflation-expectation observations are delayed one trading day before model use.",
            "The broad-dollar series is delayed two trading days before model use.",
            "State provenance columns expose source observations, carried values, staleness, and publication lag.",
        ],
    }
    raw_manifest_path = raw_dir / "macro_state_raw_manifest.json"
    raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest.update(
        {
            "provider_uri": "data/qlib/macro_state_v1",
            "raw_manifest": "data/raw/external/macro_state/macro_state_raw_manifest.json",
            "series": MACRO_STATE_SERIES,
            "availability_policy": raw_manifest["availability_policy"],
            "limitations": raw_manifest["limitations"],
        }
    )
    (provider_uri / "qlib_store_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (provider_uri / "qlib.yaml").write_text(
        "# Qlib configuration\n"
        "provider_uri: data/qlib/macro_state_v1\n"
        "redis:\n"
        "  host: null\n"
        "  port: null\n"
        "market:\n"
        "  market: all\n"
        "  market_type: stock\n",
        encoding="utf-8",
    )
    return manifest
