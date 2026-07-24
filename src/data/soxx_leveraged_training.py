from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .market_update import REPO_ROOT
from .qlib_native import build_qlib_store_from_frame
from .soxx_constituents import PRICE_COLUMNS, fetch_yahoo_daily_ohlcv
from .soxx_pit_holdings import (
    SoxxPitHoldingsPaths,
    build_pit_soxx_weight_panel,
    load_sec_soxx_nport_history,
)


ACTUAL_2X_TRAINING_PRODUCTS: dict[str, dict[str, Any]] = {
    "USD": {
        "underlying": "Dow Jones U.S. Semiconductors Index",
        "underlying_ticker": "SOXX_PROXY",
        "leverage_multiple": 2.0,
        "inception_date": "2007-01-30",
        "role": "long_history_sector_2x_calibration",
    },
    "NVDL": {
        "underlying": "NVDA",
        "underlying_ticker": "NVDA",
        "leverage_multiple": 2.0,
        "inception_date": "2022-12-13",
        "two_x_effective_date": "2024-01-22",
        "role": "short_history_single_stock_2x_calibration",
    },
    "AMDL": {
        "underlying": "AMD",
        "underlying_ticker": "AMD",
        "leverage_multiple": 2.0,
        "inception_date": "2024-03-18",
        "role": "short_history_single_stock_2x_calibration",
    },
    "MULL": {
        "underlying": "MU",
        "underlying_ticker": "MU",
        "leverage_multiple": 2.0,
        "inception_date": "2024-11-12",
        "role": "short_history_single_stock_2x_calibration",
    },
    "INTW": {
        "underlying": "INTC",
        "underlying_ticker": "INTC",
        "leverage_multiple": 2.0,
        "inception_date": "2025-02-13",
        "role": "short_history_single_stock_2x_calibration",
    },
    "MVLL": {
        "underlying": "MRVL",
        "underlying_ticker": "MRVL",
        "leverage_multiple": 2.0,
        "inception_date": "2025-03-07",
        "role": "short_history_single_stock_2x_calibration",
    },
}


@dataclass(frozen=True)
class SoxxLeveragedTrainingPaths:
    root: Path = REPO_ROOT / "data" / "raw" / "soxx_leveraged_training"

    @property
    def daily_dir(self) -> Path:
        return self.root / "daily"

    @property
    def coverage_csv(self) -> Path:
        return self.root / "coverage.csv"

    @property
    def manifest_json(self) -> Path:
        return self.root / "manifest.json"


def synthetic_daily_reset_return(
    underlying_return: pd.Series | np.ndarray,
    *,
    leverage_multiple: float = 2.0,
    annual_fee_bps: float = 95.0,
) -> pd.Series:
    values = pd.Series(
        pd.to_numeric(underlying_return, errors="coerce"),
        index=getattr(underlying_return, "index", None),
        dtype="float64",
    )
    daily_fee = annual_fee_bps / 10_000.0 / 252.0
    return (leverage_multiple * values - daily_fee).clip(lower=-0.999)


def fetch_and_store_actual_2x_training_history(
    *,
    paths: SoxxLeveragedTrainingPaths = SoxxLeveragedTrainingPaths(),
    start: str = "2007-01-30",
    end: str = "2026-07-02",
    sleep_seconds: float = 0.15,
    price_fetcher: Callable[[str], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    fetcher = price_fetcher or (
        lambda symbol: fetch_yahoo_daily_ohlcv(
            symbol,
            start=start,
            end=end,
        )
    )
    paths.daily_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    products = sorted(ACTUAL_2X_TRAINING_PRODUCTS)
    for index, symbol in enumerate(products):
        metadata = ACTUAL_2X_TRAINING_PRODUCTS[symbol]
        try:
            frame = fetcher(symbol)
            frame = frame.loc[
                frame["date"].astype(str).between(start, end)
            ].reset_index(drop=True)
            if frame.empty:
                raise RuntimeError(
                    f"{symbol} returned no rows in requested range"
                )
            missing = sorted(set(PRICE_COLUMNS) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"{symbol} 2x history missing columns {missing}"
                )
            path = paths.daily_dir / f"{symbol}.csv"
            frame.to_csv(path, index=False)
            hashes[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
            coverage_rows.append(
                {
                    "symbol": symbol,
                    "status": "ok",
                    "rows": int(len(frame)),
                    "start_date": str(frame["date"].min()),
                    "end_date": str(frame["date"].max()),
                    **metadata,
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - live provider failures
            coverage_rows.append(
                {
                    "symbol": symbol,
                    "status": "error",
                    "rows": 0,
                    "start_date": "",
                    "end_date": "",
                    **metadata,
                    "error": str(exc),
                }
            )
        if sleep_seconds > 0 and index < len(products) - 1:
            time.sleep(sleep_seconds)

    coverage = pd.DataFrame(coverage_rows).sort_values("symbol")
    coverage.to_csv(paths.coverage_csv, index=False)
    manifest = {
        "dataset": "actual_semiconductor_and_single_stock_2x_training_calibration",
        "status": (
            "ok"
            if coverage["status"].eq("ok").any()
            else "blocked_no_actual_2x_history"
        ),
        "source": "Yahoo Finance chart API",
        "requested_start": start,
        "requested_end": end,
        "products": ACTUAL_2X_TRAINING_PRODUCTS,
        "usable_products": coverage.loc[
            coverage["status"].eq("ok"), "symbol"
        ].tolist(),
        "coverage_output": paths.coverage_csv.resolve().as_posix(),
        "hashes": hashes,
        "limitations": [
            "USD is a sector-level 2x calibration series, not a SOXX clone.",
            "Single-stock 2x ETF histories start in 2022 or later and cannot create pre-2020 crash regimes.",
            "Synthetic constituent 2x paths are explicitly labeled and never represented as executable ETF returns.",
        ],
    }
    paths.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_soxx_pit_leveraged_training_panel(
    *,
    pit_paths: SoxxPitHoldingsPaths = SoxxPitHoldingsPaths(),
    leveraged_paths: SoxxLeveragedTrainingPaths = (
        SoxxLeveragedTrainingPaths()
    ),
    as_of: str = "2026-07-02",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = load_sec_soxx_nport_history(paths=pit_paths)
    if not pit_paths.price_coverage_csv.exists():
        raise FileNotFoundError(
            "SOXX PIT constituent price coverage missing: "
            f"{pit_paths.price_coverage_csv}"
        )
    if not leveraged_paths.coverage_csv.exists():
        raise FileNotFoundError(
            "actual 2x training coverage missing: "
            f"{leveraged_paths.coverage_csv}"
        )
    price_coverage = pd.read_csv(pit_paths.price_coverage_csv)
    constituent_symbols = sorted(
        price_coverage.loc[
            price_coverage["status"].eq("ok"), "symbol"
        ]
        .astype(str)
        .str.upper()
    )
    if not constituent_symbols:
        raise ValueError("blocked: no PIT constituent price histories")

    constituent_frames: list[pd.DataFrame] = []
    for symbol in constituent_symbols:
        path = pit_paths.daily_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"PIT constituent daily history missing: {path}"
            )
        frame = pd.read_csv(path)
        frame["datetime"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.loc[
            frame["datetime"].le(pd.Timestamp(as_of))
        ].copy()
        frame["instrument"] = symbol
        constituent_frames.append(frame)
    constituents = pd.concat(
        constituent_frames,
        ignore_index=True,
    ).sort_values(["instrument", "datetime"])
    calendar = pd.DatetimeIndex(
        sorted(constituents["datetime"].unique())
    )
    pit_weights = build_pit_soxx_weight_panel(
        history,
        calendar=calendar,
    )
    constituents = constituents.merge(
        pit_weights[
            [
                "datetime",
                "instrument",
                "weight_pct",
                "report_date",
                "filing_date",
                "accession",
            ]
        ],
        on=["datetime", "instrument"],
        how="left",
    )
    first_public_date = pd.Timestamp(history["filing_date"].min())
    constituents["pit_available"] = (
        constituents["datetime"].ge(first_public_date).astype(float)
    )
    constituents["pit_member"] = (
        pd.to_numeric(
            constituents["weight_pct"],
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
        .astype(float)
    )
    constituents["weight_pct"] = pd.to_numeric(
        constituents["weight_pct"],
        errors="coerce",
    ).fillna(0.0)
    constituents["instrument_type"] = 0.0
    constituents["leverage_multiple"] = 1.0
    constituents["is_actual_2x"] = 0.0

    actual_coverage = pd.read_csv(leveraged_paths.coverage_csv)
    actual_symbols = sorted(
        actual_coverage.loc[
            actual_coverage["status"].eq("ok"), "symbol"
        ]
        .astype(str)
        .str.upper()
    )
    actual_frames: list[pd.DataFrame] = []
    for symbol in actual_symbols:
        path = leveraged_paths.daily_dir / f"{symbol}.csv"
        frame = pd.read_csv(path)
        frame["datetime"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.loc[
            frame["datetime"].le(pd.Timestamp(as_of))
        ].copy()
        frame["instrument"] = symbol
        frame["weight_pct"] = 0.0
        frame["pit_available"] = 0.0
        frame["pit_member"] = 0.0
        frame["instrument_type"] = 1.0
        frame["leverage_multiple"] = float(
            ACTUAL_2X_TRAINING_PRODUCTS[symbol]["leverage_multiple"]
        )
        frame["is_actual_2x"] = 1.0
        actual_frames.append(frame)

    panel = pd.concat(
        [constituents, *actual_frames],
        ignore_index=True,
    )
    panel = panel[
        [
            "instrument",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "weight_pct",
            "pit_available",
            "pit_member",
            "instrument_type",
            "leverage_multiple",
            "is_actual_2x",
        ]
    ].sort_values(["instrument", "datetime"])
    if panel.duplicated(["instrument", "datetime"]).any():
        raise ValueError(
            "SOXX leveraged training panel contains duplicate rows"
        )
    audit = {
        "source": [
            pit_paths.history_csv.resolve().as_posix(),
            pit_paths.price_manifest_json.resolve().as_posix(),
            leveraged_paths.manifest_json.resolve().as_posix(),
        ],
        "as_of": as_of,
        "constituent_instruments": constituent_symbols,
        "actual_2x_instruments": actual_symbols,
        "pit_first_public_date": first_public_date.date().isoformat(),
        "rows": int(len(panel)),
        "synthetic_2x_storage_policy": (
            "derived labels/features only; no synthetic instrument price"
        ),
    }
    return panel.reset_index(drop=True), audit


def build_soxx_pit_leveraged_training_qlib_store(
    *,
    provider_uri: Path = (
        REPO_ROOT / "data" / "qlib" / "soxx_pit_leveraged_training_v1"
    ),
    pit_paths: SoxxPitHoldingsPaths = SoxxPitHoldingsPaths(),
    leveraged_paths: SoxxLeveragedTrainingPaths = (
        SoxxLeveragedTrainingPaths()
    ),
    as_of: str = "2026-07-02",
) -> dict[str, Any]:
    panel, audit = build_soxx_pit_leveraged_training_panel(
        pit_paths=pit_paths,
        leveraged_paths=leveraged_paths,
        as_of=as_of,
    )
    manifest = build_qlib_store_from_frame(
        panel,
        provider_uri=provider_uri,
        market="all",
        region="us",
        freq="day",
        as_of=as_of,
        source=";".join(audit["source"]),
        field_map={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$adj_close": "adj_close",
            "$volume": "volume",
            "$pit_weight_pct": "weight_pct",
            "$pit_available": "pit_available",
            "$pit_member": "pit_member",
            "$instrument_type": "instrument_type",
            "$leverage_multiple": "leverage_multiple",
            "$is_actual_2x": "is_actual_2x",
        },
    )
    manifest["training_data_audit"] = audit
    manifest["synthetic_2x_formula"] = (
        "clip(2 * underlying_daily_return - 95bp/252, -0.999, inf)"
    )
    manifest["limitations"] = [
        "PIT holdings are usable only on or after SEC filing_date.",
        "Price coverage omits unavailable delisted symbols and reports snapshot weight coverage.",
        "Actual single-stock 2x histories are calibration-only because they begin in 2022 or later.",
    ]
    durable_sidecars = {
        "sec_nport_holdings_history": (
            pit_paths.history_csv,
            Path(provider_uri).resolve() / "sec_nport_holdings_history.csv",
        ),
        "sec_nport_holdings_manifest": (
            pit_paths.manifest_json,
            Path(provider_uri).resolve()
            / "sec_nport_holdings_manifest.json",
        ),
        "constituent_price_coverage": (
            pit_paths.price_coverage_csv,
            Path(provider_uri).resolve()
            / "constituent_price_coverage.csv",
        ),
        "constituent_price_manifest": (
            pit_paths.price_manifest_json,
            Path(provider_uri).resolve()
            / "constituent_price_manifest.json",
        ),
        "actual_2x_coverage": (
            leveraged_paths.coverage_csv,
            Path(provider_uri).resolve() / "actual_2x_coverage.csv",
        ),
        "actual_2x_manifest": (
            leveraged_paths.manifest_json,
            Path(provider_uri).resolve() / "actual_2x_manifest.json",
        ),
    }
    missing_sidecars = [
        source.as_posix()
        for source, _ in durable_sidecars.values()
        if not source.exists()
    ]
    if missing_sidecars:
        raise FileNotFoundError(
            "durable SOXX training sidecars missing: "
            f"{missing_sidecars}"
        )
    for source, destination in durable_sidecars.values():
        shutil.copyfile(source, destination)
    manifest["durable_sidecars"] = {
        name: destination.name
        for name, (_, destination) in durable_sidecars.items()
    }
    manifest_path = Path(provider_uri).resolve() / "qlib_store_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
