from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .market_update import (
    OHLCV_COLUMNS,
    QLIB_FIELD_MAP,
    REPO_ROOT,
    _csv_summary,
    _fetch_start_from_existing,
    _fetch_vix_with_fallback,
    _fetch_yfinance_compatible_ohlcv,
    _normalize_ohlcv_frame,
    _normalize_vix_frame,
    _require_settled_us_daily_end,
    _write_csv,
)
from .qlib_native import QlibProviderConfig, audit_qlib_provider, build_qlib_store_from_frame


@dataclass(frozen=True)
class Multi3xEtfPairSpec:
    etf: str
    underlying: str

    @property
    def provider_name(self) -> str:
        return self.etf.lower()

    @property
    def panel_name(self) -> str:
        return f"{self.etf.lower()}_with_underlying_vix"


@dataclass(frozen=True)
class Multi3xEtfUpdatePaths:
    panel_dir: Path = REPO_ROOT / "data" / "processed" / "panels"
    raw_symbol_dir: Path = REPO_ROOT / "data" / "raw" / "symbols"
    qlib_dir: Path = REPO_ROOT / "data" / "qlib"
    registry_yaml: Path = REPO_ROOT / "data" / "datasets.yaml"


MULTI_3X_ETF_PAIRS: tuple[Multi3xEtfPairSpec, ...] = (
    Multi3xEtfPairSpec("TQQQ", "QQQ"),
    Multi3xEtfPairSpec("UPRO", "SPY"),
    Multi3xEtfPairSpec("SPXL", "SPY"),
    Multi3xEtfPairSpec("TECL", "XLK"),
    Multi3xEtfPairSpec("UDOW", "DIA"),
    Multi3xEtfPairSpec("TNA", "IWM"),
    Multi3xEtfPairSpec("MIDU", "MDY"),
    Multi3xEtfPairSpec("FAS", "XLF"),
    Multi3xEtfPairSpec("LABU", "XBI"),
    Multi3xEtfPairSpec("CURE", "XLV"),
    Multi3xEtfPairSpec("NAIL", "ITB"),
    Multi3xEtfPairSpec("RETL", "XRT"),
    Multi3xEtfPairSpec("ERX", "XLE"),
    Multi3xEtfPairSpec("DRN", "XLRE"),
    Multi3xEtfPairSpec("WANT", "XLY"),
)


def _pair_columns(spec: Multi3xEtfPairSpec) -> list[str]:
    etf = spec.etf.lower()
    underlying = spec.underlying.lower()
    return [
        "date",
        *[f"{etf}_{column}" for column in ["open", "high", "low", "close", "volume"]],
        *[f"{underlying}_{column}" for column in ["open", "high", "low", "close", "volume"]],
        "vix",
    ]


def _read_pair_panel(path: Path, spec: Multi3xEtfPairSpec) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_pair_columns(spec))
    return _normalize_pair_panel(pd.read_csv(path), spec)


def _normalize_pair_panel(frame: pd.DataFrame, spec: Multi3xEtfPairSpec) -> pd.DataFrame:
    columns = _pair_columns(spec)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"multi 3x ETF panel missing columns for {spec.etf}: {missing}")
    normalized = frame.loc[:, columns].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize().dt.date.astype("string")
    for column in columns:
        if column != "date":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    required = ["date", f"{spec.etf.lower()}_close", f"{spec.underlying.lower()}_close", "vix"]
    return (
        normalized.dropna(subset=required)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def combine_pair_frames(
    spec: Multi3xEtfPairSpec,
    *,
    etf: pd.DataFrame,
    underlying: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    etf_prefix = spec.etf.lower()
    underlying_prefix = spec.underlying.lower()
    etf_frame = _normalize_ohlcv_frame(etf).rename(
        columns={column: f"{etf_prefix}_{column}" for column in ["open", "high", "low", "close", "volume"]}
    )
    underlying_frame = _normalize_ohlcv_frame(underlying).rename(
        columns={column: f"{underlying_prefix}_{column}" for column in ["open", "high", "low", "close", "volume"]}
    )
    vix_frame = _normalize_vix_frame(vix)
    panel = etf_frame.merge(underlying_frame, on="date", how="inner").merge(vix_frame, on="date", how="inner")
    return _normalize_pair_panel(panel.loc[:, _pair_columns(spec)], spec)


def merge_pair_panel(
    existing: pd.DataFrame,
    fetched: pd.DataFrame,
    spec: Multi3xEtfPairSpec,
    *,
    replace_overlap: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    existing = _normalize_pair_panel(existing, spec) if not existing.empty else pd.DataFrame(columns=_pair_columns(spec))
    fetched = _normalize_pair_panel(fetched, spec) if not fetched.empty else pd.DataFrame(columns=_pair_columns(spec))
    if existing.empty:
        return fetched, int(len(fetched)), 0
    if fetched.empty:
        return existing, 0, 0

    existing_dates = set(existing["date"].astype(str))
    fetched_dates = set(fetched["date"].astype(str))
    appended_rows = len(fetched_dates - existing_dates)
    replaced_rows = len(existing_dates & fetched_dates) if replace_overlap else 0
    if replace_overlap:
        merged = pd.concat([existing, fetched], ignore_index=True).drop_duplicates("date", keep="last")
    else:
        merged = pd.concat([existing, fetched.loc[~fetched["date"].astype(str).isin(existing_dates)]], ignore_index=True)
    return _normalize_pair_panel(merged, spec), int(appended_rows), int(replaced_rows)


def pair_panel_to_qlib_frame(panel: pd.DataFrame, spec: Multi3xEtfPairSpec) -> pd.DataFrame:
    normalized = _normalize_pair_panel(panel, spec)
    rows: list[pd.DataFrame] = []
    for symbol in [spec.etf, spec.underlying]:
        prefix = symbol.lower()
        symbol_frame = normalized[
            [
                "date",
                f"{prefix}_open",
                f"{prefix}_high",
                f"{prefix}_low",
                f"{prefix}_close",
                f"{prefix}_volume",
                "vix",
            ]
        ].rename(
            columns={
                "date": "datetime",
                f"{prefix}_open": "open",
                f"{prefix}_high": "high",
                f"{prefix}_low": "low",
                f"{prefix}_close": "close",
                f"{prefix}_volume": "volume",
            }
        )
        symbol_frame.insert(1, "instrument", symbol)
        rows.append(symbol_frame)
    qlib_frame = pd.concat(rows, ignore_index=True)
    qlib_frame["datetime"] = pd.to_datetime(qlib_frame["datetime"]).dt.normalize()
    return qlib_frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def refresh_pair_qlib_store(
    panel: pd.DataFrame,
    spec: Multi3xEtfPairSpec,
    *,
    paths: Multi3xEtfUpdatePaths,
) -> dict[str, Any]:
    normalized = _normalize_pair_panel(panel, spec)
    qlib_frame = pair_panel_to_qlib_frame(normalized, spec)
    as_of = str(normalized["date"].max())
    provider_uri = paths.qlib_dir / spec.provider_name
    panel_path = paths.panel_dir / f"{spec.panel_name}.csv"
    manifest = build_qlib_store_from_frame(
        qlib_frame,
        provider_uri=provider_uri,
        market="all",
        region="us",
        freq="day",
        instrument_col="instrument",
        datetime_col="datetime",
        field_map=QLIB_FIELD_MAP,
        as_of=as_of,
        source=panel_path.resolve().as_posix(),
    )
    audit = audit_qlib_provider(
        QlibProviderConfig(provider_uri=provider_uri, region="us", market="all", freq="day"),
        instruments=[spec.etf, spec.underlying],
        fields=["$close", "$volume", "$vix"],
        start=str(normalized["date"].min()),
        end=as_of,
    )
    return {"manifest": manifest, "audit": audit}


def _write_etf_raw_symbol(panel: pd.DataFrame, spec: Multi3xEtfPairSpec, *, paths: Multi3xEtfUpdatePaths) -> Path:
    normalized = _normalize_pair_panel(panel, spec)
    prefix = spec.etf.lower()
    symbol_frame = normalized[
        [
            "date",
            f"{prefix}_open",
            f"{prefix}_high",
            f"{prefix}_low",
            f"{prefix}_close",
            f"{prefix}_volume",
        ]
    ].rename(
        columns={
            f"{prefix}_open": "open",
            f"{prefix}_high": "high",
            f"{prefix}_low": "low",
            f"{prefix}_close": "close",
            f"{prefix}_volume": "volume",
        }
    )
    path = paths.raw_symbol_dir / f"{spec.etf}.csv"
    _write_csv(symbol_frame.loc[:, OHLCV_COLUMNS], path)
    return path


def _registry_relative_root(paths: Multi3xEtfUpdatePaths) -> Path:
    registry_path = paths.registry_yaml.resolve()
    return REPO_ROOT if registry_path.is_relative_to(REPO_ROOT) else paths.registry_yaml.parent.resolve()


def update_dataset_registry_for_multi_3x(
    *,
    paths: Multi3xEtfUpdatePaths,
    pairs: Sequence[Multi3xEtfPairSpec] = MULTI_3X_ETF_PAIRS,
) -> dict[str, Any]:
    import yaml

    registry: dict[str, Any]
    if paths.registry_yaml.exists():
        registry = yaml.safe_load(paths.registry_yaml.read_text(encoding="utf-8")) or {}
    else:
        registry = {}
    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    relative_root = _registry_relative_root(paths)
    updated_keys: list[str] = []
    for spec in pairs:
        panel_key = spec.panel_name
        registry.setdefault("processed_panels", {})[panel_key] = _csv_summary(
            paths.panel_dir / f"{panel_key}.csv",
            relative_to=relative_root,
        )
        updated_keys.append(f"processed_panels.{panel_key}")
        registry.setdefault("symbols", {})[spec.etf] = _csv_summary(
            paths.raw_symbol_dir / f"{spec.etf}.csv",
            relative_to=relative_root,
        )
        updated_keys.append(f"symbols.{spec.etf}")
        provider_uri = paths.qlib_dir / spec.provider_name
        provider_path = provider_uri.relative_to(relative_root).as_posix() if provider_uri.exists() else provider_uri.as_posix()
        registry.setdefault("qlib", {})[spec.etf] = {
            "exists": provider_uri.exists(),
            "has_calendar": (provider_uri / "calendars" / "day.txt").exists(),
            "has_instruments": (provider_uri / "instruments" / "all.txt").exists(),
            "path": provider_path,
        }
        updated_keys.append(f"qlib.{spec.etf}")
    paths.registry_yaml.parent.mkdir(parents=True, exist_ok=True)
    paths.registry_yaml.write_text(yaml.safe_dump(registry, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return {
        "registry_path": paths.registry_yaml.resolve().as_posix(),
        "generated_at": registry["generated_at"],
        "updated_keys": updated_keys,
    }


def update_multi_3x_etf_data(
    *,
    paths: Multi3xEtfUpdatePaths | None = None,
    pairs: Sequence[Multi3xEtfPairSpec] = MULTI_3X_ETF_PAIRS,
    start: str | None = None,
    end: str,
    refresh_overlap_days: int = 10,
    replace_overlap: bool = False,
    skip_qlib: bool = False,
    skip_registry: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    _require_settled_us_daily_end(end)
    paths = paths or Multi3xEtfUpdatePaths()
    pair_results: list[dict[str, Any]] = []
    for spec in pairs:
        panel_path = paths.panel_dir / f"{spec.panel_name}.csv"
        existing = _read_pair_panel(panel_path, spec)
        old_last_date = str(existing["date"].max()) if not existing.empty else None
        pair_start = start or _fetch_start_from_existing(
            existing,
            overlap_days=refresh_overlap_days,
            fallback="2008-01-01",
        )
        etf = _fetch_yfinance_compatible_ohlcv(spec.etf, start=pair_start, end=end)
        underlying = _fetch_yfinance_compatible_ohlcv(spec.underlying, start=pair_start, end=end)
        vix, vix_source = _fetch_vix_with_fallback(start=pair_start, end=end)
        fetched = combine_pair_frames(spec, etf=etf, underlying=underlying, vix=vix)
        updated, appended_rows, replaced_rows = merge_pair_panel(
            existing,
            fetched,
            spec,
            replace_overlap=replace_overlap,
        )
        qlib_result = None
        raw_symbol_path = None
        if not dry_run:
            _write_csv(updated, panel_path)
            raw_symbol_path = _write_etf_raw_symbol(updated, spec, paths=paths)
            if not skip_qlib:
                qlib_result = refresh_pair_qlib_store(updated, spec, paths=paths)
        pair_results.append(
            {
                "etf": spec.etf,
                "underlying": spec.underlying,
                "panel_csv": panel_path.resolve().as_posix(),
                "raw_symbol_csv": raw_symbol_path.resolve().as_posix() if raw_symbol_path is not None else None,
                "provider_uri": (paths.qlib_dir / spec.provider_name).resolve().as_posix(),
                "source_api": [
                    "yfinance.Ticker.history",
                    "query2/query1.finance.yahoo.com/v8/finance/chart",
                    "polygon.io/v2/aggs",
                    vix_source,
                ],
                "start": pair_start,
                "end": end,
                "old_last_date": old_last_date,
                "new_last_date": str(updated["date"].max()) if not updated.empty else None,
                "fetched_rows": int(len(fetched)),
                "total_rows": int(len(updated)),
                "appended_rows": int(appended_rows),
                "replaced_overlap_rows": int(replaced_rows),
                "replace_overlap": bool(replace_overlap),
                "dry_run": bool(dry_run),
                "qlib": qlib_result,
            }
        )
    registry_result = None
    if not dry_run and not skip_registry:
        registry_result = update_dataset_registry_for_multi_3x(paths=paths, pairs=pairs)
    return {
        "operation": "multi_3x_etf_data_update",
        "end": end,
        "pair_results": pair_results,
        "registry": registry_result,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh multi 3x ETF qlib panels and providers.")
    parser.add_argument("--start", help="Inclusive fetch start date. Defaults to each panel last date minus overlap.")
    parser.add_argument("--end", required=True, help="Inclusive fetch end date.")
    parser.add_argument("--refresh-overlap-days", type=int, default=10)
    parser.add_argument("--replace-overlap", action="store_true")
    parser.add_argument("--skip-qlib", action="store_true")
    parser.add_argument("--skip-registry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = update_multi_3x_etf_data(
        start=args.start,
        end=args.end,
        refresh_overlap_days=args.refresh_overlap_days,
        replace_overlap=args.replace_overlap,
        skip_qlib=args.skip_qlib,
        skip_registry=args.skip_registry,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
