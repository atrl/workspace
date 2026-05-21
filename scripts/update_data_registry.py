#!/usr/bin/env python3
"""Build or check the data registry for raw, processed, qlib, and cache assets."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REGISTRY_PATH = DATA / "datasets.yaml"


def csv_metadata(path: Path) -> dict:
    meta = {"path": str(path.relative_to(ROOT)), "exists": path.exists()}
    if not path.exists():
        return meta
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = 0
        first_date = None
        last_date = None
        for row in reader:
            rows += 1
            value = row.get("date") or row.get("datetime")
            if value:
                first_date = first_date or value[:10]
                last_date = value[:10]
    meta.update({"rows": rows, "start": first_date, "end": last_date})
    return meta


def qlib_metadata(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "has_calendar": (path / "calendars" / "day.txt").exists(),
        "has_instruments": (path / "instruments" / "all.txt").exists(),
    }


def build_registry() -> dict:
    symbols = {path.stem.upper(): csv_metadata(path) for path in sorted((DATA / "raw" / "symbols").glob("*.csv"))}
    external = {path.stem: csv_metadata(path) for path in sorted((DATA / "raw" / "external").glob("*.csv"))}
    panels = {path.stem: csv_metadata(path) for path in sorted((DATA / "processed" / "panels").glob("*.csv"))}
    qlib = {path.name.upper(): qlib_metadata(path) for path in sorted((DATA / "qlib").iterdir()) if path.is_dir()}
    training = {
        path.stem: {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "bytes": path.stat().st_size}
        for path in sorted((DATA / "cache" / "training").glob("*.pkl"))
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layout": {
            "raw_symbols": "data/raw/symbols/<TICKER>.csv",
            "raw_external": "data/raw/external/<NAME>.csv",
            "processed_panels": "data/processed/panels/<asset>_with_underlying_vix.csv",
            "qlib": "data/qlib/<asset>/",
            "training_cache": "data/cache/training/<profile>.pkl",
        },
        "symbols": symbols,
        "external": external,
        "processed_panels": panels,
        "qlib": qlib,
        "training_cache": training,
    }


def _compare_current_metadata(registry: dict, current_registry: dict) -> list[str]:
    failures: list[str] = []
    comparable_keys = {
        "symbols": ("path", "exists", "rows", "start", "end"),
        "external": ("path", "exists", "rows", "start", "end"),
        "processed_panels": ("path", "exists", "rows", "start", "end"),
        "qlib": ("path", "exists", "has_calendar", "has_instruments"),
        "training_cache": ("path", "exists", "bytes"),
    }
    for section, keys in comparable_keys.items():
        current_section = current_registry.get(section, {}) or {}
        stored_section = registry.get(section, {}) or {}
        for name, current_meta in current_section.items():
            if name not in stored_section:
                failures.append(f"stale registry {section}.{name}: missing current entry")
                continue
            stored_meta = stored_section.get(name, {}) or {}
            for key in keys:
                stored_value = stored_meta.get(key)
                current_value = current_meta.get(key)
                if stored_value != current_value:
                    failures.append(
                        f"stale registry {section}.{name}.{key}: "
                        f"stored={stored_value} current={current_value}"
                    )
    return failures


def check_registry(registry: dict, current_registry: dict | None = None) -> list[str]:
    failures: list[str] = []
    for section in ["symbols", "processed_panels", "qlib"]:
        if not registry.get(section):
            failures.append(f"missing or empty section: {section}")
    for name in ["SOXL", "TQQQ", "UPRO", "SPXL", "TECL"]:
        if name not in registry.get("symbols", {}):
            failures.append(f"missing raw symbol CSV for {name}")
    if "soxl_soxx_vix_full" not in registry.get("processed_panels", {}):
        failures.append("missing canonical SOXL/SOXX/VIX processed panel")
    if "SOXL" not in registry.get("qlib", {}):
        failures.append("missing qlib store for SOXL")
    if current_registry is not None:
        failures.extend(_compare_current_metadata(registry, current_registry))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate current data/datasets.yaml without rewriting it")
    args = parser.parse_args()
    if args.check:
        if not REGISTRY_PATH.exists():
            print("update_data_registry: FAIL missing data/datasets.yaml")
            return 1
        registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
        current_registry = build_registry()
    else:
        registry = build_registry()
        REGISTRY_PATH.write_text(yaml.safe_dump(registry, sort_keys=True, allow_unicode=True), encoding="utf-8")
        current_registry = None
    failures = check_registry(registry, current_registry=current_registry)
    if failures:
        print("update_data_registry: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        "update_data_registry: OK "
        f"symbols={len(registry.get('symbols', {}))} "
        f"panels={len(registry.get('processed_panels', {}))} "
        f"qlib={len(registry.get('qlib', {}))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
