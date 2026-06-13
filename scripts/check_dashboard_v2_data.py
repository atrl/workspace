#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_MARKET_FIELDS = {"date", "open", "high", "low", "close", "volume"}
REQUIRED_MODEL_FIELDS = {"date", "strategy_nav", "buyhold_nav", "position", "action", "price"}
ACTION_VALUES = {"BUY", "SELL", "HOLD"}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing dashboard v2 file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dashboard v2 file must contain a JSON object: {path}")
    return payload


def _require_fields(row: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(field for field in required if field not in row)
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(missing)}")


def _require_unique_dates(rows: list[dict[str, Any]], label: str) -> None:
    dates = [row.get("date") for row in rows]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{label} contains duplicate dates")


def _require_numeric(row: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    for field in fields:
        try:
            float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} field {field} must be numeric") from exc


def _require_rows(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} rows must be a non-empty list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must contain JSON objects")
    return rows


def check_dashboard_v2_data(root: Path) -> dict[str, int]:
    manifest = _load(root / "manifest.json")
    market = _load(root / "market" / "SOXL.json")
    comparison = _load(root / "comparison" / "default.json")

    profiles = manifest.get("default_comparison_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("manifest default_comparison_profiles must be a non-empty list")
    auto_update = manifest.get("auto_update")
    if not isinstance(auto_update, dict):
        raise ValueError("manifest auto_update must be an object")
    if not isinstance(auto_update.get("tasks"), list):
        raise ValueError("manifest auto_update tasks must be a list")
    available_profiles = set(manifest.get("available_profiles") or [])
    unknown_defaults = sorted(profile for profile in profiles if profile not in available_profiles)
    if unknown_defaults:
        raise ValueError(f"manifest references unknown profile: {', '.join(unknown_defaults)}")
    comparison_profiles = comparison.get("compared_profiles")
    if comparison_profiles is None:
        comparison_profiles = profiles
    if not isinstance(comparison_profiles, list) or not comparison_profiles:
        raise ValueError("comparison compared_profiles must be a non-empty list")
    unknown_comparison = sorted(profile for profile in comparison_profiles if profile not in available_profiles)
    if unknown_comparison:
        raise ValueError(f"comparison references unknown profile: {', '.join(unknown_comparison)}")
    profiles_to_check = list(dict.fromkeys([*profiles, *comparison_profiles]))

    market_rows = _require_rows(market, "market SOXL")
    for row in market_rows:
        _require_fields(row, REQUIRED_MARKET_FIELDS, "market row")
        _require_numeric(row, ("open", "high", "low", "close", "volume"), "market row")
    _require_unique_dates(market_rows, "market rows")

    model_dates_by_profile: dict[str, set[str]] = {}
    for profile in profiles_to_check:
        model = _load(root / "models" / f"{profile}.json")
        if not isinstance(model.get("feature_columns"), list) or not model["feature_columns"]:
            raise ValueError(f"model {profile} feature_columns must be a non-empty list")
        if not isinstance(model.get("feature_packs"), list):
            raise ValueError(f"model {profile} feature_packs must be a list")
        if not isinstance(model.get("snapshot_columns"), list):
            raise ValueError(f"model {profile} snapshot_columns must be a list")
        rows = _require_rows(model, f"model {profile}")
        for row in rows:
            _require_fields(row, REQUIRED_MODEL_FIELDS, f"model {profile} row")
            _require_numeric(row, ("strategy_nav", "buyhold_nav", "position", "price"), f"model {profile} row")
            if row["action"] not in ACTION_VALUES:
                raise ValueError(f"model {profile} row action must be BUY, SELL, or HOLD")
        _require_unique_dates(rows, f"model {profile} rows")
        events = model.get("events")
        if not isinstance(events, list):
            raise ValueError(f"model {profile} events must be a list")
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"model {profile} events must contain JSON objects")
            if event.get("kind") in {"BUY", "SELL"} and not isinstance(event.get("feature_snapshot"), dict):
                raise ValueError(f"model {profile} buy/sell event missing feature_snapshot")
        model_dates_by_profile[profile] = {str(row["date"]) for row in rows}

    comparison_rows = comparison.get("rows")
    if not isinstance(comparison_rows, list):
        raise ValueError("comparison rows must be a list")
    for row in comparison_rows:
        if not isinstance(row, dict):
            raise ValueError("comparison rows must contain JSON objects")
        if "date" not in row:
            raise ValueError("comparison row missing fields: date")
        profile_values = row.get("profiles")
        if not isinstance(profile_values, dict):
            raise ValueError("comparison row profiles must be an object")
        unknown = sorted(profile for profile in profile_values if profile not in comparison_profiles)
        if unknown:
            raise ValueError(f"comparison row references unknown profile: {', '.join(unknown)}")
        missing = sorted(profile for profile in comparison_profiles if profile not in profile_values)
        if missing:
            raise ValueError(f"comparison row missing compared profile: {', '.join(missing)}")
        for profile in comparison_profiles:
            if row["date"] not in model_dates_by_profile[profile]:
                raise ValueError(f"comparison row date missing from model {profile}: {row['date']}")

    return {"profiles": len(profiles), "market_rows": len(market_rows), "comparison_rows": len(comparison_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dashboard v2 data contract.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("dashboard_data/v2"))
    args = parser.parse_args()
    result = check_dashboard_v2_data(args.root)
    print(
        "dashboard_v2_check=ok "
        f"profiles={result['profiles']} "
        f"market_rows={result['market_rows']} "
        f"comparison_rows={result['comparison_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
