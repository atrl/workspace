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
    available_profile_list = manifest.get("available_profiles") or []
    if not isinstance(available_profile_list, list) or not available_profile_list:
        raise ValueError("manifest available_profiles must be a non-empty list")
    available_profiles = set(available_profile_list)
    unknown_defaults = sorted(profile for profile in profiles if profile not in available_profiles)
    if unknown_defaults:
        raise ValueError(f"manifest references unknown profile: {', '.join(unknown_defaults)}")
    default_chart_profile = manifest.get("default_chart_profile")
    if default_chart_profile is not None and default_chart_profile not in available_profiles:
        raise ValueError(f"manifest default_chart_profile is unknown: {default_chart_profile}")
    has_retention_gate = manifest.get("model_retention_gate") is not None

    market_rows = _require_rows(market, "market SOXL")
    for row in market_rows:
        _require_fields(row, REQUIRED_MARKET_FIELDS, "market row")
        _require_numeric(row, ("open", "high", "low", "close", "volume"), "market row")
    _require_unique_dates(market_rows, "market rows")

    model_dates_by_profile: dict[str, set[str]] = {}
    for profile in available_profile_list:
        model = _load(root / "models" / f"{profile}.json")
        rows = _require_rows(model, f"model {profile}")
        for field in (
            "source_kind",
            "display_name",
            "runnable",
            "feature_columns",
            "feature_schema_signature",
            "snapshot_columns",
            "feature_action_summary",
        ):
            if field not in model:
                raise ValueError(f"model {profile} missing metadata field: {field}")
        if not isinstance(model["feature_action_summary"], list):
            raise ValueError(f"model {profile} feature_action_summary must be a list")
        if has_retention_gate:
            for field in ("retention_gate_pass", "retention_decision", "retention_failures"):
                if field not in model:
                    raise ValueError(f"model {profile} missing retention gate field: {field}")
            if not isinstance(model["retention_failures"], list):
                raise ValueError(f"model {profile} retention_failures must be a list")
        for row in rows:
            _require_fields(row, REQUIRED_MODEL_FIELDS, f"model {profile} row")
            _require_numeric(row, ("strategy_nav", "buyhold_nav", "position", "price"), f"model {profile} row")
            if row["action"] not in ACTION_VALUES:
                raise ValueError(f"model {profile} row action must be BUY, SELL, or HOLD")
        _require_unique_dates(rows, f"model {profile} rows")
        if not isinstance(model.get("events"), list):
            raise ValueError(f"model {profile} events must be a list")
        model_dates_by_profile[profile] = {str(row["date"]) for row in rows}

    compared_profiles = comparison.get("compared_profiles")
    if not isinstance(compared_profiles, list) or not compared_profiles:
        raise ValueError("comparison compared_profiles must be a non-empty list")
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
        unknown = sorted(profile for profile in profile_values if profile not in available_profiles)
        if unknown:
            raise ValueError(f"comparison row references unknown profile: {', '.join(unknown)}")
        missing = sorted(profile for profile in compared_profiles if profile not in profile_values)
        if missing:
            raise ValueError(f"comparison row missing compared profile: {', '.join(missing)}")
        for profile in compared_profiles:
            if row["date"] not in model_dates_by_profile[profile]:
                raise ValueError(f"comparison row date missing from model {profile}: {row['date']}")

    return {"profiles": len(available_profile_list), "market_rows": len(market_rows), "comparison_rows": len(comparison_rows)}


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
