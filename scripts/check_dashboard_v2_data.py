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


def _require_int_field(payload: dict[str, Any], field: str, label: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} {field} must be a non-negative integer")
    return value


def _check_strict_retention_gate(manifest: dict[str, Any]) -> None:
    gate = manifest.get("strict_retention_gate")
    if gate is None:
        return
    if not isinstance(gate, dict):
        raise ValueError("strict_retention_gate must be an object")
    if gate.get("status") not in {"passed", "blocked"}:
        raise ValueError("strict_retention_gate status must be passed or blocked")
    profiles_scanned = _require_int_field(gate, "profiles_scanned", "strict_retention_gate")
    passing_count = _require_int_field(gate, "passing_profile_count", "strict_retention_gate")
    if passing_count > profiles_scanned:
        raise ValueError("strict_retention_gate passing_profile_count cannot exceed profiles_scanned")
    if "max_walkforward_drawdown_pct" in gate:
        _require_numeric(gate, ("max_walkforward_drawdown_pct",), "strict_retention_gate")
    if "required_best_return_pct" in gate:
        _require_numeric(gate, ("required_best_return_pct",), "strict_retention_gate")


def _check_strict_candidate_retention_filter(manifest: dict[str, Any]) -> None:
    filter_payload = manifest.get("strict_candidate_retention_filter")
    if filter_payload is None:
        return
    if not isinstance(filter_payload, dict):
        raise ValueError("strict_candidate_retention_filter must be an object")
    if filter_payload.get("status") not in {"passed", "blocked"}:
        raise ValueError("strict_candidate_retention_filter status must be passed or blocked")
    promotable_count = _require_int_field(
        filter_payload,
        "promotable_profile_count",
        "strict_candidate_retention_filter",
    )
    actor_retention_count = _require_int_field(
        filter_payload,
        "actor_retention_allowed_count",
        "strict_candidate_retention_filter",
    )
    _require_int_field(filter_payload, "passing_profile_count", "strict_candidate_retention_filter")
    _require_int_field(filter_payload, "reference_visible_count", "strict_candidate_retention_filter")
    if actor_retention_count > promotable_count:
        raise ValueError(
            "strict_candidate_retention_filter actor_retention_allowed_count cannot exceed promotable_profile_count"
        )


def _check_final_walkforward_retention_contract(manifest: dict[str, Any]) -> None:
    contract = manifest.get("final_walkforward_retention_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError("final_walkforward_retention_contract must be an object")
    if contract.get("status") not in {"passed", "blocked"}:
        raise ValueError("final_walkforward_retention_contract status must be passed or blocked")
    final_retained_count = _require_int_field(
        contract,
        "final_retained_count",
        "final_walkforward_retention_contract",
    )
    strict_passing_count = _require_int_field(
        contract,
        "strict_passing_profile_count",
        "final_walkforward_retention_contract",
    )
    actor_retention_count = _require_int_field(
        contract,
        "candidate_filter_actor_retention_allowed_count",
        "final_walkforward_retention_contract",
    )
    if final_retained_count > actor_retention_count:
        raise ValueError(
            "final_walkforward_retention_contract final_retained_count cannot exceed actor retention count"
        )
    if final_retained_count > strict_passing_count:
        raise ValueError(
            "final_walkforward_retention_contract final_retained_count cannot exceed strict passing count"
        )
    if "reference_return_pct" in contract:
        _require_numeric(contract, ("reference_return_pct",), "final_walkforward_retention_contract")
    if "latest_reference_return_pct" in contract:
        _require_numeric(contract, ("latest_reference_return_pct",), "final_walkforward_retention_contract")
    if "required_best_return_pct" in contract:
        _require_numeric(contract, ("required_best_return_pct",), "final_walkforward_retention_contract")


def _check_latest_signal_differences(
    manifest: dict[str, Any],
    comparison: dict[str, Any],
    profiles: list[str],
) -> None:
    latest_signal = manifest.get("latest_signal")
    differences = comparison.get("latest_signal_differences")
    if latest_signal is not None:
        if not isinstance(latest_signal, dict):
            raise ValueError("manifest latest_signal must be an object")
        if latest_signal.get("comparison_included") and int(latest_signal.get("available_count") or 0) > 0:
            if not isinstance(differences, list) or not differences:
                raise ValueError("comparison latest_signal_differences must be a non-empty list")
    if differences is None:
        return
    if not isinstance(differences, list):
        raise ValueError("comparison latest_signal_differences must be a list")
    baseline_profile = comparison.get("baseline_profile")
    profile_set = set(profiles)
    required = {
        "date",
        "profile",
        "baseline_profile",
        "model_action",
        "baseline_action",
        "model_position",
        "baseline_position",
        "position_delta_vs_baseline",
    }
    for row in differences:
        if not isinstance(row, dict):
            raise ValueError("comparison latest_signal_differences rows must contain JSON objects")
        _require_fields(row, required, "latest_signal_differences row")
        if row["profile"] not in profile_set:
            raise ValueError(f"latest_signal_differences row references unknown profile: {row['profile']}")
        if row["baseline_profile"] != baseline_profile:
            raise ValueError("latest_signal_differences row baseline_profile must match comparison baseline_profile")
        if row["model_action"] not in ACTION_VALUES:
            raise ValueError("latest_signal_differences row model_action must be BUY, SELL, or HOLD")
        if row["baseline_action"] not in ACTION_VALUES:
            raise ValueError("latest_signal_differences row baseline_action must be BUY, SELL, or HOLD")
        _require_numeric(
            row,
            ("model_position", "baseline_position", "position_delta_vs_baseline"),
            "latest_signal_differences row",
        )


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
    available_profiles = set(manifest.get("available_profiles") or [])
    unknown_defaults = sorted(profile for profile in profiles if profile not in available_profiles)
    if unknown_defaults:
        raise ValueError(f"manifest references unknown profile: {', '.join(unknown_defaults)}")
    _check_strict_retention_gate(manifest)
    _check_strict_candidate_retention_filter(manifest)
    _check_final_walkforward_retention_contract(manifest)
    _check_latest_signal_differences(manifest, comparison, profiles)

    market_rows = _require_rows(market, "market SOXL")
    for row in market_rows:
        _require_fields(row, REQUIRED_MARKET_FIELDS, "market row")
        _require_numeric(row, ("open", "high", "low", "close", "volume"), "market row")
    _require_unique_dates(market_rows, "market rows")

    model_dates_by_profile: dict[str, set[str]] = {}
    for profile in profiles:
        model = _load(root / "models" / f"{profile}.json")
        rows = _require_rows(model, f"model {profile}")
        for row in rows:
            _require_fields(row, REQUIRED_MODEL_FIELDS, f"model {profile} row")
            _require_numeric(row, ("strategy_nav", "buyhold_nav", "position", "price"), f"model {profile} row")
            if row["action"] not in ACTION_VALUES:
                raise ValueError(f"model {profile} row action must be BUY, SELL, or HOLD")
        _require_unique_dates(rows, f"model {profile} rows")
        if not isinstance(model.get("events"), list):
            raise ValueError(f"model {profile} events must be a list")
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
        unknown = sorted(profile for profile in profile_values if profile not in profiles)
        if unknown:
            raise ValueError(f"comparison row references unknown profile: {', '.join(unknown)}")
        missing = sorted(profile for profile in profiles if profile not in profile_values)
        if missing:
            raise ValueError(f"comparison row missing compared profile: {', '.join(missing)}")
        for profile in profiles:
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
