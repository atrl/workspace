from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import PROFILE_DIR, list_profile_names

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "runtime" / "harness" / "profile_registry.yaml"


@dataclass(frozen=True)
class ProfileRegistryEntry:
    name: str
    family: str
    status: str
    baseline: str
    keep_actor: bool
    keep_summary: bool
    notes_ref: str


def load_profile_registry() -> dict[str, ProfileRegistryEntry]:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    entries = payload.get("profiles", [])
    registry: dict[str, ProfileRegistryEntry] = {}
    for item in entries:
        entry = ProfileRegistryEntry(
            name=str(item["name"]),
            family=str(item.get("family", "")),
            status=str(item.get("status", "candidate")),
            baseline=str(item.get("baseline", "")),
            keep_actor=bool(item.get("keep_actor", False)),
            keep_summary=bool(item.get("keep_summary", True)),
            notes_ref=str(item.get("notes_ref", "")),
        )
        registry[entry.name] = entry
    return registry


def validate_profile_registry() -> list[str]:
    registry = load_profile_registry()
    issues: list[str] = []
    configured = set(list_profile_names())
    registered = set(registry)
    missing = sorted(configured - registered)
    extra = sorted(registered - configured)
    if missing:
        issues.append(f"missing registry rows: {', '.join(missing)}")
    if extra:
        issues.append(f"registry rows without config: {', '.join(extra)}")
    canonical_by_family: dict[str, list[str]] = {}
    for entry in registry.values():
        if entry.status == "canonical":
            canonical_by_family.setdefault(entry.family, []).append(entry.name)
    duplicates = {family: names for family, names in canonical_by_family.items() if len(names) > 1}
    if duplicates:
        issues.append(
            "multiple canonical profiles: "
            + "; ".join(f"{family}={','.join(names)}" for family, names in sorted(duplicates.items()))
        )
    for entry in registry.values():
        if entry.status == "dropped" and entry.keep_actor:
            issues.append(f"dropped profile keeps actor: {entry.name}")
    return issues


def registry_payload() -> dict[str, Any]:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
