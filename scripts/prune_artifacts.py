#!/usr/bin/env python3
from __future__ import annotations

"""Registry-driven cleanup for stale artifacts and obvious repo noise."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models"
MANIFEST_PATH = ROOT / "artifacts" / "retention_manifest.json"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.registry import load_profile_registry
from src.soxl_quant.settings import artifact_stem

NOISE_PATHS = (
    ROOT / "runtime" / ".DS_Store",
    ROOT / "__pycache__",
    ROOT / "src" / "soxl_quant" / "__pycache__",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or apply artifact retention manifest.")
    parser.add_argument("--apply", action="store_true", help="delete files marked delete")
    return parser.parse_args()


def _registry_entry_for_name(profile_name: str, registry: dict[str, Any]) -> Any | None:
    return registry.get(profile_name)


def _profile_name_from_path(path: Path, *, allow_summary_namespace: bool) -> str | None:
    name = path.name
    if not name.startswith("soxl_"):
        return None
    stem = name.removeprefix("soxl_")
    for suffix in ("_actor.pt", "_summary.json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Check experiments/ first (preferred location)
    experiment_matches = sorted(ROOT.glob(f"experiments/*/{stem}/profile.yaml"))
    if experiment_matches:
        return stem
    # Fallback to legacy runtime/configs/
    exact = ROOT / "configs" / "soxl" / f"{stem}.yaml"
    if exact.exists():
        return stem
    if stem.startswith("balanced_wf_"):
        return "balanced"
    if not allow_summary_namespace:
        return None
    parts = stem.split("_")
    for idx in range(len(parts), 1, -1):
        candidate = "_".join(parts[:idx])
        experiment_matches = sorted(ROOT.glob(f"experiments/*/{candidate}/profile.yaml"))
        if experiment_matches:
            return candidate
        if (ROOT / "configs" / "soxl" / f"{candidate}.yaml").exists():
            return candidate
    return None


def build_manifest() -> dict[str, Any]:
    registry = load_profile_registry()
    entries: list[dict[str, Any]] = []
    for path in sorted(MODEL_DIR.glob("*")):
        action = "keep"
        reason = "summary retained as evidence"
        profile_name = _profile_name_from_path(path, allow_summary_namespace=path.name.endswith("_summary.json"))
        entry = _registry_entry_for_name(profile_name, registry) if profile_name else None
        if path.name.endswith("_actor.pt"):
            if entry is None:
                action = "delete"
                reason = "actor has no registered profile"
            elif entry.keep_actor:
                action = "keep"
                reason = f"registry keep_actor for {entry.status} profile"
            else:
                action = "delete"
                reason = f"registry drop actor for {entry.status} profile"
        elif path.name.endswith("_summary.json"):
            if entry is None:
                action = "keep"
                reason = "summary retained even without registry match"
            elif entry.keep_summary:
                action = "keep"
                reason = f"registry keep_summary for {entry.status} profile"
            else:
                action = "delete"
                reason = f"registry drop summary for {entry.status} profile"
        entries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "action": action,
                "reason": reason,
                "size_bytes": path.stat().st_size,
            }
        )
    for noise_path in NOISE_PATHS:
        if not noise_path.exists():
            continue
        size_bytes = noise_path.stat().st_size if noise_path.is_file() else 0
        entries.append(
            {
                "path": str(noise_path.relative_to(ROOT)),
                "action": "delete",
                "reason": "generated cache or filesystem noise",
                "size_bytes": size_bytes,
            }
        )
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "entries": entries}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def apply_manifest(manifest: dict[str, Any]) -> tuple[int, int]:
    deleted = 0
    freed = 0
    for entry in manifest["entries"]:
        if entry["action"] != "delete":
            continue
        path = ROOT / entry["path"]
        if not path.exists():
            continue
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
            deleted += 1
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    freed += child.stat().st_size
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()
            deleted += 1
    return deleted, freed


def main() -> int:
    args = parse_args()
    manifest = build_manifest()
    delete_count = sum(1 for entry in manifest["entries"] if entry["action"] == "delete")
    keep_count = sum(1 for entry in manifest["entries"] if entry["action"] == "keep")
    print(f"[manifest] keep={keep_count} delete={delete_count}")
    print(f"           wrote {MANIFEST_PATH.relative_to(ROOT)}")
    if not args.apply:
        return 0
    deleted, freed = apply_manifest(manifest)
    print(f"[apply] deleted={deleted} freed_bytes={freed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
