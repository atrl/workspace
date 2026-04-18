#!/usr/bin/env python3
from __future__ import annotations

"""Manifest-driven cleanup for superseded model artifacts."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models"
MANIFEST_PATH = ROOT / "artifacts" / "retention_manifest.json"
MEMORY_INDEX = ROOT / "memory" / "index.json"
WORKPAD_PATH = ROOT / "codex_exec_plans" / "active" / "2026-04-10-soxl-agent-loop.md"

CANONICAL_KEEP = {
    "soxl_balanced_actor.pt",
    "soxl_balanced_summary.json",
    "soxl_smoke_actor.pt",
    "soxl_smoke_summary.json",
    "soxl_balanced_wf_2020_actor.pt",
    "soxl_balanced_wf_2020_summary.json",
    "soxl_balanced_wf_2022_actor.pt",
    "soxl_balanced_wf_2022_summary.json",
    "soxl_balanced_wf_2023_2024_actor.pt",
    "soxl_balanced_wf_2023_2024_summary.json",
}

DELETE_STEMS = {
    "soxl_balanced_regime_gate_actor.pt",
    "soxl_balanced_regime_gate_v2_actor.pt",
    "soxl_balanced_stress_behavior_actor.pt",
    "soxl_balanced_stress_release_delta_actor.pt",
    "soxl_balanced_stress_select_actor.pt",
    "soxl_balanced_transition_state_actor.pt",
    "soxl_balanced_floor40_sym_bear30_actor.pt",
    "soxl_balanced_curvepow2_actor.pt",
    "soxl_balanced_actor_seed42.pt",
    "soxl_smoke_actor_floor04.pt",
    "soxl_smoke_actor_floor05.pt",
    "soxl_smoke_actor_run1.pt",
    "soxl_smoke_actor_run2.pt",
    "soxl_smoke_actor_start010.pt",
    "soxl_smoke_actor_start012.pt",
    "soxl_smoke_actor_start015.pt",
    "soxl_smoke_wf_2020_actor.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or apply artifact retention manifest.")
    parser.add_argument("--apply", action="store_true", help="delete files marked delete")
    return parser.parse_args()


def referenced_basenames() -> set[str]:
    refs = set(CANONICAL_KEEP)
    sources = [WORKPAD_PATH]
    if MEMORY_INDEX.exists():
        index = json.loads(MEMORY_INDEX.read_text(encoding="utf-8"))
        for record in index:
            for artifact in record.get("artifacts", []):
                refs.add(Path(artifact).name)
    for path in sources:
        text = path.read_text(encoding="utf-8")
        refs.update(re.findall(r"artifacts/models/([A-Za-z0-9._-]+)", text))
    return refs


def build_manifest() -> dict[str, Any]:
    referenced = referenced_basenames()
    entries = []
    for path in sorted(MODEL_DIR.glob("*")):
        basename = path.name
        reason = "canonical keep"
        action = "keep"
        if basename in referenced:
            action = "keep"
            reason = "referenced by memory/workpad or canonical set"
        elif basename in DELETE_STEMS:
            action = "delete"
            reason = "superseded actor or redundant sweep output"
        elif basename.endswith("_actor.pt") and any(
            token in basename
            for token in (
                "regime_gate",
                "stress_behavior",
                "stress_release_delta",
                "stress_select",
                "transition_state",
                "floor40_sym_bear30",
            )
        ):
            action = "delete"
            reason = "dropped branch actor"
        elif basename.endswith(".json"):
            action = "keep"
            reason = "summary retained as evidence"
        entries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "action": action,
                "reason": reason,
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {"generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z", "entries": entries}
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
        freed += path.stat().st_size
        path.unlink()
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
