#!/usr/bin/env python3
from __future__ import annotations

"""Build the generated runtime index for category-organized research artifacts."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime"
EXPERIMENTS_DIR = ROOT / "experiments"
INDEX_PATH = RUNTIME_DIR / "index.json"
INDEXED_SUFFIXES = {".csv", ".html", ".json", ".md", ".txt"}
SKIP_DIRS = {"soxl"}


def runtime_category(path: Path) -> str:
    try:
        relative = path.relative_to(RUNTIME_DIR)
        return relative.parts[0] if len(relative.parts) > 1 else "root"
    except ValueError:
        pass
    try:
        relative = path.relative_to(EXPERIMENTS_DIR)
        return f"experiments/{relative.parts[0]}" if len(relative.parts) > 1 else "experiments"
    except ValueError:
        return "other"


def _scan_directory(base_dir: Path, entries: list, root: Path) -> None:
    for path in sorted(base_dir.rglob("*")):
        relative = path.relative_to(base_dir)
        if relative.parts and relative.parts[0] in SKIP_DIRS:
            continue
        if not path.is_file() or path.suffix.lower() not in INDEXED_SUFFIXES:
            continue
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "category": runtime_category(path),
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }
        )


def build_index() -> dict:
    entries = []
    _scan_directory(RUNTIME_DIR, entries, ROOT)
    _scan_directory(EXPERIMENTS_DIR, entries, ROOT)
    # De-duplicate by path
    seen = set()
    unique_entries = []
    for e in entries:
        if e["path"] not in seen:
            seen.add(e["path"])
            unique_entries.append(e)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": unique_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if runtime/index.json is stale")
    return parser.parse_args()


def main() -> int:
    payload = build_index()
    rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if parse_args().check:
        current = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else {}
        if current.get("entries") != payload["entries"]:
            print("runtime/index.json is stale; run scripts/build_runtime_index.py")
            return 1
        print("runtime index OK")
        return 0
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {INDEX_PATH.relative_to(ROOT)} ({len(payload['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
