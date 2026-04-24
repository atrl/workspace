#!/usr/bin/env python3
from __future__ import annotations

"""Build the generated evidence index for research artifacts."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"
INDEX_PATH = EVIDENCE_DIR / "index.json"
INDEXED_SUFFIXES = {".csv", ".html", ".json", ".md", ".png", ".txt"}


def evidence_type(path: Path) -> str:
    relative = path.relative_to(EVIDENCE_DIR)
    return relative.parts[0] if len(relative.parts) > 1 else "root"


def build_index() -> dict:
    entries = []
    for path in sorted(EVIDENCE_DIR.rglob("*")):
        if not path.is_file() or path == INDEX_PATH or path.suffix.lower() not in INDEXED_SUFFIXES:
            continue
        entries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "evidence_type": evidence_type(path),
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if evidence/index.json is stale")
    return parser.parse_args()


def main() -> int:
    payload = build_index()
    rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if parse_args().check:
        current = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else {}
        if current.get("entries") != payload["entries"]:
            print("evidence/index.json is stale; run scripts/build_evidence_index.py")
            return 1
        print("evidence index OK")
        return 0
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {INDEX_PATH.relative_to(ROOT)} ({len(payload['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
