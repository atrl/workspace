#!/usr/bin/env python3
from __future__ import annotations

"""Normalize memory notes and rebuild the machine-readable index."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.memory_tools import build_index


def main() -> int:
    records = build_index()
    print(f"[done] normalized {len(records)} memory notes")
    print("       wrote memory/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
