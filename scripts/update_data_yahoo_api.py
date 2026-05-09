#!/usr/bin/env python3
"""Compatibility wrapper for the canonical market-data updater.

Preserves the historical entrypoint name while delegating to
``scripts/update_market_data.py --provider yahoo_api`` so raw mirrors and qlib
stay in sync with the canonical processed panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_market_data import main as update_market_data_main


def main() -> None:
    update_market_data_main(["--provider", "yahoo_api", *sys.argv[1:]])


if __name__ == "__main__":
    main()
