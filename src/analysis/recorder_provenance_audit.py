"""Audit that every card's declared qlib recorder actually exists in the repo.

A card whose ``summary.json`` names a ``recorder_id`` that is not present in the
worktree presents unverifiable evidence. Production promotions must not rely on
such a recorder. This module walks card summaries, resolves the declared
recorder against the card-local recorder store, and reports any gap.

It is read-only and does not resolve, start, or modify any recorder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CARD_ROOTS = ("research", "production")
SUMMARY_NAME = "summary.json"
RECORDER_DIR = "qlib_recorder"


def _declared_recorder(summary: dict) -> str | None:
    for key in ("recorder_id", "source_recorder_id", "stage1_recorder_id"):
        value = summary.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def audit_card(card_dir: Path) -> dict:
    summary_path = card_dir / SUMMARY_NAME
    if not summary_path.is_file():
        return {"card": card_dir.name, "status": "no_summary"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"card": card_dir.name, "status": "unreadable_summary"}
    recorder_id = _declared_recorder(summary)
    if not recorder_id:
        return {"card": card_dir.name, "status": "no_declared_recorder"}
    recorder_root = card_dir / RECORDER_DIR
    present = recorder_root.is_dir() and any(recorder_root.glob(f"*/{recorder_id}"))
    return {
        "card": card_dir.name,
        "status": "present" if present else "missing_recorder",
        "recorder_id": recorder_id,
        "recorder_root": str(recorder_root.relative_to(card_dir.parent.parent))
        if recorder_root.is_dir() else None,
    }


def audit_tree(root: Path) -> pd.DataFrame:
    rows = []
    for card_root in CARD_ROOTS:
        base = root / card_root
        if not base.is_dir():
            continue
        for card_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            row = audit_card(card_dir)
            row["card_root"] = card_root
            rows.append(row)
    return pd.DataFrame(rows)


def missing_recorders(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["status"].eq("missing_recorder")].reset_index(drop=True)
