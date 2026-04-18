#!/usr/bin/env python3
from __future__ import annotations

"""Search normalized memory metadata without bulk-reading every file."""

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.memory_tools import load_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search normalized memory index.")
    parser.add_argument("query", nargs="*", help="free-text query terms")
    parser.add_argument("--tag", action="append", default=[], help="required tag")
    parser.add_argument("--profile", action="append", default=[], help="required profile")
    parser.add_argument("--status", action="append", default=[], help="required status")
    parser.add_argument("--kind", action="append", default=[], help="required kind")
    parser.add_argument("--date-from", default="", help="minimum date YYYY-MM-DD")
    parser.add_argument("--date-to", default="", help="maximum date YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=8, help="max results")
    return parser.parse_args()


def record_text(record: dict[str, Any]) -> str:
    fields = [
        record.get("title", ""),
        record.get("summary", ""),
        " ".join(record.get("tags", [])),
        " ".join(record.get("profiles", [])),
        record.get("excerpt", ""),
    ]
    return " ".join(str(field) for field in fields).lower()


def passes_filters(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.tag and not set(args.tag).issubset(set(record.get("tags", []))):
        return False
    if args.profile and not set(args.profile).issubset(set(record.get("profiles", []))):
        return False
    if args.status and record.get("status") not in set(args.status):
        return False
    if args.kind and record.get("kind") not in set(args.kind):
        return False
    if args.date_from and record.get("date", "") < args.date_from:
        return False
    if args.date_to and record.get("date", "") > args.date_to:
        return False
    return True


def score_record(record: dict[str, Any], query_terms: list[str]) -> int:
    haystack = record_text(record)
    score = 0
    for term in query_terms:
        if term in haystack:
            score += 5
        if term in record.get("title", "").lower():
            score += 3
        if term in " ".join(record.get("tags", [])).lower():
            score += 2
    if record.get("status") == "dropped":
        score += 1
    return score


def main() -> int:
    args = parse_args()
    query_terms = [term.lower() for term in args.query]
    results = []
    for record in load_index():
        if not passes_filters(record, args):
            continue
        score = score_record(record, query_terms)
        if query_terms and score == 0:
            continue
        results.append((score, record))

    results.sort(key=lambda item: (item[0], item[1].get("date", "")), reverse=True)
    limited = results[: args.top_k]

    if not limited:
        print("[memory] no matches")
        return 1

    for score, record in limited:
        print(f"{record['path']}  score={score}")
        print(
            f"  date={record.get('date')} status={record.get('status')} "
            f"kind={record.get('kind')} profiles={','.join(record.get('profiles', [])) or '-'}"
        )
        print(f"  tags={','.join(record.get('tags', [])) or '-'}")
        print(f"  summary={record.get('summary', '')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
