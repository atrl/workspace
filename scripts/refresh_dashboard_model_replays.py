#!/usr/bin/env python3
"""Refresh dashboard daily replay CSVs for registry-owned model actors."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "runtime" / "harness" / "card" / "registry" / "profile.yaml"
MANIFEST_PATH = ROOT / "dashboard_data" / "model_replay_freshness.json"


@dataclass(frozen=True)
class ReplayCandidate:
    name: str
    status: str
    actor_path: Path
    profile_path: Path
    output_path: Path


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_registry_rows(root: Path) -> list[dict]:
    path = root / "runtime" / "harness" / "card" / "registry" / "profile.yaml"
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("profiles", []) if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _profile_path(root: Path, profile: str) -> Path | None:
    direct = root / "runtime" / "model" / "card" / profile / "profile.yaml"
    if direct.exists():
        return direct
    experiment_matches = sorted((root / "experiments").glob(f"*/{profile}/profile.yaml"))
    if experiment_matches:
        return experiment_matches[0]
    runtime_matches = [
        path
        for path in sorted((root / "runtime").rglob(f"card/{profile}/profile.yaml"))
        if "harness/card/registry" not in str(path)
    ]
    return runtime_matches[0] if runtime_matches else None


def discover_replay_candidates(root: Path = ROOT, profile_names: set[str] | None = None) -> list[ReplayCandidate]:
    candidates: list[ReplayCandidate] = []
    for row in _load_registry_rows(root):
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if profile_names is not None and name not in profile_names:
            continue
        status = str(row.get("status") or "")
        if status == "dropped" or not bool(row.get("keep_actor", False)):
            continue
        actor_path = root / "runtime" / "model" / "card" / name / f"soxl_{name}_actor.pt"
        if not actor_path.exists():
            continue
        profile_path = _profile_path(root, name)
        if profile_path is None:
            continue
        candidates.append(
            ReplayCandidate(
                name=name,
                status=status,
                actor_path=actor_path,
                profile_path=profile_path,
                output_path=root / "runtime" / "model" / "card" / name / f"{name}_soxl_daily.csv",
            )
        )
    return candidates


def calendar_dates(root: Path = ROOT) -> list[str]:
    path = root / "data" / "qlib" / "soxl" / "calendars" / "day.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_data_date(root: Path = ROOT) -> str:
    dates = calendar_dates(root)
    return dates[-1] if dates else ""


def safe_backtest_end(root: Path = ROOT) -> str:
    dates = calendar_dates(root)
    if len(dates) >= 2:
        return dates[-2]
    return dates[-1] if dates else ""


def csv_last_date(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            last = str(row.get("datetime") or "").strip()
    return last


def _date_ge(left: str, right: str) -> bool:
    return bool(left and right and left >= right)


def export_candidate(candidate: ReplayCandidate, *, python: str = sys.executable) -> None:
    del candidate, python
    raise RuntimeError(
        "Legacy actor replay export has been retired. Use supervised Alpha158/cadence "
        "model-selection artifacts for dashboard model evidence."
    )


def refresh_model_replays(
    root: Path,
    candidates: list[ReplayCandidate],
    *,
    latest_data_date: str,
    safe_backtest_end: str,
    force: bool = False,
    exporter: Callable[[ReplayCandidate], None] | None = None,
) -> dict:
    exporter = exporter or (lambda candidate: export_candidate(candidate, python=sys.executable))
    counts = {"current": 0, "refreshed": 0, "failed": 0, "skipped": 0}
    profiles: list[dict] = []

    for candidate in candidates:
        before_end = csv_last_date(candidate.output_path)
        row = {
            "profile": candidate.name,
            "status": candidate.status,
            "actor": _relative(candidate.actor_path, root),
            "profile_yaml": _relative(candidate.profile_path, root),
            "output": _relative(candidate.output_path, root),
            "before_end": before_end,
            "after_end": before_end,
            "result": "current",
            "reason": "",
        }
        if not safe_backtest_end:
            row["result"] = "skipped"
            row["reason"] = "missing qlib calendar"
            counts["skipped"] += 1
            profiles.append(row)
            continue
        if not force and _date_ge(before_end, safe_backtest_end):
            row["reason"] = "already at safe_backtest_end"
            counts["current"] += 1
            profiles.append(row)
            continue
        try:
            exporter(candidate)
            after_end = csv_last_date(candidate.output_path)
            row["after_end"] = after_end
            if _date_ge(after_end, safe_backtest_end):
                row["result"] = "refreshed"
                row["reason"] = "exported daily replay"
                counts["refreshed"] += 1
            else:
                row["result"] = "failed"
                row["reason"] = f"export ended at {after_end or 'unknown'}, expected {safe_backtest_end}"
                counts["failed"] += 1
        except Exception as exc:
            row["result"] = "failed"
            row["reason"] = str(exc).splitlines()[-1][:500]
            row["after_end"] = csv_last_date(candidate.output_path)
            counts["failed"] += 1
        profiles.append(row)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_data_date": latest_data_date,
        "safe_backtest_end": safe_backtest_end,
        "terminal_note": (
            "qlib daily backtests require a following calendar boundary, so model replay curves "
            "target safe_backtest_end while paper trading/latest signal can use latest_data_date."
        ),
        "counts": counts,
        "profiles": profiles,
    }
    out_path = root / "dashboard_data" / "model_replay_freshness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--profile", action="append", default=[], help="Refresh only this profile; may repeat.")
    parser.add_argument("--force", action="store_true", help="Refresh even when the CSV is already current.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for export subprocesses.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    selected = set(args.profile) if args.profile else None
    candidates = discover_replay_candidates(root, profile_names=selected)

    def exporter(candidate: ReplayCandidate) -> None:
        export_candidate(candidate, python=args.python)

    manifest = refresh_model_replays(
        root,
        candidates,
        latest_data_date=latest_data_date(root),
        safe_backtest_end=safe_backtest_end(root),
        force=args.force,
        exporter=exporter,
    )
    counts = manifest["counts"]
    print(
        "model_replay_refresh "
        f"profiles={len(manifest['profiles'])} current={counts['current']} "
        f"refreshed={counts['refreshed']} failed={counts['failed']} skipped={counts['skipped']} "
        f"latest_data_date={manifest['latest_data_date']} safe_backtest_end={manifest['safe_backtest_end']}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
