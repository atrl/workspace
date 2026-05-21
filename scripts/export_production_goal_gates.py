#!/usr/bin/env python3
"""Export scheduler production goal gates for the static dashboard."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOAL_MARKER = "--- Production Goal Gates ---"


def _slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return slug.strip("_")


def _split_status_detail(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "unknown", ""
    parts = stripped.split(maxsplit=1)
    status = parts[0]
    detail = parts[1] if len(parts) > 1 else ""
    return status, detail


def parse_goal_gates(output: str, *, checked_at: str | None = None) -> dict[str, object]:
    """Parse the `Production Goal Gates` section from scheduler health output."""

    lines = output.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == GOAL_MARKER) + 1
    except StopIteration as exc:
        raise ValueError("Production Goal Gates section not found") from exc

    gates: dict[str, dict[str, str]] = {}
    overall = {"status": "unknown", "issues": [], "detail": ""}

    for raw_line in lines[start:]:
        stripped = raw_line.strip()
        if not stripped:
            break
        if stripped.startswith("--- "):
            break
        if ":" not in stripped:
            continue
        label, rest = stripped.split(":", 1)
        status, detail = _split_status_detail(rest)
        key = _slug(label)
        if key == "overall":
            issues = [item for item in detail.split(",") if item] if status == "pending" else []
            overall = {"status": status, "issues": issues, "detail": detail}
            continue
        gates[key] = {
            "label": label.strip(),
            "status": status,
            "detail": detail,
            "raw": stripped,
        }

    return {
        "checked_at": checked_at or datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "gates": gates,
    }


def export_goal_gates(root: Path, output_path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["bash", "production/check_scheduler_health.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"check_scheduler_health.sh failed with exit code {result.returncode}: {stderr}")
    payload = parse_goal_gates(result.stdout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dashboard_data" / "production_goal_gates.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_path = args.output if args.output.is_absolute() else root / args.output
    try:
        payload = export_goal_gates(root, output_path)
    except Exception as exc:
        print(f"production_goal_gates status=failed detail={exc}", file=sys.stderr)
        return 1

    issues = payload.get("overall", {}).get("issues", []) if isinstance(payload.get("overall"), dict) else []
    issue_text = ",".join(str(issue) for issue in issues) or "none"
    status = payload.get("overall", {}).get("status", "unknown") if isinstance(payload.get("overall"), dict) else "unknown"
    print(f"production_goal_gates status={status} issues={issue_text} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
