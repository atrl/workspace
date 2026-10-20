"""Read-only job CLI without importing the scheduling/training package."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

from src.core.research_runtime import aggregate_usage, observe_job, project_task


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    state, receipts = args.state, args.receipts
    progress_path = args.state
    pid, start_identity = args.pid, args.start_identity
    exit_receipt, heartbeat = args.exit_receipt, args.heartbeat
    observed_goal = None
    if args.campaign:
        campaign = json.loads(args.campaign.read_text())
        root = args.campaign.resolve().parent.parent
        task = campaign["current_task"]
        receipts = root / task["phase_receipts"]
        run_path = root / task["run_state"]
        current = json.loads(run_path.read_text()) if run_path.exists() else {}
        state = {**current, "goal_state": campaign["goal_controller"]["observed_status"],
                 "job_state": "unverified_running" if current.get("status") in {"started", "running"}
                 else "unknown",
                 "scientific_state": task.get("last_scientific_observation", "unknown")}
        observed_goal = campaign["goal_controller"]
        progress_path = run_path
        handle_path = root / task["job_handle"]
        if handle_path.exists():
            handle = json.loads(handle_path.read_text())
            pid = pid or handle.get("pid")
            start_identity = start_identity or handle.get("start_identity")
        exit_path = root / task["execution_receipt"]
        exit_receipt = exit_receipt or (exit_path if exit_path.exists() else None)
        heartbeat = heartbeat or current.get("heartbeat_at")
    projection = project_task(receipts, state)
    job = observe_job(pid=pid, start_identity=start_identity,
                      exit_receipt=exit_receipt, progress_path=progress_path,
                      heartbeat=heartbeat, stale_after_seconds=args.stale_after)
    usage = aggregate_usage(args.usage) if args.usage else None
    # Deliberately expose only state fields, hashes, and numeric usage; never log file contents.
    return {"task": asdict(projection), "job": job, "usage": usage,
            "goal_observation": observed_goal}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path,
                        help="resolve current task references, without rewriting campaign state")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--start-identity")
    parser.add_argument("--exit-receipt", type=Path)
    parser.add_argument("--heartbeat")
    parser.add_argument("--stale-after", type=float, default=300)
    parser.add_argument("--usage", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--once", action="store_true", help="take one read-only snapshot (default)")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=10,
                        help="bounded watch count (maximum 120)")
    args = parser.parse_args(argv)
    if args.interval < 1 or args.iterations < 1 or args.iterations > 120:
        parser.error("interval must be positive and iterations must be between 1 and 120")
    last = None
    count = args.iterations if args.watch else 1
    for _ in range(count):
        current = snapshot(args)
        comparison = json.loads(json.dumps(current))
        comparison["job"]["heartbeat_age_seconds"] = None
        encoded = json.dumps(comparison, sort_keys=True, ensure_ascii=False)
        if encoded != last:
            print(json.dumps(current, sort_keys=True, ensure_ascii=False), flush=True)
            last = encoded
        if _ + 1 < count:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
