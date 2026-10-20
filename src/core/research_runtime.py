"""Read-only projections for research receipts, jobs, and local usage logs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

TERMINAL_PHASE_STATES = {
    "complete", "completed", "succeeded", "success", "skipped",
    "failed", "closed_failed", "not_promoted", "cancelled",
}

@dataclass(frozen=True)
class TaskProjection:
    goal_state: str
    job_state: str
    scientific_state: str
    current_phase: str | None
    phases: dict[str, str]
    issues: tuple[str, ...] = ()


def _read_json(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    return json.loads(path.read_text()) if path.exists() else {}


def project_task(receipts: Iterable[Mapping[str, Any]] | str | Path | None,
                 run_state: Mapping[str, Any] | str | Path | None = None) -> TaskProjection:
    """Project normalized ``{phase, status}`` receipts and state independently."""
    state = _read_json(run_state)
    if isinstance(receipts, (str, Path)):
        raw = json.loads(Path(receipts).read_text()) if Path(receipts).exists() else {}
        if isinstance(raw, list):
            receipts = raw
        elif isinstance(raw, dict) and raw.get("phase") and raw.get("status"):
            receipts = [raw]
        else:
            receipts = raw.get("receipts", []) if isinstance(raw, dict) else []
    receipt_rows = list(receipts or [])
    phases: dict[str, str] = {}
    for row in receipt_rows:
        phase = row.get("phase")
        status = row.get("status")
        if phase and status:
            phases[str(phase)] = str(status)
    # A completed receipt is authoritative over stale pending/progress fields.
    for phase, status in list(phases.items()):
        if status.lower() in {"complete", "completed", "succeeded", "success"}:
            phases[phase] = "completed"
    for phase, status in (state.get("phases") or {}).items():
        if phase not in phases:
            phases[str(phase)] = str(status)
    current = next((p for p, s in reversed(list(phases.items()))
                    if s.lower() not in TERMINAL_PHASE_STATES), None)
    goal = str(state.get("goal_state", state.get("goal_status", "unknown")))
    job = str(state.get("job_state", state.get("status", "unknown")))
    issues = []
    if job.lower() in {"running", "active", "started", "in_progress"}:
        job = "unverified_running"
        issues.append("run_state claims running; process identity was not verified")
    science = str(state.get("scientific_state", state.get("promotion_decision", "unknown")))
    state_phase = state.get("current_phase")
    if state_phase and phases.get(str(state_phase), "").lower() in {"completed", "complete", "succeeded", "success"}:
        issues.append(f"stale run_state current_phase={state_phase}; receipt is completed")
        current = next((p for p, s in reversed(list(phases.items()))
                        if s.lower() not in TERMINAL_PHASE_STATES), None)
    return TaskProjection(goal, job, science, current, phases, tuple(issues))


def process_start_identity(pid: int) -> str | None:
    """Return OS start identity when available, preventing PID-reuse matches."""
    proc = Path(f"/proc/{pid}/stat")
    if proc.exists():
        try:
            data = proc.read_text()
            return "linux:" + data[data.rfind(")") + 2:].split()[19]
        except (OSError, IndexError):
            return None
    try:  # macOS/BSD fallback; ps start string is stable for this process lifetime.
        import subprocess
        result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], check=False,
                                capture_output=True, text=True, timeout=2)
        value = result.stdout.strip()
        return "ps:" + value if result.returncode == 0 and value else None
    except (OSError, subprocess.SubprocessError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def observe_job(*, pid: int | None = None, start_identity: str | None = None,
                exit_receipt: str | Path | Mapping[str, Any] | None = None,
                progress_path: str | Path | None = None, heartbeat: str | None = None,
                stale_after_seconds: float = 300, now: float | None = None) -> dict[str, Any]:
    """Observe a process handle; never starts, signals, or restarts a child."""
    receipt = _read_json(exit_receipt)
    actual_identity = process_start_identity(int(pid)) if pid else None
    alive = bool(pid and _pid_alive(int(pid)))
    identity_matches = bool(start_identity and actual_identity == start_identity)
    receipt_exit = receipt.get("exit_code")
    if receipt_exit is None:
        receipt_status = "absent"
    elif pid is None or start_identity is None:
        receipt_status = "untrusted_missing_expected_identity"
    elif str(receipt.get("pid")) != str(pid) or str(receipt.get("start_identity")) != str(start_identity):
        receipt_status = "untrusted_mismatch"
    else:
        receipt_status = "matched"
    if receipt_status == "matched":
        status = "completed" if int(receipt["exit_code"]) == 0 else "failed"
    elif pid is None:
        status = "unknown"
    elif not alive:
        status = "missing"
    elif start_identity is None:
        status = "running_unverified_identity"
    elif actual_identity is None:
        status = "running_unverified_identity"
    elif not identity_matches:
        status = "pid_reused"
    else:
        status = "running"
    signature = None
    if progress_path and Path(progress_path).exists():
        raw = Path(progress_path).read_bytes()
        signature = hashlib.sha256(raw).hexdigest()[:16]
    age = None
    stale = None
    if heartbeat:
        try:
            stamp = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
            age = max(0.0, (datetime.now(timezone.utc).timestamp() if now is None else now) - stamp.timestamp())
            stale = age > stale_after_seconds
        except ValueError:
            stale = True
    return {"status": status, "pid": pid, "start_identity": actual_identity,
            "identity_matches": identity_matches, "exit_code": receipt_exit,
            "exit_receipt_status": receipt_status,
            "progress_signature": signature, "heartbeat_age_seconds": age,
            "heartbeat_stale": stale}


USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                "output_tokens", "reasoning_output_tokens", "total_tokens")


def aggregate_usage(path: str | Path) -> dict[str, Any]:
    """Aggregate JSONL usage records; cached/reasoning counts remain subsets."""
    totals = {key: 0 for key in USAGE_FIELDS}
    response_ids: set[str] = set()
    cumulative: dict[str, dict[str, int]] = {}
    response_totals: dict[str, dict[str, int]] = {}
    response_since_cumulative: dict[str, dict[str, int]] = {}
    fallback_seen: set[str] = set()
    models: set[str] = set()
    roles: set[str] = set()
    malformed = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                malformed += 1
                continue
            if row.get("type") != "token_usage_record":
                continue
            payload = row.get("payload") or {}
            if payload.get("model"):
                models.add(str(payload["model"]))
            if payload.get("role"):
                roles.add(str(payload["role"]))
            response = payload.get("response_id")
            if response and response in response_ids:
                continue
            if response:
                response_ids.add(response)
            usage = payload.get("usage")
            if isinstance(usage, Mapping):
                values = {k: int(usage.get(k, 0) or 0) for k in USAGE_FIELDS}
                for key, value in values.items():
                    totals[key] += value
                turn = str(payload.get("turn_id") or payload.get("root_turn_id") or "default")
                known = response_totals.setdefault(turn, {k: 0 for k in USAGE_FIELDS})
                for key, value in values.items():
                    known[key] += value
                if turn in cumulative:
                    pending = response_since_cumulative.setdefault(turn, {k: 0 for k in USAGE_FIELDS})
                    for key, value in values.items():
                        pending[key] += value
                continue
            # Older rows may omit response usage but retain cumulative turn usage.
            turn = str(payload.get("turn_id") or payload.get("root_turn_id") or "default")
            cumulative_row = payload.get("turn_token_usage")
            if isinstance(cumulative_row, Mapping):
                current = {k: int(cumulative_row.get(k, 0) or 0) for k in USAGE_FIELDS}
                previous = cumulative.get(turn, response_totals.get(turn, {k: 0 for k in USAGE_FIELDS}))
                pending = response_since_cumulative.get(turn, {k: 0 for k in USAGE_FIELDS})
                for key in USAGE_FIELDS:
                    accounted = previous[key] + pending[key]
                    totals[key] += max(0, current[key] - accounted)
                cumulative[turn] = {k: max(previous[k] + pending[k], current[k]) for k in USAGE_FIELDS}
                response_since_cumulative[turn] = {k: 0 for k in USAGE_FIELDS}
            elif response and response not in fallback_seen:
                last = payload.get("last_token_usage") or {}
                values = {k: int(last.get(k, 0) or 0) for k in USAGE_FIELDS}
                for key in USAGE_FIELDS:
                    totals[key] += values[key]
                turn = str(payload.get("turn_id") or payload.get("root_turn_id") or "default")
                known = response_totals.setdefault(turn, {k: 0 for k in USAGE_FIELDS})
                for key, value in values.items():
                    known[key] += value
                if turn in cumulative:
                    pending = response_since_cumulative.setdefault(turn, {k: 0 for k in USAGE_FIELDS})
                    for key, value in values.items():
                        pending[key] += value
                fallback_seen.add(response)
    result = {"totals": totals, "responses": len(response_ids), "malformed_lines": malformed,
            "billing": "not_inferred", "cached_input_is_subset": True,
            "reasoning_output_is_subset": True, "aggregation_mode": "full_scan"}
    if models:
        result["models"] = sorted(models)
    if roles:
        result["roles"] = sorted(roles)
    return result
