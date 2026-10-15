"""One-shot, stdlib-only supervisor for an independently permitted future pilot.

Run this file with ``python -I -S``. It never imports the research environment;
only the separately spawned child may do that, inside the registered window.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(timezone.utc)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_status(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verified_contract(root, card):
    approval = json.loads((card / "supervision_approval.json").read_text())
    if approval.get("decision") != "APPROVED_ONE_SHOT_STDLIB_WAITER":
        raise PermissionError("no supervision approval")
    bindings = approval["files_sha256"]
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("supervision binding changed: " + relative)
    registration = json.loads((card / "pilot_registration.json").read_text())
    permit = json.loads((card / "pilot_activation.json").read_text())
    if permit["registration_sha256"] != sha(card / "pilot_registration.json"):
        raise ValueError("registration differs from activation")
    for relative, expected in registration["code_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("registered code changed: " + relative)
    for filename, field in [("event_input_binding.json", "event_binding_sha256"),
                            ("bootstrap_prefix_binding.json", "prefix_binding_sha256")]:
        if sha(card / filename) != registration[field]:
            raise ValueError("registered input binding changed")
    prefix = json.loads((card / "bootstrap_prefix_binding.json").read_text())
    prefix_path = (root / prefix["object_path"]).resolve()
    if not prefix_path.is_relative_to(root) or sha(prefix_path) != prefix["object_sha256"]:
        raise ValueError("immutable prefix changed")
    if sha(root / "runtime/soxl/stage1_regime_paper/config.json") != registration["endpoint_config_sha256"]:
        raise ValueError("endpoint configuration changed")
    if sha(root / "qlib_venv/bin/python") != approval["interpreter_sha256"]:
        raise ValueError("pilot interpreter changed")
    if approval["registration_sha256"] != permit["registration_sha256"]:
        raise ValueError("waiter belongs to a different registration")
    start, stop = stamp(permit["not_before"]), stamp(permit["expires_at"])
    latest = stamp(approval["latest_child_start"])
    if not start < latest < stop or approval["maximum_child_attempts"] != 1:
        raise ValueError("invalid one-shot supervision window")
    return approval, permit, start, latest, stop


def supervise(root, card):
    root, card = Path(root).resolve(), Path(card).resolve()
    approval, permit, start, latest, stop = verified_contract(root, card)
    approval_sha = sha(card / "supervision_approval.json")
    directory = root / "runtime/research/diagnostic_forward_io_v2_20261016" / (
        "supervisor_" + permit["registration_sha256"][:16])
    directory.mkdir(parents=True, exist_ok=True)
    state = {"supervisor_pid": os.getpid(), "supervisor_started": now().isoformat(),
             "registration_sha256": permit["registration_sha256"], "phase": "waiting",
             "pilot_started": False, "child_attempts": 0, "claims_strategy_qualification": False,
             "supervision_approval_sha256": approval_sha,
             "not_before": start.isoformat(), "latest_child_start": latest.isoformat(),
             "expires_at": stop.isoformat()}
    # An existing claim is terminal for this invocation, even if its PID died.
    with (directory / "supervisor_claim.json").open("x") as claim:
        json.dump(state, claim)
        claim.flush()
        os.fsync(claim.fileno())
    status_path = directory / "status.json"
    child = None
    try:
        write_status(status_path, state)
        print(json.dumps({"phase": "waiting", "pid": os.getpid(), "status_path": str(status_path)}), flush=True)
        while now() < start:
            state["heartbeat_at"] = now().isoformat()
            write_status(status_path, state)
            time.sleep(min(30.0, max(0.0, (start - now()).total_seconds())))
        if now() >= latest:
            state["phase"] = "missed_start_window"
            return state
        # Revalidate all immutable files after the wait, before any child starts.
        if sha(card / "supervision_approval.json") != approval_sha:
            raise ValueError("supervision approval changed during wait")
        verified_contract(root, card)
        if now() >= latest:
            state["phase"] = "missed_start_window_after_identity_check"
            return state
        state.update(phase="child_attempt_registered", child_attempts=1, attempt_at=now().isoformat())
        write_status(status_path, state)
        command = [str(root / "qlib_venv/bin/python"), "-u", "-c",
                   "from pathlib import Path; import sys; "
                   "from src.schedule.diagnostic_forward_transport_v2 import run_pilot; "
                   "result=run_pilot(Path(sys.argv[1])); "
                   "sys.exit(0 if result.get('qualified_interval',False) else 2)", str(root)]
        env = dict(os.environ, PYTHONPATH=str(root), MLFLOW_DISABLE_TELEMETRY="true",
                   MLFLOW_ALLOW_FILE_STORE="true", DO_NOT_TRACK="true")
        with (directory / "pilot_output.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(phase="pilot_process_running", pilot_started=True, child_pid=child.pid,
                         child_started_at=now().isoformat())
            write_status(status_path, state)
            while child.poll() is None:
                if now() >= stop:
                    state["phase"] = "expired_child_terminated"
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                    break
                state["heartbeat_at"] = now().isoformat()
                write_status(status_path, state)
                time.sleep(1)
            state["child_returncode"] = child.returncode
            if state["phase"] == "pilot_process_running":
                state["phase"] = "child_finished" if child.returncode == 0 else "child_unqualified_or_failed"
        return state
    except BaseException as error:
        state.update(phase="supervisor_failed", error_type=type(error).__name__)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        raise
    finally:
        state["last_status_at"] = now().isoformat()
        write_status(status_path, state)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    supervise(args.root, args.root / "research/diagnostic_forward_io_v2_20261016")
