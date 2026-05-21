#!/usr/bin/env python3
from __future__ import annotations

"""Check or deploy the SOXL static dashboard bundle to Vercel."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_dashboard_static_export import check_static_export

DEPLOYMENT_URL_RE = re.compile(r"https://[A-Za-z0-9][A-Za-z0-9.-]*\.vercel\.app\b")
VERCEL_ENV_KEYS = ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_ORG_ID")
PLACEHOLDER_ENV_VALUES = {"replace_me", "replace-me"}


def ascii_safe(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")


def is_configured_env_value(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return bool(normalized) and normalized not in PLACEHOLDER_ENV_VALUES


def build_deploy_command(root: Path, *, token: str | None = None, prod: bool = True) -> list[str]:
    command = ["vercel", "deploy", str(root), "--yes", "--no-color"]
    if prod:
        command.append("--prod")
    if token:
        command.extend(["--token", token])
    return command


def sanitize_command(command: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    skip_next = False
    for part in command:
        if skip_next:
            sanitized.append("<redacted>")
            skip_next = False
            continue
        sanitized.append(part)
        if part == "--token":
            skip_next = True
    return sanitized


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            parts = shlex.split(stripped, comments=True, posix=True)
        except ValueError:
            continue
        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key in VERCEL_ENV_KEYS and is_configured_env_value(value):
                values[key] = value
    return values


def effective_env_with_file(env: Mapping[str, str], env_file: Path | None) -> tuple[dict[str, str], dict[str, object]]:
    effective_env = dict(env)
    loaded_values = read_env_file(env_file) if env_file else {}
    applied_keys: list[str] = []
    for key, value in loaded_values.items():
        if not is_configured_env_value(effective_env.get(key)):
            effective_env[key] = value
            applied_keys.append(key)
    return effective_env, {
        "path": str(env_file) if env_file else "",
        "present": bool(env_file and env_file.exists()),
        "loaded_keys": sorted(loaded_values),
        "applied_keys": sorted(applied_keys),
    }


def auth_status(
    *,
    env: Mapping[str, str],
    whoami_returncode: int,
    whoami_stdout: str,
    whoami_stderr: str,
    token_env: str = "VERCEL_TOKEN",
) -> dict[str, str]:
    if is_configured_env_value(env.get(token_env)):
        return {
            "status": "token",
            "detail": f"{token_env} is set",
        }
    if whoami_returncode == 0:
        user = whoami_stdout.strip().splitlines()[-1] if whoami_stdout.strip() else "authenticated"
        return {
            "status": "logged_in",
            "detail": f"vercel whoami succeeded as {user}",
        }
    detail = ascii_safe(whoami_stderr.strip() or whoami_stdout.strip() or "vercel whoami did not find credentials")
    return {
        "status": "blocked_no_credentials",
        "detail": f"Set VERCEL_TOKEN or run vercel login. CLI detail: {detail}",
    }


def project_link_status(root: Path, env: Mapping[str, str]) -> dict[str, str]:
    project_json = root / ".vercel" / "project.json"
    if project_json.exists():
        try:
            payload = json.loads(project_json.read_text(encoding="utf-8"))
        except Exception:
            return {
                "status": "invalid_project_link",
                "detail": ".vercel/project.json exists but is not valid JSON",
            }
        if payload.get("projectId") and payload.get("orgId"):
            return {
                "status": "linked",
                "detail": ".vercel/project.json contains project and org IDs",
            }
        return {
            "status": "invalid_project_link",
            "detail": ".vercel/project.json is missing projectId or orgId",
        }

    if is_configured_env_value(env.get("VERCEL_PROJECT_ID")) and is_configured_env_value(env.get("VERCEL_ORG_ID")):
        return {
            "status": "env_configured",
            "detail": "VERCEL_PROJECT_ID and VERCEL_ORG_ID are set",
        }

    return {
        "status": "missing_project_link",
        "detail": (
            "Run `vercel link --yes --team <team-id> --project <project-name-or-id>` "
            "or configure VERCEL_ORG_ID and VERCEL_PROJECT_ID for non-interactive deploys."
        ),
    }


def extract_deployment_url(output: str) -> str | None:
    match = DEPLOYMENT_URL_RE.search(output)
    return match.group(0) if match else None


def _write_evidence(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vercel_deployment_check.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Vercel Dashboard Deployment Check",
        "",
        f"- status: `{payload['status']}`",
        f"- checked_at: `{payload['checked_at']}`",
        f"- dashboard_latest: `{payload['static_export']['latest']}`",
        f"- paper_trading_end: `{payload['static_export']['paper_trading_end']}`",
        f"- auth: `{payload['auth']['status']}`",
        f"- project: `{payload['project']['status']}`",
    ]
    env_file = payload.get("env_file")
    if isinstance(env_file, dict):
        env_file_status = "present" if env_file.get("present") else "missing"
        applied_keys = env_file.get("applied_keys") or []
        if isinstance(applied_keys, list):
            applied_text = ",".join(str(key) for key in applied_keys) or "none"
        else:
            applied_text = "unknown"
        lines.append(f"- env_file: `{env_file_status}`")
        lines.append(f"- env_file_applied_keys: `{applied_text}`")
    if payload.get("deployment_url"):
        lines.append(f"- deployment_url: `{payload['deployment_url']}`")
    if payload.get("detail"):
        lines.extend(["", "## Detail", "", ascii_safe(str(payload["detail"]))])
    (output_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_whoami() -> subprocess.CompletedProcess[str]:
    if shutil.which("vercel") is None:
        return subprocess.CompletedProcess(["vercel", "whoami"], 127, "", "vercel CLI not found")
    return subprocess.run(
        ["vercel", "whoami", "--no-color"],
        capture_output=True,
        text=True,
        check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_env_file = Path(os.environ.get("SOXL_LAUNCH_ENV_FILE", Path.home() / ".openclaw" / "soxl.env"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime" / "production" / "card" / "vercel_remote_deploy_v1",
    )
    parser.add_argument("--deploy", action="store_true", help="Run vercel deploy after checks pass")
    parser.add_argument("--prod", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-env", default="VERCEL_TOKEN")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=default_env_file,
        help="Load missing Vercel env vars from this LaunchAgent env file.",
    )
    parser.add_argument("--no-env-file", action="store_true", help="Do not read a LaunchAgent env file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    errors, static_info = check_static_export(root)
    env_file = None if args.no_env_file else args.env_file.expanduser().resolve()
    effective_env, env_file_info = effective_env_with_file(os.environ, env_file)
    whoami = _run_whoami()
    token = effective_env.get(args.token_env)
    auth = auth_status(
        env=effective_env,
        whoami_returncode=whoami.returncode,
        whoami_stdout=whoami.stdout,
        whoami_stderr=whoami.stderr,
        token_env=args.token_env,
    )
    project = project_link_status(root, effective_env)
    payload: dict[str, object] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": "ready",
        "static_export": static_info,
        "static_errors": errors,
        "vercel_cli": shutil.which("vercel") or "",
        "env_file": env_file_info,
        "auth": auth,
        "project": project,
        "deploy_requested": args.deploy,
        "deployment_url": None,
        "detail": "",
    }
    if errors:
        payload["status"] = "static_export_failed"
        payload["detail"] = "; ".join(errors)
        _write_evidence(output_dir, payload)
        print(f"vercel_dashboard status=static_export_failed output={output_dir}")
        return 1
    if auth["status"] == "blocked_no_credentials":
        payload["status"] = "blocked_no_credentials"
        payload["detail"] = auth["detail"]
        _write_evidence(output_dir, payload)
        print(f"vercel_dashboard status=blocked_no_credentials output={output_dir}")
        return 2 if args.deploy else 0
    if project["status"] in {"missing_project_link", "invalid_project_link"}:
        payload["status"] = "blocked_no_project_link"
        payload["detail"] = project["detail"]
        _write_evidence(output_dir, payload)
        print(f"vercel_dashboard status=blocked_no_project_link output={output_dir}")
        return 3 if args.deploy else 0
    if not args.deploy:
        payload["status"] = "ready_for_deploy"
        payload["detail"] = "Static bundle and Vercel authentication are available; rerun with --deploy to publish."
        _write_evidence(output_dir, payload)
        print(f"vercel_dashboard status=ready_for_deploy output={output_dir}")
        return 0

    command = build_deploy_command(root, token=token, prod=args.prod)
    payload["command"] = sanitize_command(command)
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=effective_env)
    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    payload["deployment_url"] = extract_deployment_url(combined_output)
    payload["vercel_returncode"] = result.returncode
    if result.returncode == 0 and payload["deployment_url"]:
        payload["status"] = "deployed"
        payload["detail"] = "Vercel deployment completed."
        _write_evidence(output_dir, payload)
        print(f"vercel_dashboard status=deployed url={payload['deployment_url']} output={output_dir}")
        return 0
    payload["status"] = "deploy_failed"
    payload["detail"] = combined_output[-4000:]
    _write_evidence(output_dir, payload)
    print(f"vercel_dashboard status=deploy_failed output={output_dir}")
    return result.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
