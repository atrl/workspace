from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, TextIO

from src.workflow.cognitive_alpha_candidate_schema import (
    COGNITIVE_ALPHA_CANDIDATE_SCHEMA,
    validate_cognitive_alpha_candidate_payload,
)
from src.workflow.cognitive_alpha_evolution import build_cognitive_alpha_external_llm_candidate_provider


DEFAULT_CODEX_BIN = "codex"


def build_codex_candidate_prompt(request: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are generating SOXL CogAlpha factor candidates for a qlib research workflow.",
            "Return only JSON matching the provided output schema.",
            "Each candidate must be a vectorized pandas Series expression.",
            "Use only lagged information; do not use shift(-n), future returns, imports, file IO, network IO, or broker data.",
            "Allowed variables are open, high, low, close, volume, np, abs, log1p, sqrt, tanh, maximum, minimum, where, zscore, and identifier-safe Alpha158 columns in the request.",
            "Request JSON:",
            json.dumps(request, indent=2, sort_keys=True),
        ]
    )


def codex_exec_completion(
    request: dict[str, Any],
    *,
    codex_bin: str = DEFAULT_CODEX_BIN,
    model: str | None = None,
    cwd: Path | str | None = None,
    timeout_seconds: float = 300.0,
    sandbox: str = "read-only",
) -> dict[str, Any]:
    resolved_codex = _resolve_codex_bin(codex_bin)
    prompt = build_codex_candidate_prompt(request)
    with tempfile.TemporaryDirectory(prefix="cogalpha_codex_provider_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        schema_path = tmp_path / "schema.json"
        output_path = tmp_path / "last_message.json"
        schema_path.write_text(json.dumps(COGNITIVE_ALPHA_CANDIDATE_SCHEMA, sort_keys=True), encoding="utf-8")
        command = [
            resolved_codex,
            "exec",
            "--sandbox",
            sandbox,
            "--ask-for-approval",
            "never",
            "--output-schema",
            schema_path.as_posix(),
            "--output-last-message",
            output_path.as_posix(),
        ]
        if model:
            command.extend(["--model", model])
        if cwd is not None:
            command.extend(["--cd", str(cwd)])
        command.append("-")
        child_env = os.environ.copy()
        child_env.pop("OPENAI_API_KEY", None)
        child_env.pop("OPENAI_MODEL", None)
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env=child_env,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise ValueError(f"Codex exec provider exited {completed.returncode}" + (f": {stderr}" if stderr else ""))
        raw_output = output_path.read_text(encoding="utf-8") if output_path.exists() else completed.stdout
    payload = json.loads(raw_output)
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("Codex exec provider output JSON missing candidates list")
    validate_cognitive_alpha_candidate_payload(candidates, provider_name="Codex exec provider")
    return {"candidates": candidates}


def build_cognitive_alpha_codex_exec_candidate_provider(
    *,
    codex_bin: str = DEFAULT_CODEX_BIN,
    provider_id: str = "codex_exec_candidate_provider",
    model: str | None = None,
    cwd: Path | str | None = None,
    timeout_seconds: float = 300.0,
    sandbox: str = "read-only",
):
    def _completion(request: dict[str, Any]) -> dict[str, Any]:
        return codex_exec_completion(
            request,
            codex_bin=codex_bin,
            model=model,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            sandbox=sandbox,
        )

    provider = build_cognitive_alpha_external_llm_candidate_provider(
        completion=_completion,
        provider_id=provider_id,
        model=model or "codex-cli",
    )
    provider.provider_kind = "codex_exec_code"
    provider.openai_api_key_required = False
    provider.codex_direct_compatible = True
    provider.codex_execution_mode = "codex_exec_chatgpt_auth_no_openai_api_key"
    return provider


def codex_cli_readiness(
    *,
    codex_bin: str = DEFAULT_CODEX_BIN,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    resolved_codex = _resolve_codex_bin(codex_bin)
    child_env = os.environ.copy()
    child_env.pop("OPENAI_API_KEY", None)
    child_env.pop("OPENAI_MODEL", None)
    completed = subprocess.run(
        [resolved_codex, "doctor", "--json"],
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
        env=child_env,
    )
    if completed.returncode != 0:
        return {
            "status": "error",
            "codex_bin": resolved_codex,
            "openai_api_key_required": False,
            "error": completed.stderr.strip(),
        }
    report = json.loads(completed.stdout)
    auth = report.get("checks", {}).get("auth.credentials", {}).get("details", {})
    return {
        "status": str(report.get("overallStatus", "unknown")),
        "codex_bin": resolved_codex,
        "codex_version": str(report.get("codexVersion", "unknown")),
        "auth_mode": str(auth.get("stored auth mode", "unknown")),
        "stored_api_key": str(auth.get("stored API key", "unknown")),
        "stored_chatgpt_tokens": str(auth.get("stored ChatGPT tokens", "unknown")),
        "openai_api_key_required": False,
    }


def _resolve_codex_bin(codex_bin: str) -> str:
    if "/" in codex_bin:
        return codex_bin
    resolved = shutil.which(codex_bin)
    if not resolved:
        raise ValueError(f"Codex executable not found on PATH: {codex_bin}")
    return resolved


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin_text: str | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Generate CogAlpha candidates via Codex exec.")
    parser.add_argument("--codex-bin", default=DEFAULT_CODEX_BIN)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cd", default=None)
    parser.add_argument("--sandbox", default="read-only")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--readiness", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        if args.readiness:
            result = codex_cli_readiness(codex_bin=args.codex_bin, timeout_seconds=args.timeout_seconds)
        else:
            request_text = stdin_text if stdin_text is not None else sys.stdin.read()
            request = json.loads(request_text)
            result = codex_exec_completion(
                request,
                codex_bin=args.codex_bin,
                model=args.model,
                cwd=args.cd,
                timeout_seconds=args.timeout_seconds,
                sandbox=args.sandbox,
            )
    except Exception as exc:
        print(f"Codex exec provider failed: {exc}", file=err)
        return 1
    print(json.dumps(result, sort_keys=True), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
