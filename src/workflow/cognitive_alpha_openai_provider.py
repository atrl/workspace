from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Sequence, TextIO


DEFAULT_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


COGNITIVE_ALPHA_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "level": {"type": "string"},
                    "agent": {"type": "string"},
                    "theme": {"type": "string"},
                    "required_columns": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "lookback": {"type": "integer", "minimum": 1},
                    "code": {"type": "string"},
                    "economic_rationale": {"type": "string"},
                },
                "required": [
                    "name",
                    "level",
                    "agent",
                    "theme",
                    "required_columns",
                    "lookback",
                    "code",
                    "economic_rationale",
                ],
            },
        }
    },
    "required": ["candidates"],
}


def build_openai_responses_payload(request: dict[str, Any], *, model: str) -> dict[str, Any]:
    system_prompt = (
        "Generate vectorized pandas alpha factor expressions for SOXL research. "
        "Return only JSON matching the provided schema. Each candidate must use only lagged information, "
        "avoid future shifts, and return a pandas Series expression using available columns."
    )
    user_payload = {
        "task": "cognitive_alpha_candidate_generation",
        "contract": request,
        "output_rules": [
            "Use one expression per candidate.",
            "Allowed variables are the available OHLCV/Alpha158 column names in the contract.",
            "Do not use shift(-n), future returns, row loops, imports, file IO, network IO, or broker data.",
        ],
    }
    return {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cognitive_alpha_candidates",
                "schema": COGNITIVE_ALPHA_CANDIDATE_SCHEMA,
                "strict": True,
            }
        },
    }


def openai_responses_completion(
    request: dict[str, Any],
    *,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_OPENAI_RESPONSES_URL,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("OpenAI Responses provider requires OPENAI_API_KEY")
    if not model:
        raise ValueError("OpenAI Responses provider requires --model or OPENAI_MODEL")
    payload = build_openai_responses_payload(request, model=model)
    http_request = urllib.request.Request(
        base_url,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenAI Responses provider HTTP {exc.code}: {detail}") from exc
    response_payload = json.loads(response_body)
    candidates = _extract_candidates_from_response(response_payload)
    _validate_candidates(candidates)
    return {"candidates": candidates}


def _extract_candidates_from_response(response_payload: dict[str, Any]) -> list[dict[str, Any]]:
    text = response_payload.get("output_text")
    if not isinstance(text, str):
        text = _first_output_text(response_payload)
    if not isinstance(text, str):
        text = _first_chat_completion_text(response_payload)
    if not isinstance(text, str):
        raise ValueError("OpenAI Responses provider response did not contain output text")
    decoded = json.loads(text)
    candidates = decoded.get("candidates") if isinstance(decoded, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("OpenAI Responses provider output JSON missing candidates list")
    return candidates


def _first_output_text(response_payload: dict[str, Any]) -> str | None:
    for item in response_payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and content.get("type") in {"output_text", "text", None}:
                return text
    return None


def _first_chat_completion_text(response_payload: dict[str, Any]) -> str | None:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else None


def _validate_candidates(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        raise ValueError("OpenAI Responses provider returned no candidates")
    required = set(COGNITIVE_ALPHA_CANDIDATE_SCHEMA["properties"]["candidates"]["items"]["required"])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("OpenAI Responses provider candidate must be an object")
        missing = sorted(required - set(candidate))
        if missing:
            raise ValueError(f"OpenAI Responses provider candidate missing fields: {missing}")
        if not isinstance(candidate.get("required_columns"), list) or not candidate["required_columns"]:
            raise ValueError("OpenAI Responses provider candidate requires non-empty required_columns")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin_text: str | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    environ: dict[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Generate CogAlpha candidates via OpenAI Responses API.")
    parser.add_argument("--base-url", default=DEFAULT_OPENAI_RESPONSES_URL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    env = environ if environ is not None else os.environ
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    api_key = env.get("OPENAI_API_KEY", "")
    model = args.model or env.get("OPENAI_MODEL", "")
    if not api_key:
        print("OpenAI Responses provider requires OPENAI_API_KEY", file=err)
        return 2
    if not model:
        print("OpenAI Responses provider requires --model or OPENAI_MODEL", file=err)
        return 2
    try:
        request_text = stdin_text if stdin_text is not None else sys.stdin.read()
        request = json.loads(request_text)
        result = openai_responses_completion(
            request,
            api_key=api_key,
            model=model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"OpenAI Responses provider failed: {exc}", file=err)
        return 1
    print(json.dumps(result, sort_keys=True), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
