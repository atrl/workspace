from __future__ import annotations

import json
import sys
from typing import Any, Sequence, TextIO


def fixture_response(request: dict[str, Any]) -> dict[str, Any]:
    request_index = int(request.get("request_index", 0))
    target_count = max(int(request.get("target_candidate_count", 1)), 1)
    candidates = []
    for offset in range(target_count):
        index = request_index + offset
        candidates.append(_candidate_payload(request, index=index))
    return {"candidates": candidates}


def _candidate_payload(request: dict[str, Any], *, index: int) -> dict[str, Any]:
    variants = [
        {
            "required_columns": ["close"],
            "lookback": 5,
            "code": "close.shift(-1) / close - 1.0",
            "rationale": "Intentional future leak used only to prove the external-provider repair loop.",
        },
        {
            "required_columns": ["close"],
            "lookback": 10,
            "code": "close.pct_change(5).shift(1)",
            "rationale": "Lagged trend persistence candidate emitted by an external command provider.",
        },
        {
            "required_columns": ["volume"],
            "lookback": 10,
            "code": "np.log1p(volume).pct_change().shift(1)",
            "rationale": "Lagged liquidity impulse candidate emitted by an external command provider.",
        },
        {
            "required_columns": ["high", "low"],
            "lookback": 10,
            "code": "(high / (low + 1e-12) - 1.0).rolling(5).mean().shift(1)",
            "rationale": "Lagged intraday range pressure candidate emitted by an external command provider.",
        },
    ]
    variant = variants[index % len(variants)]
    return {
        "name": f"fixture_external_alpha_{index:03d}",
        "level": str(request.get("level", "fixture_external_provider")),
        "agent": str(request.get("agent", "FixtureExternalProvider")),
        "theme": f"{request.get('guidance_style', 'fixture')} fixture provider alpha",
        "required_columns": variant["required_columns"],
        "lookback": variant["lookback"],
        "code": variant["code"],
        "economic_rationale": variant["rationale"],
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin_text: str | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    _ = argv
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        request_text = stdin_text if stdin_text is not None else sys.stdin.read()
        request = json.loads(request_text)
        result = fixture_response(request)
    except Exception as exc:
        print(f"fixture provider failed: {exc}", file=err)
        return 1
    print(json.dumps(result, sort_keys=True), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
