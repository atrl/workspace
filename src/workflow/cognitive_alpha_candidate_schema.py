from __future__ import annotations

from typing import Any


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


def validate_cognitive_alpha_candidate_payload(candidates: list[dict[str, Any]], *, provider_name: str) -> None:
    if not candidates:
        raise ValueError(f"{provider_name} returned no candidates")
    required = set(COGNITIVE_ALPHA_CANDIDATE_SCHEMA["properties"]["candidates"]["items"]["required"])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"{provider_name} candidate must be an object")
        missing = sorted(required - set(candidate))
        if missing:
            raise ValueError(f"{provider_name} candidate missing fields: {missing}")
        if not isinstance(candidate.get("required_columns"), list) or not candidate["required_columns"]:
            raise ValueError(f"{provider_name} candidate requires non-empty required_columns")
