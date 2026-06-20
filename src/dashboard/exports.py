from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core import SCHEMA_VERSION


def _manifest_dict(manifest: Any) -> dict[str, Any]:
    payload = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    if payload.get("source") != "qlib_recorder":
        raise ValueError("dashboard exports must read qlib recorder manifests")
    return payload


def write_dashboard_export(
    path: Path,
    *,
    manifests: list[Any],
    analysis_summary: dict[str, Any],
    generated_by: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": generated_by,
        "source": "qlib_recorder",
        "artifacts": [_manifest_dict(manifest) for manifest in manifests],
        "analysis_summary": analysis_summary,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_dashboard_export(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
