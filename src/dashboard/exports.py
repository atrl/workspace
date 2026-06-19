from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core import ArtifactManifest, SCHEMA_VERSION


def export_manifest_payload(manifests: list[ArtifactManifest]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": [manifest.to_dict() for manifest in manifests],
    }


def write_dashboard_export(
    path: Path,
    *,
    manifests: list[ArtifactManifest],
    model_comparison: pd.DataFrame,
    generated_by: str,
) -> dict:
    payload = export_manifest_payload(manifests)
    payload["generated_by"] = generated_by
    payload["model_comparison"] = model_comparison.to_dict("records")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_dashboard_export(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
