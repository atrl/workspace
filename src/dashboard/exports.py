from __future__ import annotations

from typing import Any

from src.core import ArtifactManifest


def dashboard_payload(manifest: ArtifactManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    return {
        "artifact_id": payload["artifact_id"],
        "artifact_type": payload["artifact_type"],
        "schema_version": payload["schema_version"],
        "paths": payload["paths"],
        "dataset": payload["dataset"],
    }
