from __future__ import annotations

from src.core import ArtifactManifest, SCHEMA_VERSION


def export_manifest_payload(manifests: list[ArtifactManifest]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": [manifest.to_dict() for manifest in manifests],
    }
