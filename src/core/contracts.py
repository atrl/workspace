from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


DASHBOARD_MANIFEST_ALLOWED_KINDS = ("analysis_manifest", "artifact_manifest")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSnapshot:
    name: str
    universe: tuple[str, ...]
    start: str
    end: str
    calendar_path: str
    data_paths: tuple[str, ...]
    schema_version: str = "dataset.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def schema_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    artifact_type: str
    producer: str
    schema_version: str
    dataset: DatasetSnapshot
    paths: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def manifest_hash(self) -> str:
        return stable_hash(self.to_dict())


def build_dashboard_manifest_source_contract(
    manifest_source: Any,
    *,
    required: bool = True,
    allowed_kinds: tuple[str, ...] = DASHBOARD_MANIFEST_ALLOWED_KINDS,
) -> dict[str, Any]:
    allowed = tuple(str(kind) for kind in allowed_kinds)
    base = {
        "schema_version": "core.dashboard_manifest_source_contract.v1",
        "required": bool(required),
        "status": "not_checked" if not required else "missing",
        "is_complete": False,
        "kind": "",
        "path": "",
        "manifest_hash": "",
        "producer": "",
        "source_schema_version": "",
        "allowed_kinds": list(allowed),
        "missing_fields": [],
        "error": "",
        "readiness_detail": "missing / missing hash ",
    }
    if not isinstance(manifest_source, dict):
        return base

    kind = str(manifest_source.get("kind", ""))
    manifest_path = str(manifest_source.get("path", ""))
    manifest_hash = str(manifest_source.get("manifest_hash", ""))
    producer = str(manifest_source.get("producer", ""))
    contract = {
        **base,
        "status": "missing",
        "kind": kind,
        "path": manifest_path,
        "manifest_hash": manifest_hash,
        "producer": producer,
        "source_schema_version": str(manifest_source.get("schema_version", "")),
    }

    if kind not in allowed:
        contract["status"] = "weak_kind"
        contract["error"] = f"kind {kind} not allowed"
        contract["readiness_detail"] = f"{kind} / weak_kind hash {manifest_hash[:10]}"
        return contract

    missing_fields = [
        field
        for field, value in (
            ("path", manifest_path),
            ("manifest_hash", manifest_hash),
            ("producer", producer),
        )
        if not value
    ]
    if missing_fields:
        contract["status"] = "incomplete"
        contract["missing_fields"] = missing_fields
        contract["error"] = "missing " + ",".join(missing_fields)
        contract["readiness_detail"] = f"{kind} / incomplete hash {manifest_hash[:10]}"
        return contract

    contract["status"] = "complete"
    contract["is_complete"] = True
    contract["readiness_detail"] = f"{kind} / complete hash {manifest_hash[:10]}"
    return contract
