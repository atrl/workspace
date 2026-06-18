from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


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
