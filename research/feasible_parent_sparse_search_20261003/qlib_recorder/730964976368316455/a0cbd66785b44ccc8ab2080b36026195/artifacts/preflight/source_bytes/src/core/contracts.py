from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.1"
QLIB_SOURCES = {"qlib_recorder", "qlib_dataset", "qlib.data.D.features"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ArtifactManifest:
    kind: str
    path: str
    source: str
    provider_uri: str
    recorder_id: str
    date_range: tuple[str, str]
    purpose: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in QLIB_SOURCES:
            raise ValueError(f"artifact source must be qlib-derived: {sorted(QLIB_SOURCES)}")
        if not self.provider_uri:
            raise ValueError("artifact manifest requires qlib provider_uri")
        if not self.recorder_id:
            raise ValueError("artifact manifest requires qlib recorder_id")

    @property
    def artifact_id(self) -> str:
        return stable_hash(self.to_dict(include_artifact_id=False))

    def to_dict(self, *, include_artifact_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "path": self.path,
            "source": self.source,
            "provider_uri": self.provider_uri,
            "recorder_id": self.recorder_id,
            "date_range": list(self.date_range),
            "purpose": self.purpose,
            "metadata": _json_safe(self.metadata),
        }
        if include_artifact_id:
            payload["artifact_id"] = self.artifact_id
        return payload


@dataclass(frozen=True)
class RunConfig:
    name: str
    surface: str
    provider_uri: str
    market: str
    region: str
    as_of: str
    output_dir: str
    recorder_id: str
    artifacts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        active_surfaces = {"research", "production", "data"}
        if self.surface not in active_surfaces:
            raise ValueError(f"run config surface must be an active surface: {sorted(active_surfaces)}")
        first_part = self.output_dir.split("/", 1)[0]
        retired_surfaces = {"dashboard.html", "dashboard_data", "web", "scripts", "codex_exec_plans", ".codex"}
        if first_part in retired_surfaces:
            raise ValueError(f"run config output_dir points to a retired surface: {self.output_dir}")
        if first_part != self.surface:
            raise ValueError(f"run config output_dir must be under its active surface: {self.surface}")
        if not self.provider_uri or not self.recorder_id:
            raise ValueError("run config requires qlib provider_uri and recorder_id")

    @property
    def config_id(self) -> str:
        return stable_hash(self.to_manifest(include_config_id=False))

    def to_manifest(self, *, include_config_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "surface": self.surface,
            "provider_uri": self.provider_uri,
            "market": self.market,
            "region": self.region,
            "as_of": self.as_of,
            "output_dir": self.output_dir,
            "recorder_id": self.recorder_id,
            "artifacts": sorted(self.artifacts),
            "qlib_native": True,
        }
        if include_config_id:
            payload["config_id"] = self.config_id
        return payload
