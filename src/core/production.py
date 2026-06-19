from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ArtifactManifest, DatasetSnapshot, stable_hash


PRODUCTION_MANIFEST_SCHEMA_VERSION = "production.manifest.v1"
PRODUCTION_REGISTRY_SCHEMA_VERSION = "production.registry.v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def build_production_manifest(
    *,
    strategy_id: str,
    notebook_path: str | Path,
    dataset: DatasetSnapshot,
    paths: Mapping[str, str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactManifest:
    manifest_paths = {str(key): str(value) for key, value in dict(paths).items()}
    manifest_paths["notebook"] = str(notebook_path)
    return ArtifactManifest(
        artifact_id=strategy_id,
        artifact_type="production_strategy",
        producer="src.core.production",
        schema_version=PRODUCTION_MANIFEST_SCHEMA_VERSION,
        dataset=dataset,
        paths=manifest_paths,
        metadata=_json_ready(dict(metadata or {})),
    )


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": PRODUCTION_REGISTRY_SCHEMA_VERSION, "strategies": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", PRODUCTION_REGISTRY_SCHEMA_VERSION)
    payload.setdefault("strategies", {})
    return payload


def _registry_record(
    *,
    strategy_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    published_at: str,
) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    dataset = manifest.get("dataset", {})
    return {
        "strategy_id": strategy_id,
        "artifact_id": manifest.get("artifact_id", ""),
        "artifact_type": manifest.get("artifact_type", ""),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "dataset_schema_hash": stable_hash(dataset),
        "status": str(metadata.get("status", "")),
        "mode": str(metadata.get("mode", "")),
        "start": str(dataset.get("start", "")),
        "end": str(dataset.get("end", "")),
        "published_at": published_at,
    }


def _write_registry(
    *,
    registry_path: Path,
    strategy_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    published_at: str,
) -> dict[str, Any]:
    registry = _load_registry(registry_path)
    strategies = registry.setdefault("strategies", {})
    record = _registry_record(
        strategy_id=strategy_id,
        manifest=manifest,
        manifest_path=manifest_path,
        published_at=published_at,
    )
    entry = strategies.setdefault(strategy_id, {"history": []})
    history = list(entry.get("history", []))
    if all(item.get("manifest_hash") != record["manifest_hash"] for item in history):
        history.append(record)
    entry["current"] = record
    entry["history"] = history
    strategies[strategy_id] = entry
    registry["registry_hash"] = stable_hash(
        {
            "schema_version": registry["schema_version"],
            "strategies": strategies,
        }
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(_json_ready(registry), ensure_ascii=False, indent=2), encoding="utf-8")
    return _json_ready(registry)


def publish_production_manifest(
    *,
    strategy_id: str,
    output_dir: str | Path,
    registry_path: str | Path | None,
    notebook_path: str | Path,
    dataset: DatasetSnapshot,
    paths: Mapping[str, str | Path],
    metadata: Mapping[str, Any] | None = None,
    published_at: str,
    manifest_filename: str = "production_manifest.json",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_production_manifest(
        strategy_id=strategy_id,
        notebook_path=notebook_path,
        dataset=dataset,
        paths=paths,
        metadata=metadata,
    )
    payload = manifest.to_dict()
    payload["manifest_hash"] = manifest.manifest_hash()
    payload["published_at"] = published_at
    manifest_path = output / manifest_filename
    manifest_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    if registry_path is not None:
        _write_registry(
            registry_path=Path(registry_path),
            strategy_id=strategy_id,
            manifest=payload,
            manifest_path=manifest_path,
            published_at=published_at,
        )
    return _json_ready(payload)
