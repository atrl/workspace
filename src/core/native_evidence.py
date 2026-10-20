"""Small Qlib-native Dataset and completed-Recorder evidence helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd


def native_dataset(frame: pd.DataFrame, *, segments: Mapping[str, tuple]) -> Any:
    """Build/configure a Qlib DatasetH from grouped frame data."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("frame must have two-level grouped columns")
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("frame index must be a (datetime, instrument) MultiIndex")
    if list(frame.index.names) != ["datetime", "instrument"]:
        raise ValueError("frame index levels must be named datetime, instrument in that order")
    if not isinstance(frame.index.get_level_values("datetime"), pd.DatetimeIndex):
        raise ValueError("datetime index level must contain pandas timestamps")
    if frame.index.get_level_values("instrument").isna().any():
        raise ValueError("instrument index contains null values")
    if len(frame.columns) == 0 or any(value is None for value in frame.columns.get_level_values(0)):
        raise ValueError("frame requires named, non-empty column groups")
    if not isinstance(segments, Mapping) or not segments:
        raise ValueError("segments must be a non-empty mapping")
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=dict(segments))
    dataset.config(dump_all=True, recursive=True)
    # A schema-only template lets empty segment reads retain original groups,
    # column order, and dtypes even if a Qlib backend drops empty columns.
    dataset._native_evidence_empty = frame.iloc[:0].copy()
    return dataset


def prepare_groups(dataset: Any, segment: str) -> pd.DataFrame:
    """Prepare Qlib raw data and preserve its grouped column schema."""
    from qlib.data.dataset.handler import DataHandler

    frame = dataset.prepare(segment, col_set=DataHandler.CS_RAW)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Qlib CS_RAW prepare did not return a DataFrame")
    template = getattr(dataset, "_native_evidence_empty", None)
    if frame.empty and template is not None:
        if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2 or len(frame.columns) == 0:
            frame = template.copy()
        else:
            frame = frame.reindex(columns=template.columns)
        for column in template.columns:
            frame[column] = frame[column].astype(template[column].dtype)
        frame.columns = template.columns.copy()
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("Qlib CS_RAW result lost the two-level grouped columns")
    return frame


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _real_metrics(recorder: Any) -> dict[str, Any]:
    metrics = recorder.list_metrics()
    if not isinstance(metrics, Mapping):
        raise ValueError("blocked: completed Recorder has no readable metric map")
    valid = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            valid[str(key)] = value
        elif isinstance(value, Mapping):
            metric_value = value.get("value")
            if isinstance(metric_value, (int, float)) and math.isfinite(float(metric_value)):
                valid[str(key)] = metric_value
    if not valid:
        raise ValueError("blocked: completed Recorder needs a real finite metric recorded before closure")
    return valid


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(recorder: Any, card: Path, include_code_artifacts: bool) -> dict[str, Any]:
    card = Path(card).resolve()
    artifact_root = Path(recorder.get_local_dir()).resolve() / "artifacts"
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"blocked: Recorder artifacts directory missing: {artifact_root}")
    entries = []
    for path in sorted(p for p in artifact_root.rglob("*") if p.is_file()):
        relative = path.relative_to(artifact_root).as_posix()
        is_code = relative.split("/", 1)[0] in {"code", "sources", "implementation"} or relative.startswith("code_")
        digest = _sha256_file(path)
        entries.append({"path": path.relative_to(card).as_posix(), "sha256": digest,
                        "bytes": path.stat().st_size, "kind": "code" if is_code else "artifact"})
    code_paths = [entry["path"] for entry in entries if entry["kind"] == "code"]
    if include_code_artifacts and not code_paths:
        raise ValueError("blocked: code artifacts requested but Recorder contains none")
    return {"recorder_id": str(recorder.id), "entries": entries,
            "code_artifacts_required": bool(include_code_artifacts),
            "code_artifacts_included": bool(code_paths), "code_artifact_paths": code_paths}


def finalize_evidence(recorder: Any, *, card: str | Path, summary: Mapping[str, Any],
                      transport_binding: str | Path, workspace_root: str | Path,
                      include_code_artifacts: bool = True) -> dict[str, Any]:
    """Finalize only an already-finished Qlib run; never create/resume a Recorder."""
    card = Path(card).resolve()
    summary_path = card / "summary.json"
    manifest_path = card / "verification_manifest.json"
    if summary_path.exists() or manifest_path.exists():
        raise FileExistsError("blocked: refusing to mutate a sealed card with summary or verification manifest")
    if str(getattr(recorder, "status", "")).upper() != "FINISHED":
        raise ValueError("blocked: only a FINISHED Qlib Recorder can be finalized")
    metrics = _real_metrics(recorder)
    recorder_id = str(getattr(recorder, "id", ""))
    experiment_id = str(getattr(recorder, "experiment_id", ""))
    if not recorder_id or not experiment_id:
        raise ValueError("blocked: Recorder identity is incomplete")
    local_dir = Path(recorder.get_local_dir()).resolve()
    expected_dir = (card / "qlib_recorder" / experiment_id / recorder_id).resolve()
    if local_dir != expected_dir:
        raise ValueError("blocked: Recorder local directory does not match its card identity")
    manifest = _artifact_inventory(recorder, card, include_code_artifacts)
    # Preserve caller-supplied source/transport identity without normalizing it away.
    result = dict(summary)
    result.update(recorder_id=recorder_id, experiment_id=experiment_id,
                  recorder_local_dir=str(local_dir),
                  metrics=metrics, verification_manifest=str(manifest_path.relative_to(card)))
    from src.data.nas_market_data import archive_nas_recorder
    result["nas_evidence"] = archive_nas_recorder(Path(transport_binding), workspace_root=Path(workspace_root),
        card=card, recorder_id=recorder_id, experiment_id=experiment_id)
    _atomic_json(manifest_path, manifest)
    _atomic_json(summary_path, result)
    return result
