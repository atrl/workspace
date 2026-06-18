from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core import ArtifactManifest, DatasetSnapshot

from .exports import ANALYSIS_REPORT_SCHEMA_VERSION


PUBLIC_LEGACY_SIDECARS = {
    "summary": "analysis_summary.json",
    "model_comparison": "model_comparison.csv",
    "regime_metrics": "regime_metrics.csv",
    "window_metrics": "window_metrics.csv",
    "window_regime_metrics": "window_regime_metrics.csv",
    "qlib_return_frame": "qlib_return_frame.csv",
    "qlib_risk_metrics": "qlib_risk_metrics.csv",
    "segment_metrics": "segment_metrics.csv",
    "model_effect_deltas": "model_effect_deltas.csv",
    "model_capabilities": "model_capabilities.csv",
    "feature_set_specs": "feature_set_specs.json",
    "factor_ic": "factor_ic.csv",
    "qlib_portfolio_report": "qlib_portfolio_report.csv",
    "qlib_risk_analysis": "qlib_risk_analysis.csv",
    "qlib_positions": "qlib_positions.csv",
    "qlib_orders": "qlib_orders.csv",
    "qlib_indicator_analysis": "qlib_indicator_analysis.csv",
}


def _read_summary(card: Path, summary_filename: str) -> dict[str, Any]:
    path = card / summary_filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_public_paths(card: Path) -> dict[str, str]:
    return {key: filename for key, filename in PUBLIC_LEGACY_SIDECARS.items() if (card / filename).exists()}


def _csv_count(path: Path) -> int:
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _table_counts(card: Path, paths: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, filename in paths.items():
        if filename.endswith(".csv"):
            counts[key] = _csv_count(card / filename)
    return counts


def _dataset_snapshot(
    *,
    summary: dict[str, Any],
    dataset_name: str,
    universe: tuple[str, ...],
    calendar_path: str,
    data_paths: tuple[str, ...],
) -> DatasetSnapshot:
    start = str(summary.get("start") or summary.get("test_start") or "")
    end = str(summary.get("end") or summary.get("test_end") or "")
    return DatasetSnapshot(
        name=dataset_name,
        universe=universe,
        start=start,
        end=end,
        calendar_path=calendar_path,
        data_paths=data_paths,
        metadata={"source": "analysis_manifest_backfill"},
    )


def backfill_analysis_manifest(
    card_dir: str | Path,
    *,
    dataset_name: str,
    universe: tuple[str, ...],
    calendar_path: str,
    data_paths: tuple[str, ...],
    artifact_id: str | None = None,
    manifest_filename: str = "analysis_manifest.json",
    overwrite: bool = False,
) -> dict[str, Any]:
    card = Path(card_dir)
    manifest_path = card / manifest_filename
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_filename} already exists: {manifest_path}")

    paths = _existing_public_paths(card)
    if "summary" not in paths:
        raise FileNotFoundError(f"analysis summary sidecar not found under {card}")

    summary = _read_summary(card, paths["summary"])
    schema_version = str(summary.get("schema_version") or ANALYSIS_REPORT_SCHEMA_VERSION)
    metadata = {
        "generated_from": "legacy_sidecars",
        "analysis_report_schema_version": schema_version,
        "table_counts": _table_counts(card, paths),
    }
    manifest = ArtifactManifest(
        artifact_id=artifact_id or card.name,
        artifact_type="analysis_report",
        producer="src.analysis.manifest_backfill",
        schema_version="artifact.v1",
        dataset=_dataset_snapshot(
            summary=summary,
            dataset_name=dataset_name,
            universe=universe,
            calendar_path=calendar_path,
            data_paths=data_paths,
        ),
        paths=paths,
        metadata=metadata,
    )
    payload = manifest.to_dict()
    payload["manifest_hash"] = manifest.manifest_hash()
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
