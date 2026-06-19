from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core import ArtifactManifest, DatasetSnapshot

from .exports import ANALYSIS_REPORT_SCHEMA_VERSION


PUBLIC_LEGACY_SIDECARS = {
    "summary": "analysis_summary.json",
    "market_regimes": "market_regimes.csv",
    "market_regime_counts": "market_regime_counts.csv",
    "market_regime_daily_backtest": "market_regime_daily_backtest.csv",
    "model_comparison": "model_comparison.csv",
    "regime_metrics": "regime_metrics.csv",
    "window_metrics": "window_metrics.csv",
    "window_regime_metrics": "window_regime_metrics.csv",
    "qlib_return_frame": "qlib_return_frame.csv",
    "qlib_risk_metrics": "qlib_risk_metrics.csv",
    "segment_metrics": "segment_metrics.csv",
    "segment_effect_matrix": "segment_effect_matrix.csv",
    "model_effect_deltas": "model_effect_deltas.csv",
    "model_regime_window_diagnostics": "model_regime_window_diagnostics.csv",
    "model_difference_explanations": "model_difference_explanations.csv",
    "model_review_briefing": "model_review_briefing.csv",
    "failure_modes": "failure_modes.csv",
    "objective_scores": "objective_scores.csv",
    "promotion_scores": "promotion_scores.csv",
    "model_run_ledger": "model_run_ledger.csv",
    "model_run_ledger_summary": "model_run_ledger_summary.json",
    "backtest_config": "backtest_config.json",
    "walk_forward_splits": "walk_forward_splits.csv",
    "model_capabilities": "model_capabilities.csv",
    "feature_set_specs": "feature_set_specs.json",
    "factor_ic": "factor_ic.csv",
    "qlib_portfolio_report": "qlib_portfolio_report.csv",
    "qlib_risk_analysis": "qlib_risk_analysis.csv",
    "qlib_positions": "qlib_positions.csv",
    "qlib_orders": "qlib_orders.csv",
    "qlib_indicator_analysis": "qlib_indicator_analysis.csv",
}

PRIVATE_LEGACY_SIDECARS = {
    "daily_backtest": "daily_backtest.csv",
    "predictions": "predictions.csv",
    "adapter_predictions": "adapter_predictions.csv",
}

BACKFILL_BATCH_SCHEMA_VERSION = "analysis.manifest_backfill_batch.v1"


def _read_summary(card: Path, summary_filename: str) -> dict[str, Any]:
    path = card / summary_filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_public_paths(card: Path) -> dict[str, str]:
    return {key: filename for key, filename in PUBLIC_LEGACY_SIDECARS.items() if (card / filename).exists()}


def _existing_private_paths(card: Path) -> dict[str, str]:
    return {key: filename for key, filename in PRIVATE_LEGACY_SIDECARS.items() if (card / filename).exists()}


def _candidate_card_dirs(root: Path, *, manifest_filename: str) -> list[Path]:
    filenames = set(PUBLIC_LEGACY_SIDECARS.values()) | set(PRIVATE_LEGACY_SIDECARS.values()) | {manifest_filename}
    dirs = {
        path.parent
        for path in root.rglob("*")
        if path.is_file() and path.name in filenames
    }
    return sorted(dirs, key=lambda path: path.as_posix())


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


def scan_analysis_cards(
    root_dir: str | Path,
    *,
    manifest_filename: str = "analysis_manifest.json",
) -> pd.DataFrame:
    root = Path(root_dir)
    rows: list[dict[str, object]] = []
    for card in _candidate_card_dirs(root, manifest_filename=manifest_filename):
        public_paths = _existing_public_paths(card)
        private_paths = _existing_private_paths(card)
        has_manifest = (card / manifest_filename).exists()
        has_summary = "summary" in public_paths
        public_keys = sorted(public_paths)
        private_keys = sorted(private_paths)
        if has_manifest:
            status = "already_manifested"
            action = "keep_manifest"
        elif has_summary and public_paths:
            status = "backfillable"
            action = "backfill_analysis_manifest"
        else:
            status = "not_analysis_card"
            action = "ignore"

        rows.append(
            {
                "card_path": str(card),
                "relative_card_path": str(card.relative_to(root)) if card.is_relative_to(root) else str(card),
                "card_name": card.name,
                "status": status,
                "migration_action": action,
                "has_manifest": bool(has_manifest),
                "has_summary": bool(has_summary),
                "public_sidecar_count": int(len(public_paths)),
                "private_sidecar_count": int(len(private_paths)),
                "public_artifact_keys": ",".join(public_keys),
                "private_artifact_keys": ",".join(private_keys),
            }
        )
    columns = [
        "card_path",
        "relative_card_path",
        "card_name",
        "status",
        "migration_action",
        "has_manifest",
        "has_summary",
        "public_sidecar_count",
        "private_sidecar_count",
        "public_artifact_keys",
        "private_artifact_keys",
    ]
    return pd.DataFrame(rows, columns=columns)


def backfill_legacy_analysis_cards(
    root_dir: str | Path,
    *,
    dataset_name: str,
    universe: tuple[str, ...],
    calendar_path: str,
    manifest_filename: str = "analysis_manifest.json",
    overwrite: bool = False,
) -> dict[str, Any]:
    scan = scan_analysis_cards(root_dir, manifest_filename=manifest_filename)
    cards: list[dict[str, object]] = []
    backfilled = 0
    skipped = 0
    for row in scan.to_dict("records"):
        if row["status"] != "backfillable":
            if row["status"] in {"already_manifested", "not_analysis_card"}:
                skipped += 1
                cards.append(
                    {
                        "card_name": row["card_name"],
                        "card_path": row["card_path"],
                        "status": row["status"],
                        "action": "skipped",
                        "reason": row["status"],
                    }
                )
            continue

        card = Path(str(row["card_path"]))
        payload = backfill_analysis_manifest(
            card,
            dataset_name=dataset_name,
            universe=universe,
            calendar_path=calendar_path,
            data_paths=(str(card),),
            manifest_filename=manifest_filename,
            overwrite=overwrite,
        )
        backfilled += 1
        cards.append(
            {
                "card_name": row["card_name"],
                "card_path": str(card),
                "status": row["status"],
                "action": "backfilled",
                "manifest_hash": payload["manifest_hash"],
            }
        )

    return {
        "schema_version": BACKFILL_BATCH_SCHEMA_VERSION,
        "root_dir": str(Path(root_dir)),
        "backfilled": backfilled,
        "skipped": skipped,
        "cards": cards,
    }


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
