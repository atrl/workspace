from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import build_analysis_overview


PUBLIC_ARTIFACT_KEYS = {
    "summary",
    "model_comparison",
    "regime_metrics",
    "window_metrics",
    "window_regime_metrics",
    "qlib_risk_metrics",
    "objective_scores",
    "promotion_scores",
    "segment_effect_matrix",
    "model_effect_deltas",
    "model_review_briefing",
    "failure_modes",
    "backtest_config",
    "walk_forward_splits",
    "model_capabilities",
    "feature_set_specs",
    "factor_ic",
}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    data = frame.head(limit) if limit else frame
    return data.to_dict(orient="records")


def _read_manifest_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_card_path(card: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else card / path


def _default_public_paths(card: Path) -> dict[str, str]:
    public_files = {
        "summary": "analysis_summary.json",
        "model_comparison": "model_comparison.csv",
        "regime_metrics": "regime_metrics.csv",
        "window_metrics": "window_metrics.csv",
        "window_regime_metrics": "window_regime_metrics.csv",
        "qlib_risk_metrics": "qlib_risk_metrics.csv",
        "objective_scores": "objective_scores.csv",
        "promotion_scores": "promotion_scores.csv",
        "segment_effect_matrix": "segment_effect_matrix.csv",
        "model_effect_deltas": "model_effect_deltas.csv",
        "model_review_briefing": "model_review_briefing.csv",
        "failure_modes": "failure_modes.csv",
        "backtest_config": "backtest_config.json",
        "walk_forward_splits": "walk_forward_splits.csv",
        "model_capabilities": "model_capabilities.csv",
        "feature_set_specs": "feature_set_specs.json",
        "factor_ic": "factor_ic.csv",
    }
    return {name: filename for name, filename in public_files.items() if (card / filename).exists()}


def _manifest_public_paths(manifest: dict[str, Any]) -> dict[str, str]:
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        return {}
    return {str(name): str(path) for name, path in paths.items() if name in PUBLIC_ARTIFACT_KEYS}


def _card_sources(card: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    analysis_manifest = _read_manifest_if_exists(card / "analysis_manifest.json")
    artifact_manifest = _read_manifest_if_exists(card / "artifact_manifest.json")
    paths = _default_public_paths(card)
    paths.update(_manifest_public_paths(artifact_manifest))
    paths.update(_manifest_public_paths(analysis_manifest))
    if (card / "analysis_manifest.json").exists():
        paths["analysis_manifest"] = "analysis_manifest.json"
    if (card / "artifact_manifest.json").exists():
        paths["artifact_manifest"] = "artifact_manifest.json"
    return analysis_manifest, artifact_manifest, paths


def _manifest_source(card: Path, *, analysis_manifest: dict[str, Any], artifact_manifest: dict[str, Any]) -> dict[str, str]:
    if analysis_manifest:
        return {
            "kind": "analysis_manifest",
            "path": str(card / "analysis_manifest.json"),
            "schema_version": str(analysis_manifest.get("schema_version", "")),
            "manifest_hash": str(analysis_manifest.get("manifest_hash", "")),
            "producer": str(analysis_manifest.get("producer", "")),
        }
    if artifact_manifest:
        return {
            "kind": "artifact_manifest",
            "path": str(card / "artifact_manifest.json"),
            "schema_version": str(artifact_manifest.get("schema_version", "")),
            "manifest_hash": str(artifact_manifest.get("manifest_hash", "")),
            "producer": str(artifact_manifest.get("producer", "")),
        }
    return {
        "kind": "fixed_sidecars",
        "path": str(card),
        "schema_version": "",
        "manifest_hash": "",
        "producer": "",
    }


def _path_for(card: Path, paths: dict[str, str], key: str, fallback: str) -> Path:
    return _resolve_card_path(card, paths.get(key, fallback)) or (card / fallback)


def _read_csv_from_paths(card: Path, paths: dict[str, str], key: str, fallback: str) -> pd.DataFrame:
    return pd.read_csv(_path_for(card, paths, key, fallback))


def _read_optional_csv_from_paths(card: Path, paths: dict[str, str], key: str, fallback: str) -> pd.DataFrame:
    path = _path_for(card, paths, key, fallback)
    return _read_csv_if_exists(path)


def _artifact_paths(
    card: Path,
    *,
    public_paths: dict[str, str] | None = None,
    attribution_card: Path | None = None,
) -> dict[str, str]:
    paths = {
        name: str(resolved)
        for name, value in (public_paths or _default_public_paths(card)).items()
        if (resolved := _resolve_card_path(card, value)) is not None and resolved.exists()
    }
    if attribution_card is not None:
        attribution_files = {
            "attribution_summary": "analysis_summary.json",
            "attribution_feature": "feature_attribution.csv",
            "attribution_model_summaries": "model_attribution_summaries.json",
            "attribution_top_features": "top_features_by_model.csv",
        }
        paths.update(
            {
                name: str(attribution_card / filename)
                for name, filename in attribution_files.items()
                if (attribution_card / filename).exists()
            }
        )
    return paths


def _feature_set_summary(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    if comparison.empty or "feature_set" not in comparison.columns:
        return []
    rows: list[dict[str, Any]] = []
    for feature_set, group in comparison.groupby("feature_set", sort=True):
        ordered = group.sort_values(["stability_score", "relative_return_pct"], ascending=[False, False])
        best = ordered.iloc[0]
        rows.append(
            {
                "feature_set": str(feature_set),
                "variant_count": int(group["model"].nunique()),
                "best_model": str(best["model"]),
                "best_base_model": str(best.get("base_model", best["model"])),
                "avg_return_pct": float(group["return_pct"].astype(float).mean()),
                "avg_relative_return_pct": float(group["relative_return_pct"].astype(float).mean()),
                "avg_stability_score": float(group["stability_score"].astype(float).mean()),
            }
        )
    return sorted(rows, key=lambda item: item["avg_stability_score"], reverse=True)


def _load_model_fidelity(summary: dict[str, Any], card: Path) -> dict[str, Any]:
    if isinstance(summary.get("model_fidelity"), dict):
        return summary["model_fidelity"]
    manifest_path = card / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_fidelity = manifest.get("metadata", {}).get("model_fidelity")
        if isinstance(model_fidelity, dict):
            return model_fidelity
    return {
        "schema_version": "model.fidelity.v1",
        "model_count": int(summary.get("model_count", 0) or summary.get("base_model_count", 0) or 0),
        "fidelity_counts": {},
        "proxy_models": [],
        "native_models": [],
        "promotion_blocker_count": 0,
    }


def _normalize_attribution_kinds(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple, set)):
        return {str(item): None for item in value}
    return {}


def _load_attribution_card(attribution_card: Path | None, *, top_n: int) -> dict[str, Any]:
    if attribution_card is None:
        return {
            "status": "not_provided",
            "schema_version": "analysis.model_attribution.batch.v1",
            "models": [],
            "proxy_models": [],
            "attribution_kinds": {},
            "top_features_by_model": [],
        }
    summary_path = attribution_card / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    top_features = _read_csv_if_exists(attribution_card / "top_features_by_model.csv")
    return {
        "status": str(summary.get("status", "missing_summary")),
        "schema_version": str(summary.get("schema_version", "analysis.model_attribution.batch.v1")),
        "models": list(summary.get("models", [])),
        "proxy_models": list(summary.get("proxy_models", [])),
        "attribution_kinds": _normalize_attribution_kinds(summary.get("attribution_kinds", {})),
        "top_features_by_model": _records(top_features, limit=top_n),
    }


def _analysis_overview(
    summary: dict[str, Any],
    *,
    segment_effect_matrix: pd.DataFrame,
    model_review_briefing: pd.DataFrame,
) -> dict[str, Any]:
    overview = summary.get("analysis_overview")
    if isinstance(overview, dict):
        return dict(overview)
    analysis_summary = summary.get("analysis_summary")
    if isinstance(analysis_summary, dict):
        return build_analysis_overview(
            analysis_summary,
            segment_effect_matrix=segment_effect_matrix,
            model_review_briefing=model_review_briefing,
        )
    return build_analysis_overview(
        {},
        segment_effect_matrix=segment_effect_matrix,
        model_review_briefing=model_review_briefing,
    )


def load_model_matrix_dashboard(
    card_dir: str | Path,
    *,
    top_n: int = 10,
    attribution_card_dir: str | Path | None = None,
) -> dict[str, Any]:
    card = Path(card_dir)
    attribution_card = Path(attribution_card_dir) if attribution_card_dir is not None else None
    analysis_manifest, artifact_manifest, public_paths = _card_sources(card)
    summary = json.loads(_path_for(card, public_paths, "summary", "analysis_summary.json").read_text(encoding="utf-8"))
    comparison = _read_csv_from_paths(card, public_paths, "model_comparison", "model_comparison.csv")
    regime_metrics = _read_csv_from_paths(card, public_paths, "regime_metrics", "regime_metrics.csv")
    window_metrics = _read_csv_from_paths(card, public_paths, "window_metrics", "window_metrics.csv")
    window_regime_metrics = _read_optional_csv_from_paths(
        card, public_paths, "window_regime_metrics", "window_regime_metrics.csv"
    )
    qlib_risk_metrics = _read_optional_csv_from_paths(card, public_paths, "qlib_risk_metrics", "qlib_risk_metrics.csv")
    objective_scores = _read_optional_csv_from_paths(card, public_paths, "objective_scores", "objective_scores.csv")
    promotion_scores = _read_optional_csv_from_paths(card, public_paths, "promotion_scores", "promotion_scores.csv")
    segment_effect_matrix = _read_optional_csv_from_paths(
        card, public_paths, "segment_effect_matrix", "segment_effect_matrix.csv"
    )
    model_effect_deltas = _read_optional_csv_from_paths(
        card, public_paths, "model_effect_deltas", "model_effect_deltas.csv"
    )
    model_review_briefing = _read_optional_csv_from_paths(
        card, public_paths, "model_review_briefing", "model_review_briefing.csv"
    )
    failure_modes = _read_optional_csv_from_paths(card, public_paths, "failure_modes", "failure_modes.csv")
    backtest_config_path = _path_for(card, public_paths, "backtest_config", "backtest_config.json")
    backtest_config = _read_manifest_if_exists(backtest_config_path)
    walk_forward_splits = _read_optional_csv_from_paths(
        card, public_paths, "walk_forward_splits", "walk_forward_splits.csv"
    )
    model_capabilities = _read_optional_csv_from_paths(
        card, public_paths, "model_capabilities", "model_capabilities.csv"
    )
    manifest_source = analysis_manifest or artifact_manifest
    return {
        "schema_version": "dashboard.model_matrix.v1",
        "artifact_id": str(manifest_source.get("artifact_id", card.name)),
        "artifact_type": "dashboard_export",
        "manifest_source": _manifest_source(
            card,
            analysis_manifest=analysis_manifest,
            artifact_manifest=artifact_manifest,
        ),
        "summary": summary,
        "analysis_overview": _analysis_overview(
            summary,
            segment_effect_matrix=segment_effect_matrix,
            model_review_briefing=model_review_briefing,
        ),
        "model_fidelity": _load_model_fidelity(summary, card),
        "model_capabilities": _records(model_capabilities, limit=200),
        "top_models": comparison.head(top_n).to_dict(orient="records"),
        "feature_set_summary": _feature_set_summary(comparison),
        "regime_metrics": regime_metrics.to_dict(orient="records"),
        "window_metrics": window_metrics.to_dict(orient="records"),
        "window_regime_metrics": _records(window_regime_metrics, limit=200),
        "qlib_risk_metrics": _records(qlib_risk_metrics, limit=200),
        "objective": {
            "scores": _records(objective_scores, limit=200),
            "promotion_scores": _records(promotion_scores, limit=200),
        },
        "segment_effect_matrix": _records(segment_effect_matrix, limit=200),
        "model_effect_deltas": _records(model_effect_deltas, limit=200),
        "model_review_briefing": _records(model_review_briefing, limit=200),
        "failure_modes": _records(failure_modes, limit=100),
        "backtest": {
            "config": backtest_config,
            "walk_forward_splits": _records(walk_forward_splits, limit=200),
        },
        "attribution": _load_attribution_card(attribution_card, top_n=top_n),
        "artifacts": _artifact_paths(card, public_paths=public_paths, attribution_card=attribution_card),
    }


def write_model_matrix_dashboard_export(
    card_dir: str | Path,
    *,
    output_path: str | Path,
    top_n: int = 10,
    attribution_card_dir: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_model_matrix_dashboard(card_dir, top_n=top_n, attribution_card_dir=attribution_card_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
