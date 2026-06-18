from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    data = frame.head(limit) if limit else frame
    return data.to_dict(orient="records")


def _artifact_paths(card: Path, *, attribution_card: Path | None = None) -> dict[str, str]:
    public_files = {
        "summary": "analysis_summary.json",
        "model_comparison": "model_comparison.csv",
        "regime_metrics": "regime_metrics.csv",
        "window_metrics": "window_metrics.csv",
        "window_regime_metrics": "window_regime_metrics.csv",
        "qlib_risk_metrics": "qlib_risk_metrics.csv",
        "model_effect_deltas": "model_effect_deltas.csv",
        "model_capabilities": "model_capabilities.csv",
        "feature_set_specs": "feature_set_specs.json",
        "factor_ic": "factor_ic.csv",
    }
    paths = {name: str(card / filename) for name, filename in public_files.items() if (card / filename).exists()}
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


def load_model_matrix_dashboard(
    card_dir: str | Path,
    *,
    top_n: int = 10,
    attribution_card_dir: str | Path | None = None,
) -> dict[str, Any]:
    card = Path(card_dir)
    attribution_card = Path(attribution_card_dir) if attribution_card_dir is not None else None
    summary = json.loads((card / "analysis_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(card / "model_comparison.csv")
    regime_metrics = pd.read_csv(card / "regime_metrics.csv")
    window_metrics = pd.read_csv(card / "window_metrics.csv")
    window_regime_metrics = _read_csv_if_exists(card / "window_regime_metrics.csv")
    qlib_risk_metrics = _read_csv_if_exists(card / "qlib_risk_metrics.csv")
    model_effect_deltas = _read_csv_if_exists(card / "model_effect_deltas.csv")
    model_capabilities = _read_csv_if_exists(card / "model_capabilities.csv")
    return {
        "schema_version": "dashboard.model_matrix.v1",
        "artifact_id": card.name,
        "artifact_type": "dashboard_export",
        "summary": summary,
        "model_fidelity": _load_model_fidelity(summary, card),
        "model_capabilities": _records(model_capabilities, limit=200),
        "top_models": comparison.head(top_n).to_dict(orient="records"),
        "feature_set_summary": _feature_set_summary(comparison),
        "regime_metrics": regime_metrics.to_dict(orient="records"),
        "window_metrics": window_metrics.to_dict(orient="records"),
        "window_regime_metrics": _records(window_regime_metrics, limit=200),
        "qlib_risk_metrics": _records(qlib_risk_metrics, limit=200),
        "model_effect_deltas": _records(model_effect_deltas, limit=200),
        "attribution": _load_attribution_card(attribution_card, top_n=top_n),
        "artifacts": _artifact_paths(card, attribution_card=attribution_card),
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
