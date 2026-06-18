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


def _artifact_paths(card: Path) -> dict[str, str]:
    public_files = {
        "summary": "analysis_summary.json",
        "model_comparison": "model_comparison.csv",
        "regime_metrics": "regime_metrics.csv",
        "window_metrics": "window_metrics.csv",
        "qlib_risk_metrics": "qlib_risk_metrics.csv",
        "model_effect_deltas": "model_effect_deltas.csv",
        "feature_set_specs": "feature_set_specs.json",
        "factor_ic": "factor_ic.csv",
    }
    return {name: str(card / filename) for name, filename in public_files.items() if (card / filename).exists()}


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


def load_model_matrix_dashboard(card_dir: str | Path, *, top_n: int = 10) -> dict[str, Any]:
    card = Path(card_dir)
    summary = json.loads((card / "analysis_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(card / "model_comparison.csv")
    regime_metrics = pd.read_csv(card / "regime_metrics.csv")
    window_metrics = pd.read_csv(card / "window_metrics.csv")
    qlib_risk_metrics = _read_csv_if_exists(card / "qlib_risk_metrics.csv")
    model_effect_deltas = _read_csv_if_exists(card / "model_effect_deltas.csv")
    return {
        "schema_version": "dashboard.model_matrix.v1",
        "artifact_id": card.name,
        "artifact_type": "dashboard_export",
        "summary": summary,
        "top_models": comparison.head(top_n).to_dict(orient="records"),
        "feature_set_summary": _feature_set_summary(comparison),
        "regime_metrics": regime_metrics.to_dict(orient="records"),
        "window_metrics": window_metrics.to_dict(orient="records"),
        "qlib_risk_metrics": _records(qlib_risk_metrics, limit=200),
        "model_effect_deltas": _records(model_effect_deltas, limit=200),
        "artifacts": _artifact_paths(card),
    }


def write_model_matrix_dashboard_export(
    card_dir: str | Path,
    *,
    output_path: str | Path,
    top_n: int = 10,
) -> dict[str, Any]:
    payload = load_model_matrix_dashboard(card_dir, top_n=top_n)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
