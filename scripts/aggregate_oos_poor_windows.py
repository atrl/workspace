#!/usr/bin/env python3
"""Aggregate OOS poor-window diagnostics across profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.oos_diagnostics import (  # noqa: E402
    aggregate_feature_differences,
    aggregate_poor_windows,
    attribute_gate_gaps,
    normalise_gate_metrics_from_selection,
    render_gate_gap_review,
    summarize_gate_gap_attribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        type=Path,
        help="Directory containing poor_windows.csv and feature_differences.csv. Repeat for multiple profiles.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for each --input-dir. Defaults to the input directory name.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-excess", type=float, default=30.0)
    parser.add_argument("--target-max-dd", type=float, default=15.0)
    parser.add_argument("--top-features", type=int, default=30)
    parser.add_argument(
        "--selection-json",
        action="append",
        default=[],
        type=Path,
        help="Optional runtime selection.json to classify profile/window gate misses. Repeat for multiple profiles.",
    )
    parser.add_argument(
        "--selection-label",
        action="append",
        default=[],
        help="Optional label for each --selection-json. Defaults to the selection parent directory name.",
    )
    return parser.parse_args()


def _labels(input_dirs: list[Path], labels: list[str]) -> list[str]:
    if not labels:
        return [path.name for path in input_dirs]
    if len(labels) != len(input_dirs):
        raise ValueError("--label count must match --input-dir count")
    return labels


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _render_review(
    clusters: pd.DataFrame,
    features: pd.DataFrame,
    gate_gaps: pd.DataFrame | None = None,
    gate_summary: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# Aggregated OOS Poor-Window Diagnostic",
        "",
        "## Shared Bad-Window Clusters",
        "",
        "| cluster | start | end | profiles | type | mean excess | max dd | avg pos | next path |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    if clusters.empty:
        lines.append("| none | none | none | 0 | none | 0.00pp | 0.00% | 0.000 | none |")
    else:
        for row in clusters.itertuples(index=False):
            lines.append(
                "| "
                f"{row.cluster_id} | {row.cluster_start} | {row.cluster_end} | "
                f"{int(row.profile_count)} | {row.dominant_failure_type} | "
                f"{float(row.mean_excess_return):.2f}pp | {float(row.max_max_dd):.2f}% | "
                f"{float(row.mean_avg_pos):.3f} | {row.primary_optimization_path} |"
            )
    lines.extend(
        [
            "",
            "## Stable Feature Deltas",
            "",
            "| feature | profiles | mean delta_z | consistency | mean spearman | score |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if features.empty:
        lines.append("| none | 0 | 0.000 | 0.000 | nan | 0.000 |")
    else:
        for row in features.itertuples(index=False):
            corr = getattr(row, "mean_spearman_to_excess", float("nan"))
            corr_text = "nan" if pd.isna(corr) else f"{float(corr):.3f}"
            lines.append(
                "| "
                f"{row.feature} | {int(row.profile_count)} | {float(row.mean_delta_z):.3f} | "
                f"{float(row.direction_consistency):.3f} | {corr_text} | "
                f"{float(row.stability_score):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Use shared clusters to decide whether the next loop is participation, drawdown control, or model selection.",
            "- Treat feature deltas as mechanism hypotheses until they pass IC, single-signal, and walk-forward gates.",
        ]
    )
    if gate_gaps is not None and gate_summary is not None:
        lines.extend(["", render_gate_gap_review(gate_gaps, gate_summary).strip()])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    labels = _labels(args.input_dir, args.label)
    selection_labels = _labels(args.selection_json, args.selection_label) if args.selection_json else []
    windows_by_profile: dict[str, pd.DataFrame] = {}
    features_by_profile: dict[str, pd.DataFrame] = {}
    for label, directory in zip(labels, args.input_dir, strict=True):
        windows_by_profile[label] = _read_csv(directory / "poor_windows.csv")
        features_by_profile[label] = _read_csv(directory / "feature_differences.csv")

    clusters = aggregate_poor_windows(
        windows_by_profile,
        target_excess=args.target_excess,
        target_max_dd=args.target_max_dd,
    )
    features = aggregate_feature_differences(features_by_profile, top_n=args.top_features)
    metrics_by_profile: dict[str, dict[str, dict[str, float]]] = {}
    for label, selection_path in zip(selection_labels, args.selection_json, strict=True):
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        metrics_by_profile[label] = normalise_gate_metrics_from_selection(selection)
    gate_gaps = attribute_gate_gaps(
        metrics_by_profile,
        target_relative_return=args.target_excess,
        target_max_dd=args.target_max_dd,
    )
    gate_summary = summarize_gate_gap_attribution(gate_gaps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = args.output_dir / "poor_window_clusters.csv"
    feature_path = args.output_dir / "feature_delta_clusters.csv"
    review_path = args.output_dir / "aggregate_oos_review.md"
    manifest_path = args.output_dir / "manifest.json"
    clusters.to_csv(cluster_path, index=False)
    features.to_csv(feature_path, index=False)
    if args.selection_json:
        gate_gap_path = args.output_dir / "gate_gap_attribution.csv"
        gate_summary_path = args.output_dir / "gate_gap_profile_summary.csv"
        gate_gaps.to_csv(gate_gap_path, index=False)
        gate_summary.to_csv(gate_summary_path, index=False)
    else:
        gate_gap_path = None
        gate_summary_path = None
    review_path.write_text(
        _render_review(
            clusters,
            features,
            gate_gaps if args.selection_json else None,
            gate_summary if args.selection_json else None,
        ),
        encoding="utf-8",
    )
    manifest = {
        "input_dirs": [str(path) for path in args.input_dir],
        "labels": labels,
        "selection_json": [str(path) for path in args.selection_json],
        "selection_labels": selection_labels,
        "target_excess": args.target_excess,
        "target_max_dd": args.target_max_dd,
        "artifacts": {
            "clusters": str(cluster_path),
            "features": str(feature_path),
            "review": str(review_path),
        },
        "cluster_count": int(len(clusters)),
        "feature_count": int(len(features)),
        "gate_gap_count": int(len(gate_gaps)),
    }
    if gate_gap_path is not None and gate_summary_path is not None:
        manifest["artifacts"]["gate_gaps"] = str(gate_gap_path)
        manifest["artifacts"]["gate_summary"] = str(gate_summary_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        "oos_aggregate "
        f"profiles={len(labels)} clusters={len(clusters)} features={len(features)} "
        f"gate_gaps={len(gate_gaps)} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
