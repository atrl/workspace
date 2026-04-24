#!/usr/bin/env python3
"""Evaluate a seed-ensemble of actors on the fixed-split test slices.

This is the T1-A implementation of the next-step plan. It:
- loads N per-seed actors via `app.artifacts.build_actor_ensemble`
- wraps them in `EnsembleActor` (mean-logits aggregation by default)
- evaluates the ensemble + each individual member on
  test_proxy / test_2020 / test_2022 / test_2023_2024
- writes a markdown + JSON comparison report

Usage:
    qlib_venv/bin/python scripts/evaluate_ensemble.py \
        --members balanced_rule_residual_v16 \
                 balanced_rule_residual_v18 \
                 balanced_rule_residual_v19 \
        --anchor balanced_rule_residual_v19 \
        --output runtime/model/reviews/ensemble_v16_v18_v19

Reads from existing artifacts only; no training is triggered.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.artifacts import build_actor_ensemble, load_actor_bundle
from src.soxl_quant.app.evaluate import slice_frame
from src.soxl_quant.config import load_profile
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.policy.evaluation import evaluate_split
from src.soxl_quant.policy.positioning import action_levels_for_profile
from src.soxl_quant.settings import (
    QLIB_DATA_PATH,
    TEST_SEG,
    TRAIN_SEG,
    VALID_SEG,
)

SLICES: Dict[str, Tuple[str, str]] = {
    "test_proxy": TEST_SEG,
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


def _evaluate_actor(actor, test_df, profile) -> Dict[str, Dict[str, float]]:
    action_levels = action_levels_for_profile(profile)
    delta_levels = (
        np.asarray(profile.model.delta_levels, dtype=np.float32)
        if profile.model.use_hybrid_overlay
        else None
    )
    out: Dict[str, Dict[str, float]] = {}
    for slice_name, seg in SLICES.items():
        segment = slice_frame(test_df, seg)
        if len(segment) <= 120:
            continue
        metrics = evaluate_split(
            actor,
            segment,
            position_cap_config=profile.position_cap,
            execution_config=profile.execution,
            stress_threshold=profile.selection_score.stress_threshold,
            action_levels=action_levels,
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=delta_levels,
        )
        out[slice_name] = metrics
    return out


def _format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"return={metrics['return']:.2f}% sharpe={metrics['sharpe']:.3f} "
        f"maxDD={metrics['max_dd']:.2f}% avg_pos={metrics.get('avg_pos', 0.0):.3f}"
    )


def _render_markdown(
    output_path: Path,
    members: List[str],
    anchor: str,
    member_metrics: Dict[str, Dict[str, Dict[str, float]]],
    ensemble_metrics: Dict[str, Dict[str, float]],
) -> None:
    lines: List[str] = []
    lines.append(f"# Seed Ensemble vs Single-Actor ({anchor} anchor)")
    lines.append("")
    lines.append(f"Members: {', '.join(members)}")
    lines.append(f"Aggregation: mean-logits softmax -> argmax")
    lines.append("")

    for slice_name in SLICES.keys():
        lines.append(f"## {slice_name}")
        lines.append("")
        lines.append("| actor | return% | sharpe | maxDD% | avg_pos |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in members:
            metrics = member_metrics.get(m, {}).get(slice_name)
            if metrics is None:
                continue
            lines.append(
                f"| {m} | {metrics['return']:.2f} | {metrics['sharpe']:.3f} | "
                f"{metrics['max_dd']:.2f} | {metrics.get('avg_pos', 0.0):.3f} |"
            )
        ens_metrics = ensemble_metrics.get(slice_name)
        if ens_metrics is not None:
            lines.append(
                f"| **ensemble** | **{ens_metrics['return']:.2f}** | "
                f"**{ens_metrics['sharpe']:.3f}** | **{ens_metrics['max_dd']:.2f}** | "
                f"**{ens_metrics.get('avg_pos', 0.0):.3f}** |"
            )
        lines.append("")

        returns = [
            member_metrics[m][slice_name]["return"]
            for m in members
            if slice_name in member_metrics.get(m, {})
        ]
        if len(returns) >= 2:
            spread = max(returns) - min(returns)
            stdev = statistics.pstdev(returns)
            lines.append(f"- single-actor return spread: **{spread:.2f}pp**")
            lines.append(f"- single-actor return stdev: **{stdev:.2f}pp**")
            if ens_metrics is not None:
                lines.append(
                    f"- ensemble - mean(members) return: "
                    f"**{ens_metrics['return'] - statistics.mean(returns):.2f}pp**"
                )
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--members",
        nargs="+",
        required=True,
        help="Profile names (actors) to ensemble",
    )
    parser.add_argument(
        "--anchor",
        required=True,
        help="Profile name used to supply position_cap/execution/model config for evaluation",
    )
    parser.add_argument(
        "--method",
        default="mean_logits",
        choices=("mean_logits", "mean_action"),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output stem (writes <stem>.md and <stem>.json)",
    )
    args = parser.parse_args()

    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(args.anchor)
    dataset = build_dataset(
        leveraged_etf="SOXL",
        underlying_etf="SOXX",
        leverage_ratio=3.0,
        train_seg=TRAIN_SEG,
        valid_seg=VALID_SEG,
        test_seg=TEST_SEG,
    )
    test_df = segment_frame(dataset, "test")

    member_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in args.members:
        actor, bundle_path, _bundle = load_actor_bundle(name)
        print(f"[member:{name}] loaded from {bundle_path}")
        member_profile = load_profile(name)
        member_metrics[name] = _evaluate_actor(actor, test_df, member_profile)
        for slice_name, metrics in member_metrics[name].items():
            print(f"  [{slice_name}] {_format_metrics(metrics)}")

    ensemble, _bundles = build_actor_ensemble(args.members, method=args.method)
    print(f"\n[ensemble:{args.method}] members={args.members}")
    ensemble_metrics = _evaluate_actor(ensemble, test_df, profile)
    for slice_name, metrics in ensemble_metrics.items():
        print(f"  [{slice_name}] {_format_metrics(metrics)}")

    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    md_path = stem.with_suffix(".md")

    json_path.write_text(
        json.dumps(
            {
                "members": args.members,
                "anchor": args.anchor,
                "method": args.method,
                "member_metrics": member_metrics,
                "ensemble_metrics": ensemble_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _render_markdown(md_path, args.members, args.anchor, member_metrics, ensemble_metrics)
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
