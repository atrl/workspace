#!/usr/bin/env python3
"""Evaluate low-churn selective release overlays on a fixed profile actor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import QLIB_DATA_PATH, load_profile, model_artifact_path


WINDOWS = {
    "test_proxy": ("2020-01-01", "2026-12-31"),
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


def overlay_variants() -> list[tuple[str, dict[str, Any] | None]]:
    return [
        ("baseline", None),
        (
            "selective_release_cicc_v1",
            {
                "type": "selective_release",
                "boost": 0.05,
                "max_target": 0.62,
                "max_regime_risk": 0.62,
                "cooldown_threshold": 0.25,
                "vix_threshold": 0.04,
                "volume_sync_threshold": 0.68,
            },
        ),
        (
            "selective_release_tight_v1",
            {
                "type": "selective_release",
                "boost": 0.05,
                "max_target": 0.60,
                "max_regime_risk": 0.58,
                "cooldown_threshold": 0.30,
                "vix_threshold": 0.05,
                "volume_sync_threshold": 0.72,
            },
        ),
        (
            "selective_release_recovery_v1",
            {
                "type": "selective_release",
                "boost": 0.08,
                "max_target": 0.65,
                "max_regime_risk": 0.60,
                "cooldown_threshold": 0.20,
                "vix_threshold": 0.05,
                "volume_sync_threshold": 0.70,
            },
        ),
        (
            "selective_release_recovery_threshold_bypass_v1",
            {
                "type": "selective_release",
                "boost": 0.08,
                "max_target": 0.65,
                "max_regime_risk": 0.60,
                "cooldown_threshold": 0.20,
                "vix_threshold": 0.05,
                "volume_sync_threshold": 0.70,
                "rebalance_threshold_override": 0.0,
            },
        ),
        (
            "selective_release_recovery_target_lift_v1",
            {
                "type": "selective_release",
                "boost": 0.08,
                "max_target": 0.65,
                "max_regime_risk": 0.60,
                "cooldown_threshold": 0.20,
                "vix_threshold": 0.05,
                "volume_sync_threshold": 0.70,
                "min_target_delta": 0.08,
            },
        ),
        (
            "selective_release_recovery_trade_lock_bypass_v1",
            {
                "type": "selective_release",
                "boost": 0.08,
                "max_target": 0.65,
                "max_regime_risk": 0.60,
                "cooldown_threshold": 0.20,
                "vix_threshold": 0.05,
                "volume_sync_threshold": 0.70,
                "rebalance_threshold_override": 0.0,
                "trade_lock_override": 0,
            },
        ),
        (
            "selective_release_recovery_trade_lock_short_v1",
            {
                "type": "selective_release",
                "boost": 0.08,
                "max_target": 0.65,
                "max_regime_risk": 0.60,
                "cooldown_threshold": 0.20,
                "vix_threshold": 0.05,
                "volume_sync_threshold": 0.70,
                "rebalance_threshold_override": 0.0,
                "trade_lock_cap": 1,
            },
        ),
    ]


def metrics_from_result(result: dict[str, Any]) -> dict[str, float]:
    positions = np.asarray(result.get("position_history", []), dtype=float)
    return {
        "return": round(float(result["return"]), 4),
        "sharpe": round(float(result["sharpe"]), 4),
        "max_dd": round(float(result["max_dd"]), 4),
        "avg_pos": round(float(positions.mean()) if positions.size else 0.0, 6),
        "trades": int(result["trades"]),
    }


def run_backtest_grid(profile_name: str, tag: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(profile_name)
    actor_feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    overlay_packs = tuple(dict.fromkeys((*profile.feature.packs, "release_signals_v1")))
    dataset = build_dataset(feature_packs=overlay_packs, profile_name=profile.name)
    full_df = segment_frame(dataset, "test")
    price_full = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()

    model_path = model_artifact_path(profile.name, tag)
    if not model_path.exists():
        raise FileNotFoundError(f"actor checkpoint not found: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    rows: list[dict[str, Any]] = []
    for variant, overlay_config in overlay_variants():
        for window, (start, end) in WINDOWS.items():
            feature_df = full_df.loc[start:end]
            price = price_full.loc[feature_df.index]
            result = backtest_profile(
                actor=actor,
                feature_df=feature_df,
                price_series=price,
                position_cap_config=profile.position_cap,
                position_budget_config=profile.position_budget,
                execution_config=profile.execution,
                mode_router_config=profile.mode_router,
                asset_idx=0,
                feature_columns=actor_feature_columns,
                model_config=profile.model,
                overlay_config=overlay_config,
                test_seg=(start, end),
            )
            row = {
                "variant": variant,
                "window": window,
                "overlay_config": json.dumps(overlay_config or {}, sort_keys=True),
            }
            row.update(metrics_from_result(result))
            rows.append(row)
    context = {
        "profile": profile.name,
        "tag": tag,
        "model_path": str(model_path.relative_to(ROOT)),
        "actor_feature_count": len(actor_feature_columns),
        "dataset_feature_packs": list(overlay_packs),
    }
    return rows, context


def write_outputs(rows: list[dict[str, Any]], context: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "selective_release_overlay_results.csv", index=False)
    (output_dir / "selective_release_overlay_results.json").write_text(
        json.dumps({"context": context, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline = df[df["variant"] == "baseline"].set_index("window")
    lines = [
        "# Selective Release Overlay Review",
        "",
        f"- Profile: `{context['profile']}`",
        f"- Actor: `{context['model_path']}`",
        f"- Dataset feature packs for overlay columns: `{context['dataset_feature_packs']}`",
        "- Role: fixed-actor counterfactual before any retrain",
        "",
        "## Results",
        "",
        "| variant | window | return | sharpe | max_dd | avg_pos | trades | delta_return | delta_max_dd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        base = baseline.loc[row["window"]]
        lines.append(
            "| {variant} | {window} | {return:.2f}% | {sharpe:.3f} | {max_dd:.2f}% | "
            "{avg_pos:.4f} | {trades} | {delta_return:+.2f}pp | {delta_max_dd:+.2f}pp |".format(
                **row,
                delta_return=float(row["return"]) - float(base["return"]),
                delta_max_dd=float(row["max_dd"]) - float(base["max_dd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This is a fixed-actor overlay separation test. It does not create a new PPO profile.",
            "- A promotion-worthy release overlay should improve `test_2023_2024` without widening `test_2022` drawdown or lifting `test_2022` average exposure materially.",
            "- If no row clears that trade-off, the release trio remains useful as diagnostic evidence but is not ready for runtime routing.",
        ]
    )
    (output_dir / "selective_release_overlay_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selective release overlays")
    parser.add_argument("--profile", default="balanced_rule_residual_v22")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--output-dir",
        default="runtime/execution/card/selective_release_overlay_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, context = run_backtest_grid(args.profile, args.tag)
    output_dir = ROOT / args.output_dir
    write_outputs(rows, context, output_dir)
    print(f"[selective_release_overlay] wrote {output_dir.relative_to(ROOT)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
