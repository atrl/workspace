#!/usr/bin/env python3
"""Validate selective release overlays on saved walk-forward actors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_selective_release_overlay import overlay_variants
from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import QLIB_DATA_PATH, load_profile


@dataclass(frozen=True)
class WalkForwardActorSpec:
    window: str
    start: str
    end: str
    actor_path: Path


WALK_FORWARD_WINDOWS = (
    ("wf_2020", "2020-01-01", "2020-12-31"),
    ("wf_2022", "2022-01-01", "2022-12-31"),
    ("wf_2023_2024", "2024-01-01", "2024-12-31"),
)


def walk_forward_actor_specs(profile_name: str) -> list[WalkForwardActorSpec]:
    specs: list[WalkForwardActorSpec] = []
    for window, start, end in WALK_FORWARD_WINDOWS:
        actor_path = (
            ROOT
            / "runtime"
            / "model"
            / "card"
            / f"{profile_name}_{window}"
            / f"soxl_{profile_name}_{window}_actor.pt"
        )
        specs.append(WalkForwardActorSpec(window=window, start=start, end=end, actor_path=actor_path))
    return specs


def variant_subset(retained_only: bool) -> list[tuple[str, dict[str, Any] | None]]:
    variants = overlay_variants()
    if not retained_only:
        return variants
    return [
        (name, config)
        for name, config in variants
        if name in {"baseline", "selective_release_recovery_v1"}
    ]


def gate_eligible_mask(feature_df: pd.DataFrame, overlay_config: dict[str, Any] | None) -> pd.Series:
    if not overlay_config:
        return pd.Series(False, index=feature_df.index)
    regime_risk = feature_df["regime_risk"].astype(float)
    min_regime = float(overlay_config.get("min_regime_risk", float("-inf")))
    max_regime = float(overlay_config.get("max_regime_risk", float("inf")))
    cooldown_ready = feature_df["RB_COOLDOWN_PROXY"].astype(float) >= float(
        overlay_config.get("cooldown_threshold", 0.25)
    )
    release_ready = (
        feature_df["TRANS_VIX_DIVERGENCE"].astype(float) >= float(overlay_config.get("vix_threshold", 0.04))
    ) | (
        feature_df["CICC_RETURN_VOLUME_SYNC"].astype(float)
        >= float(overlay_config.get("volume_sync_threshold", 0.68))
    )
    return (regime_risk >= min_regime) & (regime_risk <= max_regime) & cooldown_ready & release_ready


def metrics_from_result(
    result: dict[str, Any],
    baseline_positions: np.ndarray | None = None,
) -> dict[str, float | int]:
    positions = np.asarray(result.get("position_history", []), dtype=float)
    metrics: dict[str, float | int] = {
        "return": round(float(result["return"]), 4),
        "sharpe": round(float(result["sharpe"]), 4),
        "max_dd": round(float(result["max_dd"]), 4),
        "avg_pos": round(float(positions.mean()) if positions.size else 0.0, 6),
        "trades": int(result["trades"]),
        "changed_days": 0,
        "avg_pos_delta": 0.0,
        "max_pos_delta": 0.0,
    }
    if baseline_positions is not None and positions.size:
        n = min(len(baseline_positions), len(positions))
        delta = positions[:n] - baseline_positions[:n]
        metrics["changed_days"] = int(np.sum(np.abs(delta) > 1e-6))
        metrics["avg_pos_delta"] = round(float(delta.mean()), 6)
        metrics["max_pos_delta"] = round(float(delta.max()) if delta.size else 0.0, 6)
    return metrics


def load_actor(actor_path: Path, profile) -> Any:
    if not actor_path.exists():
        raise FileNotFoundError(f"walk-forward actor not found: {actor_path}")
    ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor


def run_walk_forward_grid(
    profile_name: str,
    retained_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(profile_name)
    actor_feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    overlay_packs = tuple(dict.fromkeys((*profile.feature.packs, "release_signals_v1")))
    dataset = build_dataset(feature_packs=overlay_packs, profile_name=profile.name)
    full_df = segment_frame(dataset, "test")
    price_full = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()

    rows: list[dict[str, Any]] = []
    variants = variant_subset(retained_only)
    for spec in walk_forward_actor_specs(profile_name):
        actor = load_actor(spec.actor_path, profile)
        feature_df = full_df.loc[spec.start : spec.end]
        price = price_full.loc[feature_df.index]
        baseline_positions: np.ndarray | None = None

        for variant, overlay_config in variants:
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
                test_seg=(spec.start, spec.end),
            )
            if variant == "baseline":
                baseline_positions = np.asarray(result.get("position_history", []), dtype=float)
            row = {
                "profile": profile.name,
                "actor_window": spec.window,
                "start": spec.start,
                "end": spec.end,
                "actor_path": str(spec.actor_path.relative_to(ROOT)),
                "variant": variant,
                "overlay_config": json.dumps(overlay_config or {}, sort_keys=True),
            }
            row.update(metrics_from_result(result, None if variant == "baseline" else baseline_positions))
            eligible_mask = gate_eligible_mask(feature_df, overlay_config)
            row["eligible_days"] = int(eligible_mask.sum())
            if eligible_mask.any():
                eligible_index = feature_df.index[eligible_mask]
                row["first_eligible_date"] = str(eligible_index[0].date())
                row["last_eligible_date"] = str(eligible_index[-1].date())
            else:
                row["first_eligible_date"] = ""
                row["last_eligible_date"] = ""
            row["absorbed_days"] = max(int(row["eligible_days"]) - int(row["changed_days"]), 0)
            rows.append(row)

    context = {
        "profile": profile.name,
        "actor_feature_count": len(actor_feature_columns),
        "dataset_feature_packs": list(overlay_packs),
        "retained_only": retained_only,
        "windows": [
            {"window": spec.window, "start": spec.start, "end": spec.end, "actor_path": str(spec.actor_path.relative_to(ROOT))}
            for spec in walk_forward_actor_specs(profile_name)
        ],
    }
    return rows, context


def write_outputs(rows: list[dict[str, Any]], context: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "selective_release_walkforward_results.csv", index=False)
    (output_dir / "selective_release_walkforward_results.json").write_text(
        json.dumps({"context": context, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline = df[df["variant"] == "baseline"].set_index("actor_window")
    lines = [
        "# Selective Release Walk-Forward Actor Review",
        "",
        f"- Profile: `{context['profile']}`",
        f"- Dataset feature packs for overlay columns: `{context['dataset_feature_packs']}`",
        "- Role: walk-forward actor validation before runtime overlay promotion",
        "",
        "## Results",
        "",
        "| actor_window | variant | return | sharpe | max_dd | avg_pos | trades | eligible_days | changed_days | absorbed_days | delta_return | delta_max_dd |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        base = baseline.loc[row["actor_window"]]
        lines.append(
            "| {actor_window} | {variant} | {return:.2f}% | {sharpe:.3f} | {max_dd:.2f}% | "
            "{avg_pos:.4f} | {trades} | {eligible_days} | {changed_days} | {absorbed_days} | "
            "{delta_return:+.2f}pp | {delta_max_dd:+.2f}pp |".format(
                **row,
                delta_return=float(row["return"]) - float(base["return"]),
                delta_max_dd=float(row["max_dd"]) - float(base["max_dd"]),
            )
        )

    retained = df[df["variant"] == "selective_release_recovery_v1"].set_index("actor_window")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This keeps the actor fixed within each walk-forward checkpoint and changes only the execution overlay.",
            "- A promotable overlay should improve the recovery window without reopening `wf_2022` drawdown or average exposure.",
        ]
    )
    if not retained.empty:
        wf2022 = retained.loc["wf_2022"] if "wf_2022" in retained.index else None
        wf2024 = retained.loc["wf_2023_2024"] if "wf_2023_2024" in retained.index else None
        if wf2022 is not None and wf2024 is not None:
            base_2022 = baseline.loc["wf_2022"]
            base_2024 = baseline.loc["wf_2023_2024"]
            lines.extend(
                [
                    "",
                    "## Retained Gate Check",
                    "",
                    f"- `wf_2022`: return `{float(wf2022['return']):.2f}%` vs baseline `{float(base_2022['return']):.2f}%`; max_dd `{float(wf2022['max_dd']):.2f}%` vs `{float(base_2022['max_dd']):.2f}%`; eligible_days `{int(wf2022['eligible_days'])}`; changed_days `{int(wf2022['changed_days'])}`.",
                    f"- `wf_2023_2024`: return `{float(wf2024['return']):.2f}%` vs baseline `{float(base_2024['return']):.2f}%`; max_dd `{float(wf2024['max_dd']):.2f}%` vs `{float(base_2024['max_dd']):.2f}%`; eligible_days `{int(wf2024['eligible_days'])}`; changed_days `{int(wf2024['changed_days'])}`.",
                ]
            )

    (output_dir / "selective_release_walkforward_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selective release overlays on walk-forward actors")
    parser.add_argument("--profile", default="balanced_rule_residual_v22")
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="evaluate all existing selective release variants instead of baseline plus retained recovery gate",
    )
    parser.add_argument(
        "--output-dir",
        default="runtime/execution/card/selective_release_walkforward_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, context = run_walk_forward_grid(args.profile, retained_only=not args.all_variants)
    output_dir = ROOT / args.output_dir
    write_outputs(rows, context, output_dir)
    print(f"[selective_release_walkforward] wrote {output_dir.relative_to(ROOT)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
