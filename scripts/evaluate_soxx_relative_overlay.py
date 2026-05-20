#!/usr/bin/env python3
"""Evaluate a SOXX relative-weakness execution cap on a fixed actor."""

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
from src.soxl_quant.policy.execution_overlays import (
    SOXX_RELATIVE_WEAKNESS_CAP_VARIANT,
    soxx_overlay_variants,
)
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import QLIB_DATA_PATH, load_profile, model_artifact_path


WINDOWS = {
    "test_proxy": ("2020-01-01", "2026-12-31"),
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


def metrics_from_result(result: dict[str, Any]) -> dict[str, float | int]:
    positions = np.asarray(result.get("position_history", []), dtype=float)
    trace = list(result.get("decision_trace", []))
    eligible_days = sum(1 for row in trace if bool(row.get("overlay_gate_ready")))
    cap_changed_days = sum(
        1
        for row in trace
        if float(row.get("target_after_overlay", 0.0)) < float(row.get("target_before_overlay", 0.0)) - 1e-9
    )
    eligible_order_days = sum(
        1
        for row in trace
        if bool(row.get("overlay_gate_ready")) and int(row.get("order_count", 0)) > 0
    )
    return {
        "return": round(float(result["return"]), 4),
        "sharpe": round(float(result["sharpe"]), 4),
        "max_dd": round(float(result["max_dd"]), 4),
        "avg_pos": round(float(positions.mean()) if positions.size else 0.0, 6),
        "trades": int(result["trades"]),
        "eligible_days": int(eligible_days),
        "cap_changed_days": int(cap_changed_days),
        "eligible_order_days": int(eligible_order_days),
    }


def run_backtest_grid(profile_name: str, tag: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(profile_name)
    actor_feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    overlay_packs = tuple(dict.fromkeys((*profile.feature.packs, "soxx_signals", "transition_signals")))
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
    for variant, overlay_config in soxx_overlay_variants():
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
                trace_decisions=True,
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
        "diagnostic_variant": SOXX_RELATIVE_WEAKNESS_CAP_VARIANT,
    }
    return rows, context


def review_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    baseline = df[df["variant"] == "baseline"].set_index("window")
    variant = df[df["variant"] == SOXX_RELATIVE_WEAKNESS_CAP_VARIANT].set_index("window")
    test_2022 = variant.loc["test_2022"]
    base_2022 = baseline.loc["test_2022"]
    test_2023_2024 = variant.loc["test_2023_2024"]
    base_2023_2024 = baseline.loc["test_2023_2024"]

    bear_avg_pos_preserved = float(test_2022["avg_pos"]) <= float(base_2022["avg_pos"]) + 0.005
    bear_dd_preserved = float(test_2022["max_dd"]) <= float(base_2022["max_dd"]) + 0.5
    recovery_giveup_ok = float(test_2023_2024["return"]) >= float(base_2023_2024["return"]) - 5.0
    cap_has_effect = int(variant["cap_changed_days"].sum()) > 0

    checks = {
        "bear_avg_pos_preserved": bear_avg_pos_preserved,
        "bear_dd_preserved": bear_dd_preserved,
        "recovery_giveup_ok": recovery_giveup_ok,
        "cap_has_effect": cap_has_effect,
    }
    decision = "candidate" if all(checks.values()) else "blocked"
    return {
        "decision": decision,
        "checks": checks,
        "test_2022_delta_avg_pos": round(float(test_2022["avg_pos"]) - float(base_2022["avg_pos"]), 6),
        "test_2022_delta_max_dd": round(float(test_2022["max_dd"]) - float(base_2022["max_dd"]), 4),
        "test_2023_2024_delta_return": round(
            float(test_2023_2024["return"]) - float(base_2023_2024["return"]),
            4,
        ),
    }


def write_outputs(rows: list[dict[str, Any]], context: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    decision = review_decision(rows)
    df.to_csv(output_dir / "soxx_relative_overlay_results.csv", index=False)
    (output_dir / "soxx_relative_overlay_results.json").write_text(
        json.dumps({"context": context, "decision": decision, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline = df[df["variant"] == "baseline"].set_index("window")
    lines = [
        "# SOXX Relative Weakness Cap Review",
        "",
        f"- Profile: `{context['profile']}`",
        f"- Actor: `{context['model_path']}`",
        f"- Dataset feature packs for overlay columns: `{context['dataset_feature_packs']}`",
        f"- Diagnostic variant: `{context['diagnostic_variant']}`",
        f"- Decision: `{decision['decision']}`",
        "- Role: fixed-actor SOXX execution diagnostic before any retrain",
        "",
        "## Governance",
        "",
        "- Canonical baseline: `balanced_rule_residual_v22` fixed actor.",
        "- Direct comparison baseline: failed direct actor feature injection `balanced_rule_residual_v22_soxx`.",
        "- Changed layer: `execution`.",
        "- External reference: SOXX underlying ETF dataset plus prior `SIGNED_TE_REVERSION_10` factor card.",
        "- Acceptance: preserve or improve `test_2022` avg position / max drawdown while avoiding a large `test_2023_2024` return give-up.",
        "",
        "## Results",
        "",
        "| variant | window | return | sharpe | max_dd | avg_pos | trades | eligible_days | cap_changed_days | eligible_order_days | delta_return | delta_max_dd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        base = baseline.loc[row["window"]]
        lines.append(
            "| {variant} | {window} | {return:.2f}% | {sharpe:.3f} | {max_dd:.2f}% | "
            "{avg_pos:.4f} | {trades} | {eligible_days} | {cap_changed_days} | {eligible_order_days} | "
            "{delta_return:+.2f}pp | {delta_max_dd:+.2f}pp |".format(
                **row,
                delta_return=float(row["return"]) - float(base["return"]),
                delta_max_dd=float(row["max_dd"]) - float(base["max_dd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Decision Checks",
            "",
            "| check | passed |",
            "|---|---:|",
        ]
    )
    for check, passed in decision["checks"].items():
        lines.append(f"| `{check}` | `{str(bool(passed)).lower()}` |")

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- `test_2022` delta avg_pos: `{decision['test_2022_delta_avg_pos']:+.6f}`.",
            f"- `test_2022` delta max_dd: `{decision['test_2022_delta_max_dd']:+.4f}pp`.",
            f"- `test_2023_2024` delta return: `{decision['test_2023_2024_delta_return']:+.4f}pp`.",
            "- This diagnostic does not create a new PPO profile and is not a runtime promotion approval.",
        ]
    )
    (output_dir / "soxx_relative_overlay_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SOXX relative weakness overlay")
    parser.add_argument("--profile", default="balanced_rule_residual_v22")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--output-dir",
        default="runtime/execution/card/soxx_relative_weakness_cap_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, context = run_backtest_grid(args.profile, args.tag)
    output_dir = ROOT / args.output_dir
    write_outputs(rows, context, output_dir)
    print(f"[soxx_relative_overlay] wrote {output_dir.relative_to(ROOT)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
