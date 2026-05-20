#!/usr/bin/env python3
"""Validate the SOXX relative-weakness cap on saved walk-forward actors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_selective_release_walkforward import (
    execution_primary_blocker,
    load_actor,
    walk_forward_actor_specs,
)
from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib, segment_frame
from src.soxl_quant.policy.execution_overlays import (
    SOXX_RELATIVE_WEAKNESS_CAP_VARIANT,
    soxx_overlay_variants,
)
from src.soxl_quant.settings import QLIB_DATA_PATH, load_profile


def soxx_gate_eligible_mask(feature_df: pd.DataFrame, overlay_config: dict[str, Any] | None) -> pd.Series:
    if not overlay_config:
        return pd.Series(False, index=feature_df.index)
    regime_risk = feature_df["regime_risk"].astype(float)
    min_regime = float(overlay_config.get("min_regime_risk", float("-inf")))
    max_regime = float(overlay_config.get("max_regime_risk", float("inf")))
    soxx_weak = feature_df["SOXX_BIAS_10"].astype(float) <= float(
        overlay_config.get("soxx_bias_max", -0.02)
    )
    ratio_weak = feature_df["PRICE_RATIO_MOM"].astype(float) <= float(
        overlay_config.get("price_ratio_mom_max", 0.0)
    )
    no_reversion_cushion = feature_df["SIGNED_TE_REVERSION_10"].astype(float) < float(
        overlay_config.get("signed_te_reversion_max", 0.50)
    )
    return (
        (regime_risk >= min_regime)
        & (regime_risk <= max_regime)
        & soxx_weak
        & ratio_weak
        & no_reversion_cushion
    )


def soxx_eligible_decision_trace_rows(
    decision_trace: list[dict[str, Any]],
    feature_df: pd.DataFrame,
    overlay_config: dict[str, Any] | None,
    *,
    actor_window: str,
    variant: str,
) -> list[dict[str, Any]]:
    if not overlay_config:
        return []
    eligible_mask = soxx_gate_eligible_mask(feature_df, overlay_config)
    eligible_dates = {str(pd.Timestamp(date).date()) for date in feature_df.index[eligible_mask]}
    rows: list[dict[str, Any]] = []
    for item in decision_trace:
        date = str(pd.Timestamp(item["date"]).date())
        if date not in eligible_dates:
            continue
        row = {
            "actor_window": actor_window,
            "variant": variant,
            "date": date,
        }
        row.update({key: value for key, value in item.items() if key != "date"})
        row["cap_changed"] = (
            float(row.get("target_after_overlay", 0.0))
            < float(row.get("target_before_overlay", 0.0)) - 1e-9
        )
        row["primary_blocker"] = execution_primary_blocker(row)
        rows.append(row)
    return rows


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


def _cap_changed_days(trace_rows: list[dict[str, Any]]) -> int:
    return int(sum(bool(row.get("cap_changed", False)) for row in trace_rows))


def _eligible_order_days(trace_rows: list[dict[str, Any]]) -> int:
    return int(sum(int(row.get("order_count", 0)) > 0 for row in trace_rows))


def run_walk_forward_grid(profile_name: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(profile_name)
    actor_feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    overlay_packs = tuple(dict.fromkeys((*profile.feature.packs, "soxx_signals", "transition_signals")))
    dataset = build_dataset(feature_packs=overlay_packs, profile_name=profile.name)
    full_df = segment_frame(dataset, "test")
    price_full = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()

    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    variants = soxx_overlay_variants()
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
                trace_decisions=bool(overlay_config),
                test_seg=(spec.start, spec.end),
            )
            if variant == "baseline":
                baseline_positions = np.asarray(result.get("position_history", []), dtype=float)
            current_trace_rows = soxx_eligible_decision_trace_rows(
                list(result.get("decision_trace", [])),
                feature_df,
                overlay_config,
                actor_window=spec.window,
                variant=variant,
            )
            trace_rows.extend(current_trace_rows)
            eligible_mask = soxx_gate_eligible_mask(feature_df, overlay_config)
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
            row["feature_eligible_days"] = int(eligible_mask.sum())
            row["eligible_days"] = len(current_trace_rows)
            row["cap_changed_days"] = _cap_changed_days(current_trace_rows)
            row["eligible_order_days"] = _eligible_order_days(current_trace_rows)
            row["target_absorbed_days"] = max(int(row["eligible_days"]) - int(row["cap_changed_days"]), 0)
            row["position_absorbed_days"] = max(int(row["eligible_days"]) - int(row["changed_days"]), 0)
            if eligible_mask.any():
                eligible_index = feature_df.index[eligible_mask]
                row["first_eligible_date"] = str(eligible_index[0].date())
                row["last_eligible_date"] = str(eligible_index[-1].date())
            else:
                row["first_eligible_date"] = ""
                row["last_eligible_date"] = ""
            rows.append(row)

    context = {
        "profile": profile.name,
        "actor_feature_count": len(actor_feature_columns),
        "dataset_feature_packs": list(overlay_packs),
        "diagnostic_variant": SOXX_RELATIVE_WEAKNESS_CAP_VARIANT,
        "windows": [
            {
                "window": spec.window,
                "start": spec.start,
                "end": spec.end,
                "actor_path": str(spec.actor_path.relative_to(ROOT)),
            }
            for spec in walk_forward_actor_specs(profile_name)
        ],
        "trace_rows": len(trace_rows),
    }
    return rows, context, trace_rows


def review_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    baseline = df[df["variant"] == "baseline"].set_index("actor_window")
    variant = df[df["variant"] == SOXX_RELATIVE_WEAKNESS_CAP_VARIANT].set_index("actor_window")
    wf_2022 = variant.loc["wf_2022"]
    base_2022 = baseline.loc["wf_2022"]
    wf_2024 = variant.loc["wf_2023_2024"]
    base_2024 = baseline.loc["wf_2023_2024"]

    checks = {
        "wf_2022_avg_pos_preserved": float(wf_2022["avg_pos"]) <= float(base_2022["avg_pos"]) + 0.005,
        "wf_2022_max_dd_preserved": float(wf_2022["max_dd"]) <= float(base_2022["max_dd"]) + 0.5,
        "wf_2023_2024_recovery_improved": float(wf_2024["return"]) > float(base_2024["return"]),
        "wf_2023_2024_drawdown_not_materially_worse": float(wf_2024["max_dd"]) <= float(base_2024["max_dd"]) + 2.5,
        "walk_forward_target_changed": int(variant["cap_changed_days"].sum()) > 0,
        "walk_forward_position_changed": int(variant["changed_days"].sum()) > 0,
    }
    decision = "candidate" if all(checks.values()) else "blocked"
    return {
        "decision": decision,
        "checks": checks,
        "wf_2022_delta_avg_pos": round(float(wf_2022["avg_pos"]) - float(base_2022["avg_pos"]), 6),
        "wf_2022_delta_max_dd": round(float(wf_2022["max_dd"]) - float(base_2022["max_dd"]), 4),
        "wf_2023_2024_delta_return": round(float(wf_2024["return"]) - float(base_2024["return"]), 4),
        "wf_2023_2024_delta_max_dd": round(float(wf_2024["max_dd"]) - float(base_2024["max_dd"]), 4),
        "total_cap_changed_days": int(variant["cap_changed_days"].sum()),
        "total_changed_days": int(variant["changed_days"].sum()),
    }


def write_outputs(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    output_dir: Path,
    trace_rows: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = trace_rows or []
    df = pd.DataFrame(rows)
    trace_df = pd.DataFrame(trace_rows)
    decision = review_decision(rows)
    df.to_csv(output_dir / "soxx_relative_walkforward_results.csv", index=False)
    trace_df.to_csv(output_dir / "soxx_relative_walkforward_execution_trace.csv", index=False)
    (output_dir / "soxx_relative_walkforward_results.json").write_text(
        json.dumps({"context": context, "decision": decision, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "soxx_relative_walkforward_execution_trace.json").write_text(
        json.dumps({"context": context, "rows": trace_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline = df[df["variant"] == "baseline"].set_index("actor_window")
    lines = [
        "# SOXX Relative Weakness Walk-Forward Review",
        "",
        f"- Profile: `{context['profile']}`",
        f"- Dataset feature packs for overlay columns: `{context['dataset_feature_packs']}`",
        f"- Diagnostic variant: `{context['diagnostic_variant']}`",
        f"- Decision: `{decision['decision']}`",
        "- Role: saved walk-forward actor absorption validation before runtime overlay promotion",
        "",
        "## Results",
        "",
        "| actor_window | variant | return | sharpe | max_dd | avg_pos | trades | feature_eligible_days | eligible_days | cap_changed_days | changed_days | eligible_order_days | target_absorbed_days | position_absorbed_days | delta_return | delta_max_dd |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        base = baseline.loc[row["actor_window"]]
        lines.append(
            "| {actor_window} | {variant} | {return:.2f}% | {sharpe:.3f} | {max_dd:.2f}% | "
            "{avg_pos:.4f} | {trades} | {feature_eligible_days} | {eligible_days} | "
            "{cap_changed_days} | {changed_days} | {eligible_order_days} | {target_absorbed_days} | "
            "{position_absorbed_days} | "
            "{delta_return:+.2f}pp | {delta_max_dd:+.2f}pp |".format(
                **row,
                delta_return=float(row["return"]) - float(base["return"]),
                delta_max_dd=float(row["max_dd"]) - float(base["max_dd"]),
            )
        )

    lines.extend(["", "## Decision Checks", "", "| check | passed |", "|---|---:|"])
    for check, passed in decision["checks"].items():
        lines.append(f"| `{check}` | `{str(bool(passed)).lower()}` |")

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- `wf_2022` delta avg_pos: `{decision['wf_2022_delta_avg_pos']:+.6f}`.",
            f"- `wf_2022` delta max_dd: `{decision['wf_2022_delta_max_dd']:+.4f}pp`.",
            f"- `wf_2023_2024` delta return: `{decision['wf_2023_2024_delta_return']:+.4f}pp`.",
            f"- `wf_2023_2024` delta max_dd: `{decision['wf_2023_2024_delta_max_dd']:+.4f}pp`.",
            f"- Total cap-changed days: `{decision['total_cap_changed_days']}`; total position changed days: `{decision['total_changed_days']}`.",
            "- This review validates saved walk-forward actor behavior only; it is not runtime binding or auto-select approval.",
        ]
    )
    if not trace_df.empty:
        lines.extend(["", "## Eligible-Day Trace Summary", ""])
        for actor_window in sorted(trace_df["actor_window"].unique()):
            subset = trace_df[trace_df["actor_window"] == actor_window]
            primary = subset["primary_blocker"].value_counts().to_dict()
            primary_text = ", ".join(f"`{name}` {int(count)}" for name, count in sorted(primary.items()))
            cap_changed = int(subset["cap_changed"].astype(bool).sum())
            ordered = int((subset["order_count"].astype(int) > 0).sum())
            lines.append(
                f"- `{actor_window}`: traced `{len(subset)}` eligible days; cap_changed `{cap_changed}`; ordered `{ordered}`; primary {primary_text}."
            )
        lines.append("- Full eligible-day trace: `soxx_relative_walkforward_execution_trace.csv`.")

    (output_dir / "soxx_relative_walkforward_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SOXX relative weakness cap on walk-forward actors")
    parser.add_argument("--profile", default="balanced_rule_residual_v22")
    parser.add_argument(
        "--output-dir",
        default="runtime/execution/card/soxx_relative_weakness_walkforward_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, context, trace_rows = run_walk_forward_grid(args.profile)
    output_dir = ROOT / args.output_dir
    write_outputs(rows, context, output_dir, trace_rows)
    print(
        f"[soxx_relative_walkforward] wrote {output_dir.relative_to(ROOT)} "
        f"rows={len(rows)} trace_rows={len(trace_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
