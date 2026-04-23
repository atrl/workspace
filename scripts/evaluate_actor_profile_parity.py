#!/usr/bin/env python3
"""Evaluate a saved actor bundle under a possibly different profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.dashboard import write_review_daily_csv
from src.soxl_quant.app.evaluate import slice_metrics
from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib, segment_frame
from src.soxl_quant.policy.evaluation import evaluate_split
from src.soxl_quant.policy.positioning import action_levels_for_profile
from src.soxl_quant.runtime import load_actor_bundle
from src.soxl_quant.settings import QLIB_DATA_PATH, TEST_SEG, load_profile


SLICES = {
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one actor bundle under one profile's runtime semantics.")
    parser.add_argument("--profile", required=True, help="Profile used for feature schema and action interpretation")
    parser.add_argument("--model", required=True, help="Path to saved actor bundle")
    parser.add_argument("--output-json", required=True, help="JSON metrics output path")
    parser.add_argument("--output-md", required=True, help="Markdown review output path")
    parser.add_argument("--output-daily", required=True, help="Daily replay CSV output path")
    return parser.parse_args()


def fmt(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['return']:.1f}% / {metrics['sharpe']:.3f} / "
        f"{metrics['max_dd']:.1f}% / pos {metrics.get('avg_pos', 0.0):.3f} / "
        f"trades {int(metrics.get('trades', 0))}"
    )


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    model_path = Path(args.model)

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    actor, bundle_path, bundle = load_actor_bundle(profile.name, model_path=model_path)

    slices = slice_metrics(actor, test_df, SLICES, profile)
    test_proxy = evaluate_split(
        actor,
        test_df,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        execution_config=profile.execution,
        asset_idx=0,
        action_levels=action_levels_for_profile(profile),
        observation_window=profile.model.observation_window,
        hard_floor=profile.model.hard_floor,
        use_hybrid_overlay=profile.model.use_hybrid_overlay,
        delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
        feature_columns=feature_columns,
        mode_router_config=profile.mode_router,
        position_mapping=profile.model.position_mapping,
    )
    bt = backtest_profile(
        actor=actor,
        feature_df=test_df,
        price_series=price,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        execution_config=profile.execution,
        mode_router_config=profile.mode_router,
        symbol="SOXL",
        asset_idx=0,
        feature_columns=feature_columns,
        test_seg=TEST_SEG,
        model_config=profile.model,
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_daily = Path(args.output_daily)
    for path in (output_json, output_md, output_daily):
        path.parent.mkdir(parents=True, exist_ok=True)

    write_review_daily_csv(
        output_daily,
        bt["portfolio"],
        price,
        bt["position_history"],
        bt.get("trade_signals", []),
    )

    payload = {
        "profile": profile.name,
        "model": str(bundle_path),
        "model_profile": bundle.get("profile"),
        "model_seed": bundle.get("seed"),
        "model_position_mapping": getattr(profile.model, "position_mapping", "default"),
        "slices": slices,
        "test_proxy": test_proxy,
        "qlib_backtest": {key: bt[key] for key in ("return", "sharpe", "max_dd", "trades")},
        "daily_csv": str(output_daily),
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    rows = [
        f"# {profile.name} Actor-Backed Parity",
        "",
        f"- model: `{bundle_path}`",
        f"- model profile: `{bundle.get('profile')}`",
        f"- position mapping: `{getattr(profile.model, 'position_mapping', 'default')}`",
        "",
        "| window | return / sharpe / maxdd / avg_pos / trades |",
        "| --- | --- |",
    ]
    for name in ("test_2020", "test_2022", "test_2023_2024"):
        rows.append(f"| {name} | {fmt(slices[name])} |")
    rows.extend(
        [
            f"| test_proxy | {fmt(test_proxy)} |",
            f"| qlib_backtest | {bt['return']:.1f}% / {bt['sharpe']:.3f} / {bt['max_dd']:.1f}% / trades {int(bt['trades'])} |",
            "",
            "## Decision",
            "",
            "pending",
            "",
        ]
    )
    output_md.write_text("\n".join(rows), encoding="utf-8")

    print(f"[actor-profile-parity] profile={profile.name} model={bundle_path.name}")
    for name in ("test_2020", "test_2022", "test_2023_2024"):
        print(f"  - {name}: {fmt(slices[name])}")
    print(f"  - test_proxy: {fmt(test_proxy)}")
    print(f"  - qlib_backtest: {bt['return']:.1f}% / {bt['sharpe']:.3f} / {bt['max_dd']:.1f}% / trades {int(bt['trades'])}")
    print(f"[done] wrote {output_json}, {output_md}, {output_daily}")


if __name__ == "__main__":
    main()
