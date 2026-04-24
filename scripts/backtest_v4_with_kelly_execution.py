#!/usr/bin/env python3
"""Backtest v4 model with v8's Kelly-aware execution layer.

Hypothesis: v4 model + Kelly hard_floor should outperform vanilla v4
without needing any retraining.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import qlib
import torch

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.soxl_quant.config import load_profile
from src.soxl_quant.data import build_feature_frame, feature_columns_for_packs, init_qlib
from src.soxl_quant.policy.evaluation import evaluate_split
from src.soxl_quant.policy.positioning import action_levels_for_profile
from src.soxl_quant.runtime import load_actor_bundle


def main():
    qlib.init(provider_uri=str(ROOT / "data" / "qlib" / "soxl"))

    # Load v4 model (best baseline)
    v4_profile = load_profile("balanced_rule_residual_imitation_v4")
    actor, bundle_path, bundle = load_actor_bundle("balanced_rule_residual_imitation_v4")

    # Build test dataframes WITH kelly features
    v8_profile = load_profile("balanced_kelly_execution_v8")

    print("Loading test slices with Kelly features...")
    test_2020_df, _ = build_feature_frame(
        "2020-01-01", "2020-12-31",
        feature_packs=v8_profile.feature.packs,
        profile_name=v8_profile.name,
    )
    test_2022_df, _ = build_feature_frame(
        "2022-01-01", "2022-12-31",
        feature_packs=v8_profile.feature.packs,
        profile_name=v8_profile.name,
    )
    test_2023_2024_df, _ = build_feature_frame(
        "2023-01-01", "2024-12-31",
        feature_packs=v8_profile.feature.packs,
        profile_name=v8_profile.name,
    )

    v4_feature_columns = feature_columns_for_packs(v4_profile.feature.packs, v4_profile.name)
    v8_feature_columns = feature_columns_for_packs(v8_profile.feature.packs, v8_profile.name)
    kelly_cols = [c for c in v8_feature_columns if c not in v4_feature_columns]
    print(f"v4 features: {len(v4_feature_columns)}, v8 features: {len(v8_feature_columns)}")
    print(f"Kelly columns: {kelly_cols}")

    slices = {
        "test_2020": test_2020_df,
        "test_2022": test_2022_df,
        "test_2023_2024": test_2023_2024_df,
    }

    # --- Run 1: v4 model + Kelly execution layer ---
    print("\n=== v4 Model + Kelly Execution Layer ===\n")
    for name, df in slices.items():
        metrics = evaluate_split(
            actor,
            df,
            position_cap_config=v4_profile.position_cap,
            execution_config=v4_profile.execution,
            stress_threshold=v4_profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(v4_profile),
            observation_window=v4_profile.model.observation_window,
            hard_floor=v4_profile.model.hard_floor,
            use_hybrid_overlay=v4_profile.model.use_hybrid_overlay,
            delta_levels=v4_profile.model.delta_levels if v4_profile.model.use_hybrid_overlay else None,
            mode_router_config=v4_profile.mode_router,
            feature_columns=v4_feature_columns,  # actor sees only v4 features
        )
        print(f"[{name}] return={metrics['return']:.2f}% sharpe={metrics['sharpe']:.3f} "
              f"max_dd={metrics['max_dd']:.2f}% avg_pos={metrics.get('avg_pos', 0):.3f}")

    # --- Run 2: v4 model vanilla (no Kelly execution) ---
    # To disable Kelly execution, we need to pass a df WITHOUT kelly columns
    # or ensure evaluation.py doesn't see them.
    # evaluation.py checks: if "kelly_position_21" in row.index
    # So if we strip kelly cols from df, Kelly logic won't trigger.
    print("\n=== v4 Model Vanilla (no Kelly execution) ===\n")
    for name, df in slices.items():
        vanilla_df = df[v4_feature_columns + ["close"]] if "close" in df.columns else df[v4_feature_columns]
        metrics = evaluate_split(
            actor,
            vanilla_df,
            position_cap_config=v4_profile.position_cap,
            execution_config=v4_profile.execution,
            stress_threshold=v4_profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(v4_profile),
            observation_window=v4_profile.model.observation_window,
            hard_floor=v4_profile.model.hard_floor,
            use_hybrid_overlay=v4_profile.model.use_hybrid_overlay,
            delta_levels=v4_profile.model.delta_levels if v4_profile.model.use_hybrid_overlay else None,
            mode_router_config=v4_profile.mode_router,
            feature_columns=v4_feature_columns,
        )
        print(f"[{name}] return={metrics['return']:.2f}% sharpe={metrics['sharpe']:.3f} "
              f"max_dd={metrics['max_dd']:.2f}% avg_pos={metrics.get('avg_pos', 0):.3f}")


if __name__ == "__main__":
    main()
