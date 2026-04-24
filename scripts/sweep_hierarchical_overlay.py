#!/usr/bin/env python3
"""Parameter sweep for hierarchical two-timescale overlay.

Quickly tests many combinations of budget, delta_scale, slow_freq, etc.
Uses existing actor(s) — no training required.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/sweep_hierarchical_overlay.py \
        --delta-actor artifacts/models/soxl_balanced_actor.pt \
        --base-policy rule --mode-router rule \
        --output runtime/model/reviews/hierarchical_sweep.json
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib
from src.soxl_quant.policy.environment import TrancheLedger, build_price_path
from src.soxl_quant.policy.execution import DEFAULT_EXECUTION, execution_regime, rebalance_threshold
from src.soxl_quant.policy.modes import ModeState, classify_rule_mode
from src.soxl_quant.policy.observation import make_observation
from src.soxl_quant.policy.policy_builder import actor_action
from src.soxl_quant.policy.positioning import rule_based_position, action_band_from_obs
from src.soxl_quant.settings import LOOKBACK, QLIB_DATA_PATH, load_profile
from prototype_hierarchical_two_speed import (
    HierarchicalConfig, get_base_position, get_delta, compose_position,
    evaluate_hierarchical, load_actor_bundle,
)


def run_sweep(
    delta_actor,
    df: pd.DataFrame,
    profile,
    feature_columns: List[str],
    param_grid: Dict[str, List],
    base_actor=None,
) -> List[Dict]:
    """Run grid search over hierarchical parameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        cfg = HierarchicalConfig(
            slow_freq=params.get("slow_freq", 5),
            base_policy_type=params.get("base_policy", "rule"),
            mode_router_type=params.get("mode_router", "rule"),
            trend_hold_budget=params.get("trend_budget", 1.0),
            defensive_hold_budget=params.get("defensive_budget", 0.6),
            crash_rebuild_budget=params.get("crash_budget", 0.8),
            flat_wait_budget=params.get("flat_budget", 0.3),
            delta_scale=params.get("delta_scale", 1.0),
            delta_clip=params.get("delta_clip", 0.3),
            composition=params.get("composition", "additive"),
            hard_floor=params.get("hard_floor", 0.2),
        )

        result = evaluate_hierarchical(
            delta_actor, df, cfg,
            base_actor=base_actor,
            position_cap_config=profile.position_cap,
            execution_config=profile.execution,
            observation_window=max(int(profile.model.observation_window), 1),
            feature_columns=feature_columns,
            profile=profile,
        )

        dates = pd.to_datetime(df.index)
        test_2020_mask = (dates >= "2020-01-01") & (dates <= "2020-12-31")
        test_2022_mask = (dates >= "2022-01-01") & (dates <= "2022-12-31")
        test_2023_2024_mask = (dates >= "2023-01-01") & (dates <= "2024-12-31")

        def slice_metrics(values, positions, mask):
            vals = [v for v, m in zip(values, mask) if m]
            pos = [p for p, m in zip(positions, mask) if m]
            if len(vals) < 2:
                return {}
            ret = vals[-1] / vals[0] - 1.0
            rets = pd.Series(vals).pct_change().dropna()
            sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
            rm = pd.Series(vals).cummax()
            dd = ((rm - pd.Series(vals)) / rm).max()
            return {"return": ret, "sharpe": sharpe, "max_dd": dd, "avg_pos": np.mean(pos) if pos else 0.0}

        row = {
            "params": params,
            "proxy": {
                "return": result["return"],
                "sharpe": result["sharpe"],
                "max_dd": result["max_dd"],
                "avg_pos": result["avg_pos"],
                "trades": result["trades"],
            },
            "test_2020": slice_metrics(result["values"], result["positions"], test_2020_mask),
            "test_2022": slice_metrics(result["values"], result["positions"], test_2022_mask),
            "test_2023_2024": slice_metrics(result["values"], result["positions"], test_2023_2024_mask),
        }
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-actor", type=str, required=True)
    parser.add_argument("--base-actor", type=str, default=None)
    parser.add_argument("--profile", type=str, default="balanced")
    parser.add_argument("--output", type=str, default="runtime/model/reviews/hierarchical_sweep.json")
    parser.add_argument("--skip-data-refresh", action="store_true")
    args = parser.parse_args()

    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(args.profile)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)

    delta_actor, _ = load_actor_bundle(args.delta_actor, profile)
    base_actor = None
    if args.base_actor:
        base_actor, _ = load_actor_bundle(args.base_actor, profile)

    dataset = build_dataset(feature_packs=tuple(profile.feature.packs), profile_name=profile.name)
    df = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    # Parameter grid (narrow for quick prototype)
    param_grid = {
        "slow_freq": [5, 10],
        "flat_budget": [0.15, 0.30],
        "defensive_budget": [0.50, 0.70],
        "crash_budget": [0.60, 0.90],
        "trend_budget": [1.00],
        "delta_scale": [0.3, 0.6],
        "delta_clip": [0.20, 0.30],
        "composition": ["additive"],
        "hard_floor": [0.20],
    }

    print(f"[sweep] Running {np.prod([len(v) for v in param_grid.values()])} combinations...")
    results = run_sweep(delta_actor, df, profile, feature_columns, param_grid, base_actor)

    # Sort by proxy Sharpe / (1 + MaxDD)
    def score(r):
        p = r["proxy"]
        return p["sharpe"] / (1.0 + p["max_dd"])

    results.sort(key=score, reverse=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"top_10": results[:10], "all": results}, f, indent=2, default=str)

    print(f"[sweep] Top result:")
    print(json.dumps(results[0], indent=2, default=str))
    print(f"\nWrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
