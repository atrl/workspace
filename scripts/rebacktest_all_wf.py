#!/usr/bin/env python3
"""Re-run backtest for all walk-forward actors with corrected test_seg."""
import sys
sys.path.insert(0, "src")

import json
import torch
from pathlib import Path
import pandas as pd

from soxl_quant.config import load_profile
from soxl_quant.data import build_dataset, segment_frame, init_qlib
from soxl_quant.policy.policy_builder import build_policy
from soxl_quant.backtest import backtest_profile
from soxl_quant.policy.observation import observation_dim_for_profile, FEATURE_COLUMNS
from soxl_quant.policy.policy_builder import feature_columns_for_packs
from soxl_quant.settings import QLIB_DATA_PATH

def rebacktest_actor(profile_name, actor_path, test_start, test_end, window_name):
    profile = load_profile(profile_name)
    bundle = torch.load(actor_path, map_location="cpu")

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(feature_packs=tuple(profile.feature.packs), profile_name=profile.name)
    full_df = pd.concat([
        segment_frame(dataset, "train"),
        segment_frame(dataset, "valid"),
        segment_frame(dataset, "test"),
    ]).sort_index()

    test_df = full_df.loc[test_start:test_end]
    price_series = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()
    price_series = price_series.loc[test_start:test_end]

    obs_dim = observation_dim_for_profile(profile)
    policy, actor = build_policy(obs_dim, profile)
    actor.load_state_dict(bundle["state_dict"])
    actor.eval()

    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    test_seg = (str(test_df.index.min().date()), str(test_df.index.max().date()))

    bt = backtest_profile(
        actor, test_df, price_series,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        asset_idx=0, feature_columns=feature_columns,
        model_config=profile.model, mode_router_config=profile.mode_router,
        test_seg=test_seg,
    )
    return {
        "window": window_name,
        "return": bt["return"], "sharpe": bt["sharpe"],
        "max_dd": bt["max_dd"], "trades": bt["trades"],
    }

actors = [
    ("balanced", "runtime/model/card/balanced_wf_2020/soxl_balanced_wf_2020_actor.pt", "2020-01-01", "2020-12-31", "balanced_wf_2020"),
    ("balanced", "runtime/model/card/balanced_wf_2022/soxl_balanced_wf_2022_actor.pt", "2022-01-01", "2022-12-31", "balanced_wf_2022"),
    ("balanced_rule_residual_v19", "runtime/model/card/balanced_rule_residual_v19_wf_2020/soxl_balanced_rule_residual_v19_wf_2020_actor.pt", "2020-01-01", "2020-12-31", "v19_wf_2020"),
    ("balanced_rule_residual_v19", "runtime/model/card/balanced_rule_residual_v19_wf_2022/soxl_balanced_rule_residual_v19_wf_2022_actor.pt", "2022-01-01", "2022-12-31", "v19_wf_2022"),
    ("balanced_rule_residual_v19_reg", "runtime/model/card/balanced_rule_residual_v19_reg_wf_2020/soxl_balanced_rule_residual_v19_reg_wf_2020_actor.pt", "2020-01-01", "2020-12-31", "v19_reg_wf_2020"),
    ("balanced_rule_residual_v19_reg", "runtime/model/card/balanced_rule_residual_v19_reg_wf_2022/soxl_balanced_rule_residual_v19_reg_wf_2022_actor.pt", "2022-01-01", "2022-12-31", "v19_reg_wf_2022"),
    ("balanced_rule_residual_v19_reg", "runtime/model/card/balanced_rule_residual_v19_reg_wf_2023_2024/soxl_balanced_rule_residual_v19_reg_wf_2023_2024_actor.pt", "2024-01-01", "2024-12-31", "v19_reg_wf_2024"),
]

results = []
for profile_name, actor_path, test_start, test_end, window_name in actors:
    if not Path(actor_path).exists():
        print(f"SKIP {window_name}: actor not found")
        continue
    try:
        r = rebacktest_actor(profile_name, actor_path, test_start, test_end, window_name)
        print(f"{r['window']}: return={r['return']:.1f}% sharpe={r['sharpe']:.2f} maxdd={r['max_dd']:.1f}% trades={r['trades']}")
        results.append(r)
    except Exception as e:
        print(f"ERROR {window_name}: {e}")

with open("runtime/model/walk_forward_rebacktest_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to runtime/model/walk_forward_rebacktest_results.json")
