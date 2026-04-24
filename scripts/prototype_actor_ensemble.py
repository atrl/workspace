#!/usr/bin/env python3
"""Dynamic ensemble: switch between v19 (aggressive) and mode_v1 (defensive) actors based on regime."""
import sys
sys.path.insert(0, "/Users/rziiiii/.openclaw/workspace")

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib
from src.soxl_quant.policy.environment import TrancheLedger, build_price_path
from src.soxl_quant.policy.execution import (
    DEFAULT_EXECUTION,
    execution_buy_cooldown_days,
    execution_regime,
    execution_trade_lock_days,
    rebalance_threshold,
)
from src.soxl_quant.policy.modes import ModeState, mode_band, route_mode
from src.soxl_quant.policy.observation import make_observation
from src.soxl_quant.policy.policy_builder import actor_action, build_policy
from src.soxl_quant.policy.positioning import target_position_for_profile
from src.soxl_quant.settings import LOOKBACK, QLIB_DATA_PATH, load_profile


def evaluate_ensemble(
    actor_aggressive, actor_defensive,
    df, profile,
    regime_threshold=0.55,
    switch_cooldown=5,
):
    cash = 100_000.0
    peak = cash
    pos = 0.0
    ledger = TrancheLedger()
    exec_cfg = profile.execution or DEFAULT_EXECUTION
    buy_cooldown = 0
    trade_lock = 0
    price_path = build_price_path(df)
    values = [cash]
    positions = []
    stress_positions = []
    mode_state = ModeState()

    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    current_actor = actor_aggressive
    actor_label = "aggressive"
    cooldown = 0
    switches = 0

    for idx in range(LOOKBACK, len(df) - 1):
        obs = make_observation(df, idx, pos, cash, peak, ledger, price_path,
                               observation_window=profile.model.observation_window,
                               feature_columns=feature_columns)
        mode_state = route_mode(obs, mode_state, profile.mode_router) if profile.mode_router.enabled else ModeState()
        current_mode = mode_state.name

        # Regime-based switching with cooldown
        regime_risk = float(np.nan_to_num(df.iloc[idx]["regime_risk"]))
        if cooldown <= 0:
            if regime_risk >= regime_threshold and actor_label != "defensive":
                current_actor = actor_defensive
                actor_label = "defensive"
                cooldown = switch_cooldown
                switches += 1
            elif regime_risk < regime_threshold - 0.10 and actor_label != "aggressive":
                current_actor = actor_aggressive
                actor_label = "aggressive"
                cooldown = switch_cooldown
                switches += 1
        else:
            cooldown -= 1

        action = actor_action(current_actor, obs)
        target_pos = target_position_for_profile(
            action, obs, config=profile.position_cap, action_levels=np.asarray(profile.model.action_levels, dtype=np.float32),
            hard_floor=profile.model.hard_floor,
            mode_band=mode_band(current_mode, profile.mode_router) if current_mode else None,
        )

        row = df.iloc[idx]
        realized_vol = float(np.nan_to_num(row["realized_vol_10"]))
        trend_strength = abs(float(np.nan_to_num(row["trend_gap_21"])))
        cum_drop_30 = float(np.nan_to_num(row["cum_drop_30"]))
        relative_strength = float(np.nan_to_num(row["relative_strength_21"]))
        vix_zscore = float(np.nan_to_num(row["vix_zscore"]))
        downside_ratio_10 = float(np.nan_to_num(row["downside_ratio_10"]))

        regime = execution_regime(regime_risk, realized_vol, trend_strength, cum_drop_30=cum_drop_30,
            relative_strength=relative_strength, vix_zscore=vix_zscore, downside_ratio_10=downside_ratio_10, config=exec_cfg)
        direction = "buy" if target_pos > pos else ("sell" if target_pos < pos else "neutral")
        min_rebalance = rebalance_threshold(regime_risk, realized_vol, trend_strength, cum_drop_30=cum_drop_30,
            relative_strength=relative_strength, vix_zscore=vix_zscore, downside_ratio_10=downside_ratio_10,
            direction=direction, mode=current_mode, config=exec_cfg, mode_router_config=profile.mode_router)
        if trade_lock > 0:
            if abs(target_pos - pos) >= min_rebalance: target_pos = pos
            trade_lock -= 1
        if buy_cooldown > 0:
            if target_pos > pos: target_pos = pos
            buy_cooldown -= 1
        if abs(target_pos - pos) < min_rebalance: target_pos = pos

        next_ret = float(df.iloc[idx + 1]["return"])
        turnover = abs(target_pos - pos)
        cash *= 1.0 + target_pos * next_ret - turnover * 0.001
        peak = max(peak, cash)
        ledger.update(pos, target_pos, float(price_path[idx]))
        if target_pos > pos + 1e-6:
            buy_cooldown = execution_buy_cooldown_days(cum_drop_30, exec_cfg, regime)
        if turnover > exec_cfg.min_trade_turnover:
            trade_lock = execution_trade_lock_days(exec_cfg, regime)
        pos = target_pos
        positions.append(pos)
        if regime_risk >= 0.65:
            stress_positions.append(pos)
        values.append(cash)

    account = np.array(values)
    daily_ret = np.diff(account) / account[:-1]
    peak_curve = np.maximum.accumulate(account)
    return {
        "return": (account[-1] / account[0] - 1) * 100,
        "sharpe": float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)),
        "max_dd": float(np.max((peak_curve - account) / peak_curve) * 100),
        "avg_pos": float(np.mean(positions)) if positions else 0.0,
        "stress_avg_pos": float(np.mean(stress_positions)) if stress_positions else 0.0,
        "switches": switches,
    }


def main():
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile("balanced_rule_residual_v19")

    # Load v19 actor
    bundle_v19 = torch.load("artifacts/models/soxl_balanced_rule_residual_v19_actor.pt", map_location="cpu")
    _, actor_v19 = build_policy(int(bundle_v19["obs_dim"]), profile)
    actor_v19.load_state_dict(bundle_v19["state_dict"])
    actor_v19.eval()

    # Load mode_v1 actor
    bundle_m1 = torch.load("artifacts/models/soxl_balanced_rule_residual_mode_v1_actor.pt", map_location="cpu")
    _, actor_m1 = build_policy(int(bundle_m1["obs_dim"]), profile)
    actor_m1.load_state_dict(bundle_m1["state_dict"])
    actor_m1.eval()

    dataset = build_dataset(feature_packs=tuple(profile.feature.packs), profile_name=profile.name)
    full_df = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    slices = {
        "test_proxy": ("2020-01-01", "2026-03-31"),
        "test_2020": ("2020-01-01", "2020-12-31"),
        "test_2022": ("2022-01-01", "2022-12-31"),
        "test_2023_2024": ("2023-01-01", "2024-12-31"),
    }

    configs = [
        ("v19_only", None, None),
        ("mode_v1_only", None, None),
        ("ensemble_rt0.55_cd5", 0.55, 5),
        ("ensemble_rt0.60_cd5", 0.60, 5),
        ("ensemble_rt0.65_cd5", 0.65, 5),
        ("ensemble_rt0.55_cd10", 0.55, 10),
        ("ensemble_rt0.60_cd10", 0.60, 10),
    ]

    results = {}
    for label, rt, cd in configs:
        for sname, (s, e) in slices.items():
            seg = full_df.loc[s:e].copy()
            if len(seg) <= 120: continue
            if label == "v19_only":
                from src.soxl_quant.policy.evaluation import evaluate_split
                m = evaluate_split(actor_v19, seg, position_cap_config=profile.position_cap, execution_config=profile.execution,
                    action_levels=np.asarray(profile.model.action_levels, dtype=np.float32), observation_window=profile.model.observation_window,
                    hard_floor=profile.model.hard_floor, feature_columns=feature_columns_for_packs(profile.feature.packs, profile.name),
                    mode_router_config=profile.mode_router)
            elif label == "mode_v1_only":
                from src.soxl_quant.policy.evaluation import evaluate_split
                m = evaluate_split(actor_m1, seg, position_cap_config=profile.position_cap, execution_config=profile.execution,
                    action_levels=np.asarray(profile.model.action_levels, dtype=np.float32), observation_window=profile.model.observation_window,
                    hard_floor=profile.model.hard_floor, feature_columns=feature_columns_for_packs(profile.feature.packs, profile.name),
                    mode_router_config=profile.mode_router)
            else:
                m = evaluate_ensemble(actor_v19, actor_m1, seg, profile, regime_threshold=rt, switch_cooldown=cd)
            for k, v in m.items():
                results[f"{label}_{sname}_{k}"] = v

    print("\n" + "="*120)
    print(f"{'Config':<22} {'Slice':<16} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'AvgPos':>8} {'StressPos':>10} {'Switches':>10}")
    print("="*120)
    for label, _, _ in configs:
        for sname in slices:
            r = results.get(f"{label}_{sname}_return", float("nan"))
            sh = results.get(f"{label}_{sname}_sharpe", float("nan"))
            dd = results.get(f"{label}_{sname}_max_dd", float("nan"))
            ap = results.get(f"{label}_{sname}_avg_pos", float("nan"))
            sp = results.get(f"{label}_{sname}_stress_avg_pos", float("nan"))
            sw = results.get(f"{label}_{sname}_switches", float("nan"))
            print(f"{label:<22} {sname:<16} {r:>9.1f}% {sh:>8.2f} {dd:>7.1f}% {ap:>8.3f} {sp:>10.3f} {sw:>10.0f}")
    print("="*120)

    out = Path("runtime/model/reviews/actor_ensemble_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
