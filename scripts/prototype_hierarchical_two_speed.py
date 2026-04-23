#!/usr/bin/env python3
"""Prototype Direction A: Two-Timescale Hierarchical Architecture.

This script evaluates a hierarchical composition of two policies:
- Slow upper layer: mode-based base position + risk budget (low frequency)
- Fast lower layer: actor-based delta timing (high frequency)

The key insight is to DECOUPLE risk-budget decisions from timing decisions:
- Upper layer decides WHEN and HOW MUCH risk to take (mode-based)
- Lower layer decides HOW to time within that budget (actor-based)

Usage:
    # Rule-based base + actor delta (freeze both)
    PYTHONPATH=. qlib_venv/bin/python scripts/prototype_hierarchical_two_speed.py \
        --delta-actor artifacts/models/soxl_balanced_actor.pt \
        --base-policy rule \
        --mode-router rule \
        --slow-freq 5

    # Actor base + actor delta (both freeze, upper updates every N days)
    PYTHONPATH=. qlib_venv/bin/python scripts/prototype_hierarchical_two_speed.py \
        --base-actor artifacts/models/soxl_balanced_actor.pt \
        --delta-actor artifacts/models/soxl_balanced_actor.pt \
        --slow-freq 5

    # With learned mode router
    PYTHONPATH=. qlib_venv/bin/python scripts/prototype_hierarchical_two_speed.py \
        --delta-actor artifacts/models/soxl_balanced_actor.pt \
        --mode-router learned
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.soxl_quant.policy.modes import ModeState, mode_band, route_mode, classify_rule_mode
from src.soxl_quant.policy.observation import make_observation
from src.soxl_quant.policy.policy_builder import actor_action, build_policy
from src.soxl_quant.policy.positioning import (
    target_position_for_profile,
    rule_based_position,
    action_band_from_obs,
)
from src.soxl_quant.settings import LOOKBACK, QLIB_DATA_PATH, load_profile


def load_actor_bundle(bundle_path: str, profile):
    """Load actor from bundle checkpoint (wrapper format with metadata)."""
    bundle = torch.load(bundle_path, map_location="cpu")
    obs_dim = int(bundle["obs_dim"])
    _, actor = build_policy(obs_dim, profile)
    actor.load_state_dict(bundle["state_dict"])
    actor.eval()
    return actor, bundle


@dataclass(frozen=True)
class HierarchicalConfig:
    """Configuration for two-timescale hierarchical overlay."""

    # Upper layer (slow) config
    slow_freq: int = 5  # days between base position updates
    base_policy_type: str = "rule"  # "rule", "actor", "constant"
    base_constant_pos: float = 0.50  # if base_policy_type == "constant"
    mode_router_type: str = "rule"  # "rule", "learned", "none"

    # Risk budget scaling per mode
    trend_hold_budget: float = 1.00
    defensive_hold_budget: float = 0.60
    crash_rebuild_budget: float = 0.80
    flat_wait_budget: float = 0.30

    # Lower layer (fast) config
    delta_scale: float = 1.00  # multiplier on delta actor output
    delta_clip: float = 0.30  # max absolute delta

    # Composition
    composition: str = "additive"  # "additive", "multiplicative", "gate"
    use_hard_floor: bool = True
    hard_floor: float = 0.20


def mode_to_budget(mode: str, cfg: HierarchicalConfig) -> float:
    """Map mode name to risk budget multiplier."""
    if mode == "trend_hold":
        return cfg.trend_hold_budget
    if mode == "defensive_hold":
        return cfg.defensive_hold_budget
    if mode == "crash_rebuild":
        return cfg.crash_rebuild_budget
    return cfg.flat_wait_budget


def get_base_position(
    obs: np.ndarray,
    base_policy_type: str,
    base_actor=None,
    base_constant_pos: float = 0.50,
    hard_floor: float = 0.20,
) -> float:
    """Get base position from upper (slow) layer."""
    if base_policy_type == "rule":
        return rule_based_position(obs)
    if base_policy_type == "constant":
        return base_constant_pos
    if base_policy_type == "actor" and base_actor is not None:
        action = actor_action(base_actor, obs)
        # Use simple mapping: action index -> position level
        # For hybrid overlay actors, action 0 is usually neutral/low
        levels = np.linspace(0.0, 1.0, 11)
        return float(levels[min(action, len(levels) - 1)])
    # Default: mid-point
    floor, cap = action_band_from_obs(obs, hard_floor=hard_floor)
    return (floor + cap) / 2.0


def get_delta(
    obs: np.ndarray,
    delta_actor,
    delta_scale: float = 1.00,
    delta_clip: float = 0.30,
) -> float:
    """Get delta from lower (fast) layer.
    
    Supports both hybrid overlay actors (5 delta levels) and standard
    discrete actors (11 position levels). For standard actors, we map
    the position output to a delta range.
    """
    if delta_actor is None:
        return 0.0
    action = actor_action(delta_actor, obs)
    
    # Try to infer action space size from actor
    n_actions = getattr(delta_actor, "output_dim", None)
    if n_actions is None:
        # Infer from last layer
        for p in delta_actor.parameters():
            pass
        if p.dim() >= 2:
            n_actions = p.shape[0]
        else:
            n_actions = 11
    
    if n_actions == 5:
        # Hybrid overlay: delta_levels are [-0.10, -0.05, 0.0, 0.05, 0.10]
        delta_levels = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
        clipped_action = int(np.clip(action, 0, len(delta_levels) - 1))
        raw_delta = delta_levels[clipped_action]
    else:
        # Standard discrete actor: map 0..N-1 to [-1, +1]
        # Center is (N-1)/2, map to 0; ends map to +/-1
        center = (n_actions - 1) / 2.0
        raw_delta = (action - center) / center if center > 0 else 0.0
    
    delta = float(np.clip(raw_delta * delta_scale, -delta_clip, delta_clip))
    return delta


def compose_position(
    base_pos: float,
    delta: float,
    budget: float,
    composition: str = "additive",
    hard_floor: float = 0.20,
) -> float:
    """Compose base position and delta according to composition rule."""
    if composition == "additive":
        # final = base + delta * budget
        final = base_pos + delta * budget
    elif composition == "multiplicative":
        # final = base * (1 + delta * budget)
        final = base_pos * (1.0 + delta * budget)
    elif composition == "gate":
        # delta only active when budget allows
        if budget < 0.5:
            final = base_pos
        else:
            final = base_pos + delta * budget
    else:
        final = base_pos + delta * budget

    return float(np.clip(final, hard_floor, 1.0))


def evaluate_hierarchical(
    delta_actor,
    df: pd.DataFrame,
    cfg: HierarchicalConfig,
    base_actor=None,
    cash0: float = 100_000.0,
    position_cap_config=None,
    execution_config=None,
    action_levels=None,
    observation_window: int = 1,
    feature_columns: Optional[list] = None,
    profile=None,
) -> Dict[str, any]:
    """Evaluate hierarchical two-timescale policy on a single dataframe."""
    cash = cash0
    peak = cash0
    pos = 0.0
    ledger = TrancheLedger()
    exec_cfg = execution_config or DEFAULT_EXECUTION
    buy_cooldown = 0
    trade_lock = 0
    price_path = build_price_path(df)
    values = [cash]
    positions = []
    stress_positions = []
    mode_state = ModeState()

    # Upper layer state
    base_pos = 0.0
    current_mode = "flat_wait"
    current_budget = 1.0
    days_since_base_update = 0

    for idx in range(LOOKBACK, len(df) - 1):
        obs = make_observation(
            df, idx, pos, cash, peak, ledger, price_path,
            observation_window=observation_window, feature_columns=feature_columns,
        )

        # === UPPER LAYER (slow) ===
        days_since_base_update += 1
        if days_since_base_update >= cfg.slow_freq:
            days_since_base_update = 0

            # Update mode
            if cfg.mode_router_type == "rule":
                current_mode = classify_rule_mode(obs)
            elif cfg.mode_router_type == "learned":
                from src.soxl_quant.policy.modes import classify_learned_mode
                current_mode = classify_learned_mode(obs)
            else:
                current_mode = "flat_wait"

            # Update base position
            base_pos = get_base_position(
                obs, cfg.base_policy_type, base_actor,
                cfg.base_constant_pos, cfg.hard_floor,
            )

            # Update risk budget
            current_budget = mode_to_budget(current_mode, cfg)

        # === LOWER LAYER (fast) ===
        delta = get_delta(obs, delta_actor, cfg.delta_scale, cfg.delta_clip)

        # === COMPOSITION ===
        target_pos = compose_position(
            base_pos, delta, current_budget,
            cfg.composition, cfg.hard_floor,
        )

        # === EXECUTION (same as existing overlay scripts) ===
        row = df.iloc[idx]
        regime_risk = float(np.nan_to_num(row["regime_risk"]))
        realized_vol = float(np.nan_to_num(row["realized_vol_10"]))
        trend_strength = abs(float(np.nan_to_num(row["trend_gap_21"])))
        cum_drop_30 = float(np.nan_to_num(row["cum_drop_30"]))
        relative_strength = float(np.nan_to_num(row["relative_strength_21"]))
        vix_zscore = float(np.nan_to_num(row["vix_zscore"]))
        downside_ratio_10 = float(np.nan_to_num(row["downside_ratio_10"]))

        regime = execution_regime(
            regime_risk, realized_vol, trend_strength, cum_drop_30=cum_drop_30,
            relative_strength=relative_strength, vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10, config=exec_cfg,
        )
        direction = "buy" if target_pos > pos else ("sell" if target_pos < pos else "neutral")
        min_rebalance = rebalance_threshold(
            regime_risk, realized_vol, trend_strength, cum_drop_30=cum_drop_30,
            relative_strength=relative_strength, vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            direction=direction, mode=current_mode,
            config=exec_cfg,
        )
        if trade_lock > 0:
            if abs(target_pos - pos) >= min_rebalance:
                target_pos = pos
            trade_lock -= 1
        if buy_cooldown > 0:
            if target_pos > pos:
                target_pos = pos
            buy_cooldown -= 1
        if abs(target_pos - pos) < min_rebalance:
            target_pos = pos

        next_ret = float(df.iloc[idx + 1]["return"])
        turnover = abs(target_pos - pos)
        cash *= 1.0 + target_pos * next_ret - turnover * 0.001
        peak = max(peak, cash)
        pos = target_pos

        values.append(cash)
        positions.append(pos)
        if regime_risk > 0.65:
            stress_positions.append(pos)

        ledger.update(pos, target_pos, float(price_path[idx]))

    returns = pd.Series(values).pct_change().dropna()
    total_return = cash / cash0 - 1.0
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    running_max = pd.Series(values).cummax()
    drawdowns = (running_max - pd.Series(values)) / running_max
    max_dd = drawdowns.max()
    avg_pos = np.mean(positions)
    stress_avg_pos = np.mean(stress_positions) if stress_positions else 0.0
    trades = sum(1 for i in range(1, len(positions)) if abs(positions[i] - positions[i - 1]) > 1e-6)

    return {
        "return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_pos": avg_pos,
        "stress_avg_pos": stress_avg_pos,
        "trades": trades,
        "final_value": cash,
        "values": values,
        "positions": positions,
    }


def main():
    parser = argparse.ArgumentParser(description="Two-timescale hierarchical overlay prototype")
    parser.add_argument("--delta-actor", type=str, required=True, help="Path to delta (fast) actor checkpoint")
    parser.add_argument("--base-actor", type=str, default=None, help="Path to base (slow) actor checkpoint")
    parser.add_argument("--base-policy", type=str, default="rule", choices=["rule", "actor", "constant"])
    parser.add_argument("--base-constant-pos", type=float, default=0.50)
    parser.add_argument("--mode-router", type=str, default="rule", choices=["rule", "learned", "none"])
    parser.add_argument("--slow-freq", type=int, default=5, help="Days between base position updates")
    parser.add_argument("--delta-scale", type=float, default=1.00)
    parser.add_argument("--delta-clip", type=float, default=0.30)
    parser.add_argument("--composition", type=str, default="additive", choices=["additive", "multiplicative", "gate"])
    parser.add_argument("--trend-budget", type=float, default=1.00)
    parser.add_argument("--defensive-budget", type=float, default=0.60)
    parser.add_argument("--crash-budget", type=float, default=0.80)
    parser.add_argument("--flat-budget", type=float, default=0.30)
    parser.add_argument("--hard-floor", type=float, default=0.20)
    parser.add_argument("--profile", type=str, default="balanced", help="Profile for feature/execution config")
    parser.add_argument("--output", type=str, default="runtime/post_train_review/hierarchical_overlay.json")
    parser.add_argument("--skip-data-refresh", action="store_true")
    args = parser.parse_args()

    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(args.profile)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)

    # Build delta actor from bundle
    delta_actor, _ = load_actor_bundle(args.delta_actor, profile)

    # Build base actor if needed
    base_actor = None
    if args.base_policy == "actor" and args.base_actor:
        base_actor, _ = load_actor_bundle(args.base_actor, profile)

    # Build datasets
    dataset = build_dataset(
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    df = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    # Config
    hcfg = HierarchicalConfig(
        slow_freq=args.slow_freq,
        base_policy_type=args.base_policy,
        base_constant_pos=args.base_constant_pos,
        mode_router_type=args.mode_router,
        trend_hold_budget=args.trend_budget,
        defensive_hold_budget=args.defensive_budget,
        crash_rebuild_budget=args.crash_budget,
        flat_wait_budget=args.flat_budget,
        delta_scale=args.delta_scale,
        delta_clip=args.delta_clip,
        composition=args.composition,
        hard_floor=args.hard_floor,
    )

    # Evaluate on full test period
    result = evaluate_hierarchical(
        delta_actor, df, hcfg,
        base_actor=base_actor,
        position_cap_config=profile.position_cap,
        execution_config=profile.execution,
        action_levels=None,
        observation_window=max(int(profile.model.observation_window), 1),
        feature_columns=feature_columns,
        profile=profile,
    )

    # Slice metrics
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

    output = {
        "config": {
            "slow_freq": args.slow_freq,
            "base_policy": args.base_policy,
            "mode_router": args.mode_router,
            "composition": args.composition,
            "delta_scale": args.delta_scale,
            "delta_clip": args.delta_clip,
            "budgets": {
                "trend_hold": args.trend_budget,
                "defensive_hold": args.defensive_budget,
                "crash_rebuild": args.crash_budget,
                "flat_wait": args.flat_budget,
            },
        },
        "proxy": {
            "return": result["return"],
            "sharpe": result["sharpe"],
            "max_dd": result["max_dd"],
            "avg_pos": result["avg_pos"],
            "stress_avg_pos": result["stress_avg_pos"],
            "trades": result["trades"],
        },
        "test_2020": slice_metrics(result["values"], result["positions"], test_2020_mask),
        "test_2022": slice_metrics(result["values"], result["positions"], test_2022_mask),
        "test_2023_2024": slice_metrics(result["values"], result["positions"], test_2023_2024_mask),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Hierarchical overlay result written to {args.output}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
