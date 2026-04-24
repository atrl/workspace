#!/usr/bin/env python3
"""Prototype Approach C: Drawdown-budget action masking on a fixed actor.

This script evaluates an existing actor with an optional drawdown-budget
position cap. It does NOT retrain anything — it measures the mechanical
celling of hard position constraints on a learned policy.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget.py \
        --profile balanced_rule_residual_v19 --budget 0.30

    PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget.py \
        --profile balanced_rule_residual_mode_v1 --budget 0.28
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
from src.soxl_quant.policy.evaluation import evaluate_split
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


@dataclass(frozen=True)
class BudgetConfig:
    """Position-level drawdown budget configuration."""

    budget: float = 0.30
    # (consumed_start, consumed_end, max_position) tuples
    thresholds: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.30, 1.00),
        (0.30, 0.60, 0.75),
        (0.60, 0.90, 0.50),
        (0.90, 1.00, 0.25),
        (1.00, float("inf"), 0.10),
    )
    regime_conditional: bool = True
    bull_budget_bonus: float = 0.05  # extra budget in low-vol bull regimes
    bear_budget_penalty: float = 0.05  # reduced budget in high-vol bear regimes


def max_position_for_budget(
    current_drawdown: float,
    budget: float,
    thresholds: Tuple[Tuple[float, float, float], ...],
    regime_risk: float = 0.5,
    realized_vol: float = 0.4,
    regime_conditional: bool = True,
    bull_budget_bonus: float = 0.05,
    bear_budget_penalty: float = 0.05,
) -> float:
    """Return the maximum allowed position given current drawdown budget state."""
    effective_budget = budget
    if regime_conditional:
        if regime_risk < 0.35 and realized_vol < 0.40:
            effective_budget += bull_budget_bonus
        elif regime_risk > 0.65 or realized_vol > 0.60:
            effective_budget -= bear_budget_penalty
        effective_budget = max(effective_budget, 0.10)

    consumed = current_drawdown / max(effective_budget, 1e-6)
    for start, end, max_pos in thresholds:
        if start <= consumed < end:
            return float(max_pos)
    # Fallback if consumed >= last threshold end
    return float(thresholds[-1][2])


def evaluate_split_with_budget(
    actor,
    df: pd.DataFrame,
    budget_cfg: BudgetConfig,
    cash0: float = 100_000.0,
    position_cap_config=None,
    execution_config=None,
    asset_idx: int = 0,
    stress_threshold: float = 0.65,
    action_levels: Optional[np.ndarray] = None,
    observation_window: int = 1,
    hard_floor: Optional[float] = None,
    use_hybrid_overlay: bool = False,
    delta_levels: Optional[np.ndarray] = None,
    feature_columns: Optional[List[str]] = None,
    mode_router_config=None,
) -> Dict[str, float]:
    """Variant of evaluate_split that clips position based on drawdown budget."""
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

    for idx in range(LOOKBACK, len(df) - 1):
        obs = make_observation(
            df,
            idx,
            pos,
            cash,
            peak,
            ledger,
            price_path,
            asset_idx=asset_idx,
            observation_window=observation_window,
            feature_columns=feature_columns,
        )
        mode_state = route_mode(obs, mode_state, mode_router_config) if mode_router_config is not None else ModeState(name="flat_wait", days_in_mode=1)
        action = actor_action(actor, obs)
        target_pos = target_position_for_profile(
            action,
            obs,
            config=position_cap_config,
            action_levels=action_levels,
            hard_floor=hard_floor,
            use_hybrid_overlay=use_hybrid_overlay,
            delta_levels=delta_levels,
            mode_band=mode_band(mode_state.name, mode_router_config) if mode_router_config and getattr(mode_router_config, "enabled", False) else None,
        )

        row = df.iloc[idx]
        regime_risk = float(np.nan_to_num(row["regime_risk"]))
        realized_vol = float(np.nan_to_num(row["realized_vol_10"]))
        trend_strength = abs(float(np.nan_to_num(row["trend_gap_21"])))
        cum_drop_30 = float(np.nan_to_num(row["cum_drop_30"]))
        relative_strength = float(np.nan_to_num(row["relative_strength_21"]))
        vix_zscore = float(np.nan_to_num(row["vix_zscore"]))
        downside_ratio_10 = float(np.nan_to_num(row["downside_ratio_10"]))

        # --- Drawdown budget constraint (Approach C) ---
        current_drawdown = abs(cash / max(peak, 1e-8) - 1.0)
        max_allowed_pos = max_position_for_budget(
            current_drawdown=current_drawdown,
            budget=budget_cfg.budget,
            thresholds=budget_cfg.thresholds,
            regime_risk=regime_risk,
            realized_vol=realized_vol,
            regime_conditional=budget_cfg.regime_conditional,
            bull_budget_bonus=budget_cfg.bull_budget_bonus,
            bear_budget_penalty=budget_cfg.bear_budget_penalty,
        )
        target_pos = min(target_pos, max_allowed_pos)
        # --- End drawdown budget constraint ---

        regime = execution_regime(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            config=exec_cfg,
        )
        direction = "buy" if target_pos > pos else ("sell" if target_pos < pos else "neutral")
        min_rebalance = rebalance_threshold(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            direction=direction,
            mode=mode_state.name if mode_router_config and getattr(mode_router_config, "enabled", False) else None,
            config=exec_cfg,
            mode_router_config=mode_router_config,
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
        ledger.update(pos, target_pos, float(price_path[idx]))
        if target_pos > pos + 1e-6:
            buy_cooldown = execution_buy_cooldown_days(cum_drop_30, exec_cfg, regime)
        if turnover > exec_cfg.min_trade_turnover:
            trade_lock = execution_trade_lock_days(exec_cfg, regime)
        pos = target_pos
        positions.append(pos)
        if regime_risk >= stress_threshold:
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
        "stress_days": float(len(stress_positions)),
    }


def load_actor_for_profile(profile_name: str, model_path: Optional[Path] = None):
    """Load actor and profile from saved bundle."""
    profile = load_profile(profile_name)
    bundle_path = model_path or Path(f"artifacts/models/soxl_{profile_name}_actor.pt")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model not found: {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu")
    obs_dim = int(bundle["obs_dim"])
    _, actor = build_policy(obs_dim, profile)
    actor.load_state_dict(bundle["state_dict"])
    actor.eval()
    return actor, profile, bundle


def run_sweep(
    profile_name: str,
    model_path: Optional[Path] = None,
    budgets: Optional[List[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the actor across multiple budget configs and return metrics table."""
    budgets = budgets or [None, 0.35, 0.30, 0.28, 0.25]
    init_qlib(QLIB_DATA_PATH)
    actor, profile, bundle = load_actor_for_profile(profile_name, model_path)
    dataset = build_dataset(
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    full_df = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    slices = {
        "test_proxy": ("2020-01-01", "2026-03-31"),
        "test_2020": ("2020-01-01", "2020-12-31"),
        "test_2022": ("2022-01-01", "2022-12-31"),
        "test_2023_2024": ("2023-01-01", "2024-12-31"),
    }

    action_levels = np.asarray(profile.model.action_levels, dtype=np.float32)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)

    results: Dict[str, Dict[str, float]] = {}

    for budget in budgets:
        label = "no_budget" if budget is None else f"budget_{budget:.2f}"
        budget_cfg = BudgetConfig(budget=budget) if budget is not None else None

        for slice_name, (start, end) in slices.items():
            segment = full_df.loc[start:end].copy()
            if len(segment) <= 120:
                continue
            if budget_cfg is not None:
                metrics = evaluate_split_with_budget(
                    actor,
                    segment,
                    budget_cfg=budget_cfg,
                    position_cap_config=profile.position_cap,
                    execution_config=profile.execution,
                    action_levels=action_levels,
                    observation_window=profile.model.observation_window,
                    hard_floor=profile.model.hard_floor,
                    feature_columns=feature_columns,
                    mode_router_config=profile.mode_router,
                )
            else:
                metrics = evaluate_split(
                    actor,
                    segment,
                    position_cap_config=profile.position_cap,
                    execution_config=profile.execution,
                    action_levels=action_levels,
                    observation_window=profile.model.observation_window,
                    hard_floor=profile.model.hard_floor,
                    feature_columns=feature_columns,
                    mode_router_config=profile.mode_router,
                )
            for k, v in metrics.items():
                results[f"{label}_{slice_name}_{k}"] = v

    return results


def main():
    parser = argparse.ArgumentParser(description="Prototype drawdown-budget action masking")
    parser.add_argument("--profile", default="balanced_rule_residual_v19", help="Profile name")
    parser.add_argument("--model-path", default=None, help="Optional custom model path")
    parser.add_argument("--budgets", default="no_budget,0.35,0.30,0.28,0.25", help="Comma-separated budget values")
    parser.add_argument("--output", default="runtime/model/reviews/drawdown_budget_sweep.json", help="Output JSON path")
    args = parser.parse_args()

    budget_list: List[Optional[float]] = []
    for b in args.budgets.split(","):
        b = b.strip()
        if b.lower() in ("none", "no_budget", "null", ""):
            budget_list.append(None)
        else:
            budget_list.append(float(b))

    print(f"[prototype_drawdown_budget] profile={args.profile} budgets={budget_list}")
    results = run_sweep(
        profile_name=args.profile,
        model_path=Path(args.model_path) if args.model_path else None,
        budgets=budget_list,
    )

    # Pretty-print table
    print("\n" + "=" * 90)
    print(f"{'Slice':<18} {'Metric':<12} ", end="")
    for b in budget_list:
        print(f"{('no_budget' if b is None else f'b{b:.2f}'):>12}", end="")
    print()
    print("=" * 90)

    slices = ["test_proxy", "test_2020", "test_2022", "test_2023_2024"]
    metrics = ["return", "sharpe", "max_dd", "avg_pos"]
    for slice_name in slices:
        for metric in metrics:
            print(f"{slice_name:<18} {metric:<12} ", end="")
            for b in budget_list:
                label = "no_budget" if b is None else f"budget_{b:.2f}"
                key = f"{label}_{slice_name}_{metric}"
                val = results.get(key, float("nan"))
                if metric in ("return", "max_dd"):
                    print(f"{val:>11.1f}%", end="")
                elif metric == "sharpe":
                    print(f"{val:>12.2f}", end="")
                else:
                    print(f"{val:>12.3f}", end="")
            print()
        print("-" * 90)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote detailed results to {output_path}")


if __name__ == "__main__":
    main()
