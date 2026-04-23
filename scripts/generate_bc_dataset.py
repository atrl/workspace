#!/usr/bin/env python3
"""Generate Behavioral Cloning dataset from rulebase or existing actor.

Direction B: Behavioral Cloning + Conservative RL
- Phase 1: Generate high-quality demonstration trajectories
- Phase 2: Train actor via supervised learning (BC)
- Phase 3: Conservative PPO fine-tuning

This script handles Phase 1: dataset generation.

Usage:
    # Generate from rulebase (default)
    PYTHONPATH=. qlib_venv/bin/python scripts/generate_bc_dataset.py \
        --output data/bc_rulebase_dataset.pkl \
        --n-samples 10000

    # Generate from existing actor (behavioral cloning from v4/v19)
    PYTHONPATH=. qlib_venv/bin/python scripts/generate_bc_dataset.py \
        --actor artifacts/models/soxl_balanced_actor.pt \
        --output data/bc_actor_dataset.pkl \
        --n-samples 10000

    # Generate mixed dataset (rulebase + actor)
    PYTHONPATH=. qlib_venv/bin/python scripts/generate_bc_dataset.py \
        --actor artifacts/models/soxl_balanced_actor.pt \
        --mix-ratio 0.5 \
        --output data/bc_mixed_dataset.pkl \
        --n-samples 10000
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib
from src.soxl_quant.policy.environment import TrancheLedger, build_price_path, EpisodeConfig
from src.soxl_quant.policy.observation import make_observation
from src.soxl_quant.policy.policy_builder import actor_action
from src.soxl_quant.policy.positioning import (
    rule_based_position,
    action_levels_for_profile,
    hybrid_overlay_position,
)
from src.soxl_quant.settings import LOOKBACK, QLIB_DATA_PATH, load_profile


def generate_from_rulebase(
    df: pd.DataFrame,
    n_samples: int,
    feature_columns: List[str],
    observation_window: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (obs, action) pairs by sampling rulebase decisions."""
    rng = random.Random(seed)
    price_path = build_price_path(df)
    obs_list = []
    action_list = []

    # Use discrete action levels (0-10 mapping to 0.0-1.0)
    levels = np.linspace(0.0, 1.0, 11)

    attempts = 0
    max_attempts = n_samples * 10

    while len(obs_list) < n_samples and attempts < max_attempts:
        attempts += 1
        idx = rng.randrange(LOOKBACK, len(df) - 1)

        # Random initial position and value for diversity
        pos = rng.uniform(0.0, 1.0)
        value = 100_000.0 * (1.0 + rng.uniform(-0.2, 0.2))
        peak = value * (1.0 + rng.uniform(0.0, 0.3))
        ledger = TrancheLedger()

        obs = make_observation(
            df, idx, pos, value, peak, ledger, price_path,
            observation_window=observation_window, feature_columns=feature_columns,
        )

        # Get rulebase target position
        rule_pos = rule_based_position(obs)

        # Map to nearest discrete action
        action = int(np.argmin(np.abs(levels - rule_pos)))

        obs_list.append(obs)
        action_list.append(action)

    if not obs_list:
        raise RuntimeError("Failed to generate any BC samples")

    return np.stack(obs_list, axis=0), np.asarray(action_list, dtype=np.int64)


def generate_from_actor(
    actor,
    df: pd.DataFrame,
    n_samples: int,
    feature_columns: List[str],
    observation_window: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (obs, action) pairs by rolling out an existing actor."""
    rng = random.Random(seed)
    price_path = build_price_path(df)
    obs_list = []
    action_list = []

    attempts = 0
    max_attempts = n_samples * 10

    while len(obs_list) < n_samples and attempts < max_attempts:
        attempts += 1
        idx = rng.randrange(LOOKBACK, len(df) - 1)

        pos = rng.uniform(0.0, 1.0)
        value = 100_000.0 * (1.0 + rng.uniform(-0.2, 0.2))
        peak = value * (1.0 + rng.uniform(0.0, 0.3))
        ledger = TrancheLedger()

        obs = make_observation(
            df, idx, pos, value, peak, ledger, price_path,
            observation_window=observation_window, feature_columns=feature_columns,
        )

        action = actor_action(actor, obs)

        obs_list.append(obs)
        action_list.append(action)

    if not obs_list:
        raise RuntimeError("Failed to generate any BC samples from actor")

    return np.stack(obs_list, axis=0), np.asarray(action_list, dtype=np.int64)


def generate_mixed(
    actor,
    df: pd.DataFrame,
    n_samples: int,
    mix_ratio: float,
    feature_columns: List[str],
    observation_window: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mixed dataset: mix_ratio from actor, rest from rulebase."""
    n_actor = int(n_samples * mix_ratio)
    n_rule = n_samples - n_actor

    obs_actor, act_actor = generate_from_actor(
        actor, df, n_actor, feature_columns, observation_window, seed=seed,
    )
    obs_rule, act_rule = generate_from_rulebase(
        df, n_rule, feature_columns, observation_window, seed=seed + 1,
    )

    obs = np.vstack([obs_actor, obs_rule])
    actions = np.concatenate([act_actor, act_rule])

    # Shuffle
    perm = np.random.RandomState(seed + 2).permutation(len(obs))
    return obs[perm], actions[perm]


def main():
    parser = argparse.ArgumentParser(description="Generate BC dataset")
    parser.add_argument("--actor", type=str, default=None, help="Path to actor checkpoint (optional)")
    parser.add_argument("--mix-ratio", type=float, default=0.0, help="Ratio of actor vs rulebase samples (0=rulebase only)")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/bc_dataset.pkl", help="Output path")
    parser.add_argument("--profile", type=str, default="balanced", help="Profile for config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-data-refresh", action="store_true")
    args = parser.parse_args()

    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(args.profile)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    observation_window = max(int(profile.model.observation_window), 1)

    dataset = build_dataset(
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    df = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    # Load actor if needed
    actor = None
    if args.actor or args.mix_ratio > 0:
        from src.soxl_quant.policy.policy_builder import build_policy
        bundle = torch.load(args.actor, map_location="cpu")
        obs_dim = int(bundle["obs_dim"])
        _, actor = build_policy(obs_dim, profile)
        actor.load_state_dict(bundle["state_dict"])
        actor.eval()

    # Generate dataset
    if args.mix_ratio > 0 and actor is not None:
        obs, actions = generate_mixed(
            actor, df, args.n_samples, args.mix_ratio,
            feature_columns, observation_window, args.seed,
        )
    elif actor is not None:
        obs, actions = generate_from_actor(
            actor, df, args.n_samples, feature_columns, observation_window, args.seed,
        )
    else:
        obs, actions = generate_from_rulebase(
            df, args.n_samples, feature_columns, observation_window, args.seed,
        )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"obs": obs, "actions": actions, "n_samples": len(obs)}, f)

    # Stats
    action_counts = np.bincount(actions, minlength=11)
    print(f"BC dataset saved to {args.output}")
    print(f"  Total samples: {len(obs)}")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Action distribution:")
    for i, count in enumerate(action_counts):
        if count > 0:
            print(f"    action {i}: {count} ({count/len(actions)*100:.1f}%)")


if __name__ == "__main__":
    main()
