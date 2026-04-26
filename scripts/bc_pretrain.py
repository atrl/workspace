#!/usr/bin/env python3
"""Behavioral Cloning pretrain on generated dataset.

Direction B Phase 2: Train actor via supervised learning on BC dataset.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/bc_pretrain.py \
        --bc-dataset data/bc_rulebase_10k.pkl \
        --profile balanced \
        --output runtime/model/card/bc_pretrain/bc_pretrain_actor.pt \
        --steps 5000 --lr 0.0003 --batch-size 256
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from src.soxl_quant.data import feature_columns_for_packs, init_qlib
from src.soxl_quant.policy.policy_builder import build_policy, supervised_pretrain
from src.soxl_quant.settings import QLIB_DATA_PATH, load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-dataset", type=str, required=True)
    parser.add_argument("--profile", type=str, default="balanced")
    parser.add_argument("--output", type=str, default="runtime/model/card/bc_pretrain/bc_pretrain_actor.pt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(args.profile)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    obs_dim = len(feature_columns) * max(int(profile.model.observation_window), 1) + 19 + 1

    # Build policy (gets actor + critic)
    policy, actor = build_policy(obs_dim, profile)

    # Load BC dataset
    with open(args.bc_dataset, "rb") as f:
        data = pickle.load(f)
    obs_array = data["obs"]
    action_array = data["actions"]

    print(f"[BC] Dataset: {len(obs_array)} samples, obs_dim={obs_array.shape[1]}")
    print(f"[BC] Action distribution: {np.bincount(action_array, minlength=11)}")

    # Pretrain
    metrics = supervised_pretrain(
        actor, obs_array, action_array,
        steps=args.steps, lr=args.lr, batch_size=args.batch_size,
    )

    print(f"[BC] Pretrain complete: loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f}")

    # Save as bundle
    from src.soxl_quant.app.artifacts import persist_actor_bundle
    model_path = persist_actor_bundle(profile, actor, obs_dim, args.seed, "bc_pretrain")
    print(f"[BC] Saved to {model_path}")


if __name__ == "__main__":
    main()
