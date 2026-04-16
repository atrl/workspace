"""Ablate each MoE expert by forcing it to make all decisions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import build_dataset, segment_frame, init_qlib
from src.soxl_quant.rl import (
    action_to_position,
    build_price_path,
    make_observation,
    PositionLedger,
    PositionCapConfig,
    LOOKBACK,
    RegimeMoENet,
)
from src.soxl_quant.settings import load_profile


def evaluate_with_forced_expert(actor, test_df, price_path, profile, forced_expert: int):
    """Run a fast evaluator forcing a single expert."""
    pos = 0.0
    val = 100_000.0
    peak = 100_000.0
    ledger = PositionLedger()
    positions = []
    values = [val]
    trades = 0

    net = actor.preprocess
    for idx in range(LOOKBACK, len(test_df) - 1):
        obs = make_observation(test_df, idx, pos, val, peak, ledger, price_path, asset_idx=0)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            # gate
            gate_logits = net.gate(x)
            gate_weights = torch.zeros_like(gate_logits)
            gate_weights[0, forced_expert] = 1.0
            # experts
            expert_outs = torch.stack([expert(x) for expert in net.experts], dim=1)
            routed = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
            shared = net.shared_expert(x)
            hidden = net.output(torch.cat([routed, shared], dim=-1))
            logits = actor.last(hidden)
            action = int(logits.argmax(-1).item())

        target_pos = action_to_position(action, obs, config=profile.position_cap)
        if abs(target_pos - pos) >= 0.001:
            trades += 1
        next_ret = float(test_df.iloc[idx + 1]["return"])
        turnover = abs(target_pos - pos)
        val *= 1.0 + target_pos * next_ret - turnover * 0.001
        peak = max(peak, val)
        ledger.update(pos, target_pos, float(price_path[idx]))
        pos = target_pos
        positions.append(pos)
        values.append(val)

    account = np.array(values)
    daily_ret = np.diff(account) / account[:-1]
    peak_curve = np.maximum.accumulate(account)
    return {
        "return": (account[-1] / account[0] - 1) * 100,
        "sharpe": float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)),
        "max_dd": float(np.max((peak_curve - account) / peak_curve) * 100),
        "avg_pos": float(np.mean(positions)) if positions else 0.0,
        "trades": trades,
    }


def main():
    init_qlib(ROOT / "qlib_soxl_binary")
    dataset = build_dataset()
    test_df = segment_frame(dataset, "test")
    price_path = build_price_path(test_df)
    profile = load_profile("balanced")

    ckpt = torch.load(ROOT / "artifacts" / "models" / "soxl_balanced_moe_actor.pt", map_location="cpu", weights_only=False)
    obs_dim = ckpt["obs_dim"]
    net = RegimeMoENet(obs_dim, hidden_size=64, n_experts=3, device="cpu", top_k=1, noise_std=0.1)
    actor = torch.nn.Sequential(net, torch.nn.Linear(64, 11))
    # Load weights manually because build_policy uses tianshou Actor wrapper
    from src.soxl_quant.rl import build_policy
    policy, actor_obj = build_policy(obs_dim, profile, use_moe=True)
    actor_obj.load_state_dict(ckpt["state_dict"])
    actor_obj.eval()

    # Also run the natural gate path for comparison
    from src.soxl_quant.rl import actor_action
    def natural_eval():
        pos = 0.0
        val = 100_000.0
        peak = 100_000.0
        ledger = PositionLedger()
        positions = []
        values = [val]
        trades = 0
        for idx in range(LOOKBACK, len(test_df) - 1):
            obs = make_observation(test_df, idx, pos, val, peak, ledger, price_path, asset_idx=0)
            action = actor_action(actor_obj, obs)
            target_pos = action_to_position(action, obs, config=profile.position_cap)
            if abs(target_pos - pos) >= 0.001:
                trades += 1
            next_ret = float(test_df.iloc[idx + 1]["return"])
            turnover = abs(target_pos - pos)
            val *= 1.0 + target_pos * next_ret - turnover * 0.001
            peak = max(peak, val)
            ledger.update(pos, target_pos, float(price_path[idx]))
            pos = target_pos
            positions.append(pos)
            values.append(val)
        account = np.array(values)
        daily_ret = np.diff(account) / account[:-1]
        peak_curve = np.maximum.accumulate(account)
        return {
            "return": (account[-1] / account[0] - 1) * 100,
            "sharpe": float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)),
            "max_dd": float(np.max((peak_curve - account) / peak_curve) * 100),
            "avg_pos": float(np.mean(positions)) if positions else 0.0,
            "trades": trades,
        }

    print("=" * 70)
    print("MoE Expert Ablation on SOXL Test Set")
    print("=" * 70)

    nat = natural_eval()
    print(f"\n[natural]   return={nat['return']:.1f}% sharpe={nat['sharpe']:.2f} maxdd={nat['max_dd']:.1f}% trades={nat['trades']} avg_pos={nat['avg_pos']:.2f}")

    for exp in range(3):
        res = evaluate_with_forced_expert(actor_obj, test_df, price_path, profile, exp)
        print(f"[expert {exp}] return={res['return']:.1f}% sharpe={res['sharpe']:.2f} maxdd={res['max_dd']:.1f}% trades={res['trades']} avg_pos={res['avg_pos']:.2f}")


if __name__ == "__main__":
    main()
