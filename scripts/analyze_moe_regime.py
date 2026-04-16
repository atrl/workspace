"""Analyze MoE gate routing vs market regimes and resulting positions."""
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
    actor_action,
    build_policy,
    make_observation,
    PositionLedger,
    build_price_path,
    LOOKBACK,
)
from src.soxl_quant.settings import load_profile


def load_actor(checkpoint_path: Path, profile):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    policy, actor = build_policy(obs_dim, profile, use_moe=True)
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor


def extract_gate_probs(actor, obs: np.ndarray):
    """Return softmax gate probabilities for a single observation."""
    net = actor.preprocess
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = net.gate(x)
        probs = torch.softmax(logits, dim=-1)
    return probs.squeeze(0).cpu().numpy()


def main():
    init_qlib(ROOT / "qlib_soxl_binary")
    dataset = build_dataset()
    test_df = segment_frame(dataset, "test")
    price_path = build_price_path(test_df)

    profile = load_profile("balanced")
    actor = load_actor(ROOT / "artifacts" / "models" / "soxl_balanced_moe_actor.pt", profile)

    records = []
    ledger = PositionLedger()
    pos = 0.0
    peak_val = 100_000.0
    val = 100_000.0

    for idx in range(LOOKBACK, len(test_df) - 1):
        obs = make_observation(test_df, idx, pos, val, peak_val, ledger, price_path, asset_idx=0)
        action = actor_action(actor, obs)
        target_pos = action_to_position(action, obs, config=profile.position_cap)

        gate_probs = extract_gate_probs(actor, obs)
        selected_expert = int(gate_probs.argmax())

        row = test_df.iloc[idx]
        records.append({
            "date": row.name,
            "regime_risk": float(row["regime_risk"]),
            "realized_vol_10": float(row["realized_vol_10"]),
            "cum_drop_30": float(row["cum_drop_30"]),
            "anchor_gap_180": float(row["anchor_gap_180"]),
            "trend_gap_21": float(row["trend_gap_21"]),
            "vol_coef_21": float(row["vol_coef_21"]),
            "vix_zscore": float(row["vix_zscore"]),
            "action": int(action),
            "position": float(target_pos),
            "expert_0": float(gate_probs[0]),
            "expert_1": float(gate_probs[1]),
            "expert_2": float(gate_probs[2]),
            "selected_expert": selected_expert,
        })

        next_ret = float(test_df.iloc[idx + 1]["return"])
        turnover = abs(target_pos - pos)
        val *= 1.0 + target_pos * next_ret - turnover * 0.001
        peak_val = max(peak_val, val)
        ledger.update(pos, target_pos, float(price_path[idx]))
        pos = target_pos

    df = pd.DataFrame(records).set_index("date")

    print("=" * 70)
    print("MoE Regime Routing Analysis")
    print("=" * 70)

    # Overall expert usage
    print("\n[Overall expert selection]")
    print(df["selected_expert"].value_counts(normalize=True).sort_index())

    # Regime buckets
    df["regime_bucket"] = pd.cut(df["regime_risk"], bins=[0, 0.35, 0.55, 0.75, 1.0],
                                  labels=["low", "mid", "high", "extreme"])
    print("\n[Expert selection by regime_risk bucket]")
    print(pd.crosstab(df["regime_bucket"], df["selected_expert"], normalize="index"))

    # Volatility buckets
    df["vol_bucket"] = pd.cut(df["realized_vol_10"], bins=[0, 0.35, 0.55, 0.85, 10.0],
                               labels=["low", "mid", "high", "extreme"])
    print("\n[Expert selection by realized_vol bucket]")
    print(pd.crosstab(df["vol_bucket"], df["selected_expert"], normalize="index"))

    # Position stats per expert
    print("\n[Position statistics by selected expert]")
    print(df.groupby("selected_expert")["position"].describe())

    # Position stats per regime bucket
    print("\n[Position statistics by regime bucket]")
    print(df.groupby("regime_bucket")["position"].describe())

    # Action distribution per expert
    print("\n[Mean action by expert]")
    print(df.groupby("selected_expert")["action"].mean())

    # Save for manual inspection
    out_path = ROOT / "runtime" / "moe_regime_analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\nDetailed trace saved to {out_path}")


if __name__ == "__main__":
    main()
