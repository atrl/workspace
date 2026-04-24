#!/usr/bin/env python3
"""Fit a learned mode classifier to replace the hard-coded classify_rule_mode.

Training data is generated from historical SOXL bars. Labels come from a
hindsight-optimal analysis: for each day, we look at the next N days of
returns and drawdowns to determine which mode *would have been* optimal.
This produces smoother, transition-aware labels than the existing rule-based
classifier.

Usage:
    PYTHONPATH=. qlib_venv/bin/python scripts/fit_learned_mode_classifier.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.soxl_quant.data import build_dataset, init_qlib
from src.soxl_quant.settings import QLIB_DATA_PATH

# Features used by the original classify_rule_mode (plus a few extras)
FEATURE_COLS = [
    "regime_risk",
    "realized_vol_10",
    "trend_gap_21",
    "anchor_gap_180",
    "cum_drop_30",
    "relative_strength_21",
    "vix_zscore",
    "downside_ratio_10",
    # Extras that help transition discrimination
    "drawdown_streak_63",
    "below_anchor_ratio_63",
    "regime_risk_velocity_5",
]

MODE_NAMES = ["flat_wait", "crash_rebuild", "trend_hold", "defensive_hold"]


class ModeClassifierNet(nn.Module):
    """Small MLP that classifies regime mode + predicts forward drawdown."""

    def __init__(self, n_features: int = 11, n_modes: int = 4, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mode_head = nn.Linear(hidden // 2, n_modes)
        self.dd_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        mode_logits = self.mode_head(h)
        dd_pred = torch.sigmoid(self.dd_head(h))  # predict dd as fraction [0,1]
        return mode_logits, dd_pred


def hindsight_optimal_mode(
    df: pd.DataFrame,
    idx: int,
    forward_window: int = 10,
    danger_dd: float = 0.12,
    danger_ret: float = -0.03,
    trend_return: float = 0.06,
    trend_dd_limit: float = 0.05,
    rebuild_return: float = 0.08,
) -> str:
    """Determine the mode that would have been optimal from idx looking forward.

    Tightened thresholds to avoid over-labeling flat_wait.
    """
    if idx + forward_window >= len(df):
        return "defensive_hold"

    future = df.iloc[idx + 1 : idx + 1 + forward_window]
    fwd_returns = future["return"].to_numpy()
    fwd_cum = np.cumprod(1.0 + fwd_returns) - 1.0
    fwd_max_dd = float(np.max(np.maximum.accumulate(1 + fwd_cum) - (1 + fwd_cum)))
    fwd_total_ret = float(fwd_cum[-1])

    current_cum_drop = float(df.iloc[idx]["cum_drop_30"])
    current_regime_risk = float(df.iloc[idx]["regime_risk"])

    # flat_wait: only when BOTH deep drawdown AND negative return ahead
    if fwd_max_dd >= danger_dd and fwd_total_ret <= danger_ret:
        return "flat_wait"

    # crash_rebuild: deep damage + strong rebound
    if current_cum_drop <= -0.10 and fwd_total_ret >= rebuild_return and current_regime_risk <= 0.55:
        return "crash_rebuild"

    # trend_hold: strong clean uptrend
    if fwd_total_ret >= trend_return and fwd_max_dd < trend_dd_limit:
        return "trend_hold"

    return "defensive_hold"


def build_training_data(
    df: pd.DataFrame,
    forward_window: int = 10,
    lookback: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (features, mode_labels, dd_labels) arrays."""
    rows: List[Dict[str, float]] = []
    for idx in range(lookback, len(df) - forward_window):
        row = df.iloc[idx]
        feats = [float(np.nan_to_num(row.get(c, 0.0))) for c in FEATURE_COLS]
        mode_label = hindsight_optimal_mode(df, idx, forward_window=forward_window)
        # Auxiliary label: forward max drawdown (scaled to [0,1])
        future = df.iloc[idx + 1 : idx + 1 + forward_window]
        fwd_returns = future["return"].to_numpy()
        fwd_cum = np.cumprod(1.0 + fwd_returns) - 1.0
        fwd_max_dd = float(np.max(np.maximum.accumulate(1 + fwd_cum) - (1 + fwd_cum)))
        dd_label = min(fwd_max_dd, 1.0)
        rows.append({"feats": feats, "mode": mode_label, "dd": dd_label})

    X = np.array([r["feats"] for r in rows], dtype=np.float32)
    y_mode = np.array([MODE_NAMES.index(r["mode"]) for r in rows], dtype=np.int64)
    y_dd = np.array([r["dd"] for r in rows], dtype=np.float32).reshape(-1, 1)
    return X, y_mode, y_dd


def train_classifier(
    X: np.ndarray,
    y_mode: np.ndarray,
    y_dd: np.ndarray,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    aux_weight: float = 0.5,
    val_ratio: float = 0.15,
    seed: int = 42,
    class_weights: str | None = None,
) -> ModeClassifierNet:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X)
    n_val = int(n * val_ratio)
    perm = np.random.permutation(n)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    X_train = torch.from_numpy(X[train_idx])
    y_mode_train = torch.from_numpy(y_mode[train_idx])
    y_dd_train = torch.from_numpy(y_dd[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_mode_val = torch.from_numpy(y_mode[val_idx])
    y_dd_val = torch.from_numpy(y_dd[val_idx])

    train_ds = TensorDataset(X_train, y_mode_train, y_dd_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ModeClassifierNet(n_features=X.shape[1], dropout=0.25)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if class_weights == "balanced":
        counts = np.bincount(y_mode, minlength=len(MODE_NAMES))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        class_weight_tensor = torch.from_numpy(weights.astype(np.float32))
        print(f"  Class weights: {dict(zip(MODE_NAMES, weights.round(3).tolist()))}")
    else:
        class_weight_tensor = None
    mode_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    dd_criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb_mode, yb_dd in train_dl:
            opt.zero_grad()
            logits, dd_pred = model(xb)
            loss_mode = mode_criterion(logits, yb_mode)
            loss_dd = dd_criterion(dd_pred, yb_dd)
            loss = loss_mode + aux_weight * loss_dd
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits, v_dd = model(X_val)
            v_loss_mode = mode_criterion(v_logits, y_mode_val).item()
            v_loss_dd = dd_criterion(v_dd, y_dd_val).item()
            v_loss = v_loss_mode + aux_weight * v_loss_dd
            v_acc = float((v_logits.argmax(dim=1) == y_mode_val).float().mean().item())

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d} | train_loss={np.mean(train_losses):.4f} | val_loss={v_loss:.4f} val_acc={v_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def compare_with_rule(df: pd.DataFrame, model: ModeClassifierNet) -> Dict[str, float]:
    """Compare learned classifier against rule-based classifier on historical data."""
    from src.soxl_quant.policy.modes import classify_rule_mode
    from src.soxl_quant.data import STATIC_INDEX

    match = 0
    total = 0
    confusion = {m: {m2: 0 for m2 in MODE_NAMES} for m in MODE_NAMES}
    for idx in range(120, len(df) - 10):
        row = df.iloc[idx]
        feats = np.array([float(np.nan_to_num(row.get(c, 0.0))) for c in FEATURE_COLS], dtype=np.float32)
        obs = np.zeros(max(STATIC_INDEX.values()) + 1, dtype=np.float32)
        for c in FEATURE_COLS:
            if c in STATIC_INDEX:
                obs[STATIC_INDEX[c]] = feats[FEATURE_COLS.index(c)]
        rule_mode = classify_rule_mode(obs)
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(feats).unsqueeze(0))
            learned_mode = MODE_NAMES[int(logits.argmax(dim=1).item())]
        confusion[rule_mode][learned_mode] += 1
        if rule_mode == learned_mode:
            match += 1
        total += 1

    print("\nRule -> Learned confusion matrix:")
    header = "Rule \\ Learned"
    print(f"{header:>16}", end="")
    for m in MODE_NAMES:
        print(f"{m:>14}", end="")
    print()
    for rm in MODE_NAMES:
        print(f"{rm:>16}", end="")
        for lm in MODE_NAMES:
            print(f"{confusion[rm][lm]:>14}", end="")
        print()

    return {
        "agreement_rate": match / max(total, 1),
        "total_samples": total,
        "confusion": confusion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/other/mode_classifier.pt")
    args = parser.parse_args()

    print("[fit_learned_mode_classifier] Building dataset...")
    init_qlib(QLIB_DATA_PATH)
    from src.soxl_quant.data import build_dataset
    dataset = build_dataset(feature_packs=(), profile_name="balanced")
    df = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    print(f"[fit_learned_mode_classifier] Generating labels (forward_window={args.forward_window})...")
    X, y_mode, y_dd = build_training_data(df, forward_window=args.forward_window)
    print(f"  samples={len(X)} | mode distribution: {dict(zip(*np.unique(y_mode, return_counts=True)))}")

    print("[fit_learned_mode_classifier] Training...")
    model = train_classifier(X, y_mode, y_dd, epochs=args.epochs, seed=args.seed, class_weights="balanced")

    print("[fit_learned_mode_classifier] Comparing with rule-based classifier...")
    cmp = compare_with_rule(df, model)
    print(f"  Agreement rate: {cmp['agreement_rate']:.2%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "mode_names": MODE_NAMES,
        "hidden": 32,
        "forward_window": args.forward_window,
        "agreement_rate": cmp["agreement_rate"],
    }, output_path)
    print(f"[fit_learned_mode_classifier] Saved to {output_path}")


if __name__ == "__main__":
    main()
