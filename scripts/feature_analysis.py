#!/usr/bin/env python3
"""Feature IC and SHAP analysis for the SOXL actor."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import build_dataset, flatten_feature_frame, init_qlib, segment_frame
from src.soxl_quant.data import FEATURE_COLUMNS
from src.soxl_quant.policy.environment import TrancheLedger, build_price_path
from src.soxl_quant.policy.observation import EXTRA_INDEX, make_observation
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import (
    LOOKBACK,
    QLIB_DATA_PATH,
    load_profile,
    model_artifact_path,
)

PROFILE_NAME = "balanced"

def load_actor(profile_name: str):
    profile = load_profile(profile_name)
    model_path = model_artifact_path(profile_name)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    policy, actor = build_policy(obs_dim, profile, use_moe=ckpt.get("use_moe", False))
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor, profile, obs_dim



def build_observation_matrix(df: pd.DataFrame, price_path: np.ndarray, asset_idx: int = 0):
    """Build all observations for a dataframe (neutral position)."""
    obs_list = []
    ledger = TrancheLedger()
    pos = 0.0
    val = 100_000.0
    peak = 100_000.0
    for idx in range(LOOKBACK, len(df)):
        obs = make_observation(df, idx, pos, val, peak, ledger, price_path, asset_idx=asset_idx)
        obs_list.append(obs)
        if idx < len(df) - 1:
            ledger.update(pos, pos, float(price_path[idx]))
    return np.array(obs_list, dtype=np.float32)


def compute_feature_ic(df: pd.DataFrame):
    """Compute Spearman rank IC for each feature vs next-day return."""
    features = FEATURE_COLUMNS.copy()
    next_ret = df["return"].shift(-1)
    ic_records = []
    for feat in features:
        mask = df[feat].notna() & next_ret.notna()
        if mask.sum() < 30:
            continue
        ic = df.loc[mask, feat].corr(next_ret[mask], method="spearman")
        ic_records.append({"feature": feat, "ic": float(ic)})
    ic_df = pd.DataFrame(ic_records).sort_values("ic", key=abs, ascending=False)
    return ic_df


def shap_analysis(actor, df: pd.DataFrame, price_path: np.ndarray, n_background: int = 100, n_samples: int = 300):
    """Run SHAP on actor action predictions."""
    obs_matrix = build_observation_matrix(df, price_path)
    
    def model_fn(obs_batch):
        with torch.no_grad():
            tensor = torch.as_tensor(obs_batch, dtype=torch.float32)
            out = actor(tensor)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=-1)
            actions = torch.arange(logits.shape[-1], dtype=torch.float32)
            expected_action = (probs * actions).sum(dim=-1).numpy()
            return expected_action
    
    bg_idx = np.random.choice(len(obs_matrix), min(n_background, len(obs_matrix)), replace=False)
    background = obs_matrix[bg_idx]
    explainer = shap.KernelExplainer(model_fn, background)
    
    sample_idx = np.random.choice(len(obs_matrix), min(n_samples, len(obs_matrix)), replace=False)
    shap_values = explainer.shap_values(obs_matrix[sample_idx], nsamples="auto")
    
    feature_names = FEATURE_COLUMNS + list(EXTRA_INDEX.keys()) + [f"asset_{i}" for i in range(5)]
    if len(feature_names) < obs_matrix.shape[1]:
        feature_names += [f"extra_{i}" for i in range(len(feature_names), obs_matrix.shape[1])]
    feature_names = feature_names[:obs_matrix.shape[1]]
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
    
    return shap_df, shap_values, obs_matrix[sample_idx], feature_names


def main():
    print("Loading actor...")
    actor, profile, obs_dim = load_actor(PROFILE_NAME)
    
    print("Loading SOXL data...")
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    full_df = flatten_feature_frame(segment_frame(dataset, "test"))
    price_path = build_price_path(full_df)
    
    print("Computing feature IC...")
    ic_df = compute_feature_ic(full_df)
    print(ic_df.head(15).to_string(index=False))
    
    print("\nRunning SHAP analysis (this may take 2-3 minutes)...")
    shap_df, shap_values, samples, feature_names = shap_analysis(actor, full_df, price_path, n_background=100, n_samples=300)
    print("\nTop 15 features by mean |SHAP|:")
    print(shap_df.head(15).to_string(index=False))
    
    out_dir = ROOT / "runtime" / "feature_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(out_dir / "ic.csv", index=False)
    shap_df.to_csv(out_dir / "shap_importance.csv", index=False)
    
    summary = {
        "top_ic": ic_df.head(10).to_dict(orient="records"),
        "top_shap": shap_df.head(10).to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
