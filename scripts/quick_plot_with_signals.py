"""Quickly regenerate the backtest plot with trade signals using the current best model."""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import build_dataset, segment_frame, init_qlib
from src.soxl_quant.rl import build_policy
from src.soxl_quant.backtest import backtest_profile, save_results
from src.soxl_quant.settings import load_profile, TEST_SEG, QLIB_DATA_PATH

init_qlib(QLIB_DATA_PATH)
dataset = build_dataset()
test_df = segment_frame(dataset, "test")
price_series = dataset.handler.price_frame.droplevel("instrument")["price"].loc[TEST_SEG[0] : TEST_SEG[1]]

profile = load_profile("balanced")
ckpt = torch.load(ROOT / "artifacts" / "models" / "soxl_balanced_moe_actor.pt", map_location="cpu", weights_only=False)
policy, actor = build_policy(ckpt["obs_dim"], profile, use_moe=True)
actor.load_state_dict(ckpt["state_dict"])
actor.eval()

result = backtest_profile(
    actor=actor,
    feature_df=test_df,
    price_series=price_series,
    position_cap_config=profile.position_cap,
    symbol="SOXL",
    asset_idx=0,
    test_seg=TEST_SEG,
)
result["slices"] = {}

save_results({"balanced": result})
print(f"Plot saved to {ROOT / 'soxl_qlib_strategy_v2.png'}")
print(f"return={result['return']:.1f}% sharpe={result['sharpe']:.2f} maxdd={result['max_dd']:.1f}% trades={result['trades']}")
