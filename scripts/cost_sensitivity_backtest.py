#!/usr/bin/env python3
"""Run cost-sensitivity backtests for a saved actor bundle.

This is a post-train realism check:
- keep the actor fixed
- vary execution costs
- measure whether the strategy only looks good under optimistic friction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, init_qlib, segment_frame
from src.soxl_quant.runtime import bundle_position_cap_config, load_actor_bundle
from src.soxl_quant.settings import EXTRA_OBS_DIM, N_ASSETS, QLIB_DATA_PATH, TEST_SEG
from src.soxl_quant.data import FEATURE_COLUMNS


DEFAULT_SCENARIOS = (
    ("base_10bps", 0.0010, 0.0010, 5.0),
    ("stress_15bps", 0.0015, 0.0015, 5.0),
    ("stress_20bps", 0.0020, 0.0020, 5.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost sensitivity backtests for a saved SOXL actor bundle.")
    parser.add_argument("--profile", required=True, help="Profile name used to build the actor.")
    parser.add_argument("--model", required=True, help="Path to *_actor.pt bundle.")
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    test_df = segment_frame(dataset, "test")
    price_series = dataset.handler.price_frame.droplevel("instrument")["price"].loc[TEST_SEG[0] : TEST_SEG[1]]

    actor, bundle_path, bundle = load_actor_bundle(args.profile, model_path=model_path)
    expected_obs_dim = len(FEATURE_COLUMNS) + EXTRA_OBS_DIM + N_ASSETS
    model_obs_dim = int(bundle.get("obs_dim", -1))
    if model_obs_dim != expected_obs_dim:
        payload = {
            "status": "failed",
            "reason": "obs_dim_mismatch",
            "profile": args.profile,
            "model": str(bundle_path),
            "model_obs_dim": model_obs_dim,
            "expected_obs_dim": expected_obs_dim,
            "message": "saved actor bundle is not replay-compatible with the current feature set",
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[cost-sensitivity] failed: obs_dim mismatch model={model_obs_dim} expected={expected_obs_dim}")
        print(f"[done] wrote {output_path}")
        return
    position_cap = bundle_position_cap_config(bundle, args.profile)

    scenarios = []
    for name, open_cost, close_cost, min_cost in DEFAULT_SCENARIOS:
        result = backtest_profile(
            actor=actor,
            feature_df=test_df,
            price_series=price_series,
            position_cap_config=position_cap,
            symbol="SOXL",
            asset_idx=0,
            test_seg=TEST_SEG,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
        scenarios.append(
            {
                "name": name,
                "open_cost": open_cost,
                "close_cost": close_cost,
                "min_cost": min_cost,
                "return": float(result["return"]),
                "sharpe": float(result["sharpe"]),
                "max_dd": float(result["max_dd"]),
                "trades": int(result["trades"]),
            }
        )

    base = scenarios[0]
    payload = {
        "profile": args.profile,
        "model": str(bundle_path),
        "scenarios": scenarios,
        "deltas_vs_base": [
            {
                "name": row["name"],
                "return_delta": row["return"] - base["return"],
                "sharpe_delta": row["sharpe"] - base["sharpe"],
                "max_dd_delta": row["max_dd"] - base["max_dd"],
            }
            for row in scenarios[1:]
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[cost-sensitivity] profile={args.profile} model={bundle_path.name}")
    for row in scenarios:
        print(
            f"  - {row['name']}: return={row['return']:.1f}% sharpe={row['sharpe']:.3f} "
            f"maxdd={row['max_dd']:.2f}% trades={row['trades']}"
        )
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
