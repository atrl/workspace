#!/usr/bin/env python3
"""Export daily SOXL backtest rows for a saved actor/profile pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.app.dashboard import write_review_daily_csv
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import MODEL_DIR, QLIB_DATA_PATH, TEST_SEG, load_profile, model_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily SOXL backtest rows for a saved profile actor.")
    parser.add_argument("--profile", required=True, help="Profile name, e.g. balanced or balanced_transition_state")
    parser.add_argument("--tag", default="", help="Artifact tag, e.g. wf_2020")
    parser.add_argument("--model", type=Path, default=None, help="Explicit actor bundle path")
    parser.add_argument("--output", required=True, help="CSV path to write")
    parser.add_argument("--test-start", default="", help="Override test start date (for walk-forward windows)")
    parser.add_argument("--test-end", default="", help="Override test end date (for walk-forward windows)")
    return parser.parse_args()


def resolve_model_path(
    profile_name: str,
    tag: str = "",
    *,
    explicit_model: Path | None = None,
    model_dir: Path = MODEL_DIR,
) -> Path:
    if explicit_model is not None:
        path = explicit_model
        if not path.exists():
            raise FileNotFoundError(f"actor checkpoint not found: {path}")
        return path

    primary = model_artifact_path(profile_name, tag)
    if primary.exists():
        return primary

    cleaned_tag = tag.strip()
    if cleaned_tag:
        walk_forward_path = (
            model_dir
            / f"{profile_name}_{cleaned_tag}"
            / f"soxl_{profile_name}_{cleaned_tag}_actor.pt"
        )
        if walk_forward_path.exists():
            return walk_forward_path

    raise FileNotFoundError(f"actor checkpoint not found for profile={profile_name} tag={tag!r}: {primary}")


def load_profile_backtest_inputs(profile) -> dict[str, object]:
    feature_packs = tuple(profile.feature.packs)
    dataset = build_dataset(feature_packs=feature_packs, profile_name=profile.name)
    test_df = segment_frame(dataset, "test")
    price = None
    price_frame = getattr(dataset.handler, "price_frame", None)
    if price_frame is not None:
        price = price_frame.droplevel("instrument")["price"].loc[test_df.index]
    return {
        "dataset": dataset,
        "test_df": test_df,
        "price": price,
        "feature_columns": feature_columns_for_packs(feature_packs, profile.name),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(args.profile)
    inputs = load_profile_backtest_inputs(profile)
    test_df = inputs["test_df"]
    price = inputs["price"]
    feature_columns = inputs["feature_columns"]

    if args.test_start and args.test_end:
        test_df = test_df.loc[args.test_start:args.test_end]
        price = price.loc[test_df.index]
        test_seg = (args.test_start, args.test_end)
    else:
        test_seg = TEST_SEG

    model_path = resolve_model_path(args.profile, args.tag, explicit_model=args.model)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    result = backtest_profile(
        actor=actor,
        feature_df=test_df,
        price_series=price,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        execution_config=profile.execution,
        symbol="SOXL",
        test_seg=test_seg,
        model_config=profile.model,
        mode_router_config=profile.mode_router,
        feature_columns=feature_columns,
    )
    write_review_daily_csv(
        output,
        result["portfolio"],
        price,
        result["position_history"],
        result.get("trade_signals", []),
    )
    print(output)


if __name__ == "__main__":
    main()
