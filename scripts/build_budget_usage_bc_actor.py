#!/usr/bin/env python3
from __future__ import annotations

"""Pretrain a budget-usage actor from a retained fast expert actor."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.soxl_quant.app.artifacts import persist_actor_bundle
from src.soxl_quant.app.evaluate import slice_metrics
from src.soxl_quant.app.train import set_global_seed
from src.soxl_quant.config import load_profile
from src.soxl_quant.data import (
    build_dataset,
    feature_columns_for_packs,
    init_qlib,
    refresh_qlib_binary,
    segment_frame,
)
from src.soxl_quant.policy.evaluation import evaluate_split
from src.soxl_quant.policy.observation import observation_dim_for_profile
from src.soxl_quant.policy.policy_builder import (
    build_budget_usage_warm_start_dataset_from_frame,
    build_policy,
    supervised_pretrain,
)
from src.soxl_quant.policy.positioning import action_levels_for_profile
from src.soxl_quant.runtime import load_actor_bundle
from src.soxl_quant.settings import CSV_PATH, QLIB_DATA_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="risk_budget_bc_warmstart_v1")
    parser.add_argument("--expert-profile", default="balanced_rule_residual_imitation_v4")
    parser.add_argument("--expert-model", type=Path, default=Path("artifacts/models/soxl_balanced_rule_residual_imitation_v4_actor.pt"))
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--artifact-tag", default="bc")
    parser.add_argument("--output-json", type=Path, default=Path("runtime/post_train_review/risk_budget_bc_warmstart_v1_bc.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runtime/post_train_review/risk_budget_bc_warmstart_v1_bc.md"))
    parser.add_argument("--skip-data-refresh", action="store_true")
    return parser.parse_args()


def _label_distribution(actions: np.ndarray) -> dict[str, int]:
    counts = Counter(int(v) for v in actions.tolist())
    return {str(key): int(counts.get(key, 0)) for key in range(int(actions.max()) + 1)}


def _fmt(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['return']:.1f}% / Sharpe {metrics['sharpe']:.3f} / "
        f"MaxDD {metrics['max_dd']:.1f}% / avg_pos {metrics.get('avg_pos', 0.0):.3f}"
    )


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    slices = payload["slices"]
    assert isinstance(slices, dict)
    lines = [
        f"# {payload['profile']} BC pretrain",
        "",
        "Budget-usage behavior cloning from a retained fast expert actor on the train split only.",
        "",
        "## Dataset",
        "",
        f"- expert_profile: `{payload['expert_profile']}`",
        f"- expert_model: `{payload['expert_model']}`",
        f"- samples: `{payload['samples']}`",
        f"- label_distribution: `{payload['label_distribution']}`",
        "",
        "## BC Stats",
        "",
        f"- loss: `{payload['bc_stats']['loss']:.6f}`",
        f"- accuracy: `{payload['bc_stats']['accuracy']:.6f}`",
        f"- steps: `{payload['bc_stats']['steps']}`",
        "",
        "## Fixed-Split Probe",
        "",
        "| slice | return / sharpe / maxdd / avg_pos |",
        "| --- | --- |",
    ]
    for name in ("test_2020", "test_2022", "test_2023_2024"):
        lines.append(f"| {name} | {_fmt(slices[name])} |")
    lines.append(f"| test_proxy | {_fmt(payload['test_proxy'])} |")
    lines.extend(["", f"Actor bundle: `{payload['model_path']}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    profile = load_profile(args.profile)
    expert_profile = load_profile(args.expert_profile)

    if not args.skip_data_refresh:
        refresh_qlib_binary(CSV_PATH, QLIB_DATA_PATH)
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    train_df = segment_frame(dataset, "train")
    test_df = segment_frame(dataset, "test")

    expert_actor, expert_path, _expert_bundle = load_actor_bundle(args.expert_profile, args.expert_model)
    obs_array, action_array = build_budget_usage_warm_start_dataset_from_frame(
        train_df,
        profile=profile,
        expert_actor=expert_actor,
        expert_profile=expert_profile,
        sample_size=int(args.sample_size),
        rng_seed=int(args.seed),
    )

    obs_dim = observation_dim_for_profile(profile)
    _policy, actor = build_policy(obs_dim, profile)
    bc_stats = supervised_pretrain(
        actor,
        obs_array,
        action_array,
        steps=int(args.steps),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
    )
    model_path = persist_actor_bundle(
        profile,
        actor,
        obs_dim=obs_dim,
        seed=int(args.seed),
        effective_tag=args.artifact_tag,
    )

    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    slices = slice_metrics(
        actor,
        test_df,
        {
            "test_2020": ("2020-01-01", "2020-12-31"),
            "test_2022": ("2022-01-01", "2022-12-31"),
            "test_2023_2024": ("2023-01-01", "2024-12-31"),
        },
        profile,
    )
    test_proxy = evaluate_split(
        actor,
        test_df,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        execution_config=profile.execution,
        asset_idx=0,
        action_levels=action_levels_for_profile(profile),
        observation_window=profile.model.observation_window,
        hard_floor=profile.model.hard_floor,
        use_hybrid_overlay=profile.model.use_hybrid_overlay,
        delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
        feature_columns=feature_columns,
        mode_router_config=profile.mode_router,
        position_mapping=profile.model.position_mapping,
    )
    payload: dict[str, object] = {
        "profile": profile.name,
        "expert_profile": expert_profile.name,
        "expert_model": str(expert_path),
        "seed": int(args.seed),
        "samples": int(len(obs_array)),
        "label_distribution": _label_distribution(action_array),
        "bc_stats": bc_stats,
        "model_path": str(model_path),
        "slices": slices,
        "test_proxy": test_proxy,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[bc:{profile.name}] samples={len(obs_array)} stats={bc_stats} model={model_path}")
    print(f"[bc:{profile.name}] test_proxy={_fmt(test_proxy)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
