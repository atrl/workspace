#!/usr/bin/env python3
from __future__ import annotations

"""No-promotion oracle distillation from a replay daily path."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

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
    build_budget_usage_oracle_dataset_from_frame,
    build_policy,
    supervised_pretrain,
)
from src.soxl_quant.policy.positioning import action_levels_for_profile
from src.soxl_quant.settings import CSV_PATH, QLIB_DATA_PATH


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="risk_budget_oracle_distill_v1")
    parser.add_argument("--replay-daily", type=Path, default=Path("runtime/post_train_review/risk_budget_usage_v1_soxl_daily.csv"))
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use every overlapping replay row")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--artifact-tag", default="oracle")
    parser.add_argument("--output-json", type=Path, default=Path("runtime/post_train_review/risk_budget_oracle_distill_v1.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runtime/post_train_review/risk_budget_oracle_distill_v1.md"))
    parser.add_argument("--skip-data-refresh", action="store_true")
    return parser.parse_args()


def _label_distribution(actions: np.ndarray, n_actions: int) -> dict[str, int]:
    counts = Counter(int(v) for v in actions.tolist())
    return {str(key): int(counts.get(key, 0)) for key in range(int(n_actions))}


def _fmt(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['return']:.1f}% / Sharpe {metrics['sharpe']:.3f} / "
        f"MaxDD {metrics['max_dd']:.1f}% / avg_pos {metrics.get('avg_pos', 0.0):.3f}"
    )


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    slices = payload["slices"]
    assert isinstance(slices, dict)
    lines = [
        f"# {payload['profile']} Oracle Distillation",
        "",
        "No-promotion diagnostic trained from a replay daily path.",
        "",
        "This actor is an upper-bound capacity probe only. It uses replay labels from held-out dates and must not be promoted.",
        "",
        "## Dataset",
        "",
        f"- replay_daily: `{payload['replay_daily']}`",
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
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(payload["decision"]),
            "",
            f"Actor bundle: `{payload['model_path']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    profile = load_profile(args.profile)
    replay_daily = pd.read_csv(args.replay_daily, parse_dates=["datetime"])

    if not args.skip_data_refresh:
        refresh_qlib_binary(CSV_PATH, QLIB_DATA_PATH)
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    test_df = segment_frame(dataset, "test")

    obs_array, action_array = build_budget_usage_oracle_dataset_from_frame(
        test_df,
        replay_daily=replay_daily,
        profile=profile,
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
    n_actions = len(profile.model.delta_levels) if profile.model.use_hybrid_overlay else len(action_levels_for_profile(profile))
    payload: dict[str, object] = {
        "profile": profile.name,
        "replay_daily": str(args.replay_daily),
        "seed": int(args.seed),
        "samples": int(len(obs_array)),
        "label_distribution": _label_distribution(action_array, n_actions),
        "bc_stats": bc_stats,
        "model_path": _display_path(Path(model_path)),
        "slices": slices,
        "test_proxy": test_proxy,
        "decision": (
            "no-promotion diagnostic only; labels come from held-out replay daily path"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[oracle:{profile.name}] samples={len(obs_array)} stats={bc_stats} model={model_path}")
    print(f"[oracle:{profile.name}] test_proxy={_fmt(test_proxy)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
