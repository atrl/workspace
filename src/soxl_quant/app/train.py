from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ..backtest import backtest_profile
from ..data import feature_columns_for_packs, init_qlib, refresh_qlib_binary
from ..policy.evaluation import evaluate_split
from ..policy.environment import build_initial_states
from ..policy.observation import observation_dim_for_profile
from ..policy.policy_builder import (
    build_policy,
    build_warm_start_dataset,
    supervised_pretrain,
)
from ..policy.positioning import action_levels_for_profile
from ..settings import CSV_PATH, QLIB_DATA_PATH, ROOT
from .artifacts import (
    artifact_tag_suffix,
    checkpoint_retention_gate_payload,
    final_model_retention_payload,
    persist_actor_bundle,
    persist_retained_checkpoint_bundles,
    persist_summary_bundle,
)
from .dashboard import export_review_daily_csv
from .evaluate import slice_metrics
from .promote import build_promotion_report
from .selection import train_with_selection
from .sources import build_multi_asset_sources

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_profile(
    profile,
    train_df,
    valid_df,
    test_df,
    price_series,
    smoke: bool,
    seed: int,
    artifact_tag: str = "",
    multi_asset_sources=None,
    use_moe: bool = False,
    ablate_vix: bool = False,
    retention_gate_evidence: dict | None = None,
) -> Dict[str, object]:
    obs_dim = observation_dim_for_profile(profile)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    if multi_asset_sources:
        initial_states = build_initial_states(multi_asset_sources, profile, smoke)
    else:
        pkl_path = ROOT / f"data/soxl_profile_{profile.name}.pkl"
        train_df.to_pickle(pkl_path)
        initial_states = build_initial_states([(pkl_path, train_df, 0)], profile, smoke)
    if not initial_states:
        raise RuntimeError(f"no initial states generated for profile={profile.name}")

    effective_tag = artifact_tag_suffix(artifact_tag, use_moe=use_moe, ablate_vix=ablate_vix)

    set_global_seed(seed)
    print(
        f"\n[train:{profile.name}{'_moe' if use_moe else ''}] seed={seed} episodes={len(initial_states)} "
        f"iters={profile.max_iters} epi/iter={profile.episode_per_iter}"
    )

    policy, actor = build_policy(obs_dim, profile, use_moe=use_moe, hard_regime_routing=use_moe)

    # Load from existing checkpoint if specified (warm-start from trained model)
    if profile.checkpoint_path and not smoke:
        ckpt = Path(profile.checkpoint_path)
        if ckpt.exists():
            print(f"[checkpoint:{profile.name}] loading from {ckpt}")
            bundle = torch.load(ckpt, map_location="cpu")
            actor.load_state_dict(bundle["state_dict"])
            print(f"[checkpoint:{profile.name}] loaded successfully")
        else:
            print(f"[checkpoint:{profile.name}] WARNING: {ckpt} not found, using fresh init")

    if profile.warm_start.enabled and not smoke:
        print(
            f"[warm-start:{profile.name}] sampling {profile.warm_start.sample_size} transitions "
            f"for {profile.warm_start.steps} BC steps"
        )
        try:
            ws_obs, ws_actions = build_warm_start_dataset(
                initial_states,
                profile,
                sample_size=int(profile.warm_start.sample_size),
                rng_seed=int(seed),
            )
            ws_stats = supervised_pretrain(
                actor,
                ws_obs,
                ws_actions,
                steps=int(profile.warm_start.steps),
                lr=float(profile.warm_start.lr),
                batch_size=int(profile.warm_start.batch_size),
            )
            print(
                f"[warm-start:{profile.name}] final loss={ws_stats['loss']:.4f} "
                f"acc={ws_stats['accuracy']:.3f} steps={ws_stats['steps']}"
            )
        except Exception as exc:
            print(f"[warm-start:{profile.name}] skipped due to error: {exc!r}")

    selection = train_with_selection(
        policy=policy,
        actor=actor,
        initial_states=initial_states,
        obs_dim=obs_dim,
        profile=profile,
        valid_df=valid_df,
        smoke=smoke,
        test_df=test_df if profile.selection_score.use_pareto else None,
    )
    valid_metrics = selection["best_metrics"]
    print(
        f"[valid:{profile.name}] return={valid_metrics['return']:.1f}% "
        f"sharpe={valid_metrics['sharpe']:.2f} maxdd={valid_metrics['max_dd']:.1f}% "
        f"best_score={selection['best_score']:.2f}"
    )

    model_path = persist_actor_bundle(profile, actor, obs_dim=obs_dim, seed=seed, effective_tag=effective_tag)
    checkpoint_retention_decision = checkpoint_retention_gate_payload(
        profile,
        selection.get("retained_checkpoints", []),
        retention_gate_evidence,
    )
    retained_checkpoint_records = persist_retained_checkpoint_bundles(
        profile=profile,
        actor=actor,
        obs_dim=obs_dim,
        seed=seed,
        effective_tag=effective_tag,
        retained_checkpoints=selection.get("retained_checkpoints", []),
        retention_gate_evidence=retention_gate_evidence,
    )
    slice_result = slice_metrics(
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
    test_seg = (
        str(test_df.index.min().date()) if hasattr(test_df.index.min(), "date") else str(test_df.index.min()),
        str(test_df.index.max().date()) if hasattr(test_df.index.max(), "date") else str(test_df.index.max()),
    )
    bt_result = backtest_profile(
        actor,
        test_df,
        price_series,
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        asset_idx=0,
        feature_columns=feature_columns,
        model_config=profile.model,
        mode_router_config=profile.mode_router,
        test_seg=test_seg,
    )
    # CRITICAL FIX: Use main backtest metrics instead of evaluate_split metrics
    # evaluate_split has systematic gaps of 80-110pp vs backtest_profile
    if "return" in bt_result and "sharpe" in bt_result and "max_dd" in bt_result:
        test_proxy["return"] = bt_result["return"]
        test_proxy["sharpe"] = bt_result["sharpe"]
        test_proxy["max_dd"] = bt_result["max_dd"]
        # Propagate relative performance for walk-forward gate
        if "bnh_return" in bt_result:
            test_proxy["buy_and_hold_return"] = bt_result["bnh_return"]
        if "relative_outperformance" in bt_result:
            test_proxy["relative_return"] = bt_result["relative_outperformance"]
        # Also fix test_2022 slice metrics (used for fixed-split gate: max_dd check)
        if "test_2022" in slice_result:
            bt_2022 = backtest_profile(
                actor,
                test_df,
                price_series,
                position_cap_config=profile.position_cap,
                position_budget_config=profile.position_budget,
                asset_idx=0,
                feature_columns=feature_columns,
                model_config=profile.model,
                mode_router_config=profile.mode_router,
                test_seg=("2022-01-01", "2022-12-31"),
            )
            if "return" in bt_2022 and "sharpe" in bt_2022 and "max_dd" in bt_2022:
                slice_result["test_2022"]["return"] = bt_2022["return"]
                slice_result["test_2022"]["sharpe"] = bt_2022["sharpe"]
                slice_result["test_2022"]["max_dd"] = bt_2022["max_dd"]
        # Rebuild promotion report with corrected metrics
        promotion_report = build_promotion_report(profile, slice_result, test_proxy)
    else:
        promotion_report = build_promotion_report(profile, slice_result, test_proxy)
    final_model_retention_decision = final_model_retention_payload(
        profile=profile,
        model_path=model_path,
        promotion_report=promotion_report,
        retention_gate_evidence=retention_gate_evidence,
    )
    summary_path = persist_summary_bundle(
        profile=profile,
        seed=seed,
        effective_tag=effective_tag,
        model_path=model_path,
        valid_metrics=valid_metrics,
        selection=selection,
        slice_result=slice_result,
        test_proxy=test_proxy,
        promotion_report=promotion_report,
        retained_checkpoints=retained_checkpoint_records,
        checkpoint_retention_decision=checkpoint_retention_decision,
        final_model_retention_decision=final_model_retention_decision,
    )
    bt_result["valid"] = valid_metrics
    bt_result["selection_history"] = selection["history"]
    bt_result["retained_checkpoints"] = retained_checkpoint_records
    bt_result["checkpoint_retention_decision"] = checkpoint_retention_decision
    bt_result["final_model_retention_decision"] = final_model_retention_decision
    bt_result["model_path"] = str(model_path)
    bt_result["summary_path"] = str(summary_path)
    bt_result["slices"] = slice_result
    bt_result["test_proxy"] = test_proxy
    bt_result["promotion"] = promotion_report
    review_csv_path = export_review_daily_csv(profile.name, bt_result, price_series, effective_tag)
    if review_csv_path is not None:
        bt_result["review_daily_csv"] = str(review_csv_path)
    return bt_result

__all__ = [
    "build_multi_asset_sources",
    "run_profile",
    "set_global_seed",
    "train_with_selection",
]
