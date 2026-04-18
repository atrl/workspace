from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import torch

from ..backtest import backtest_profile
from ..data import init_qlib, refresh_qlib_binary
from ..policy.evaluation import evaluate_split
from ..policy.environment import build_initial_states
from ..policy.observation import observation_dim_for_profile
from ..policy.policy_builder import build_policy
from ..policy.positioning import action_levels_for_profile
from ..settings import CSV_PATH, QLIB_DATA_PATH, ROOT
from .artifacts import artifact_tag_suffix, persist_actor_bundle, persist_summary_bundle
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
) -> Dict[str, object]:
    obs_dim = observation_dim_for_profile(profile)
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
    selection = train_with_selection(
        policy=policy,
        actor=actor,
        initial_states=initial_states,
        obs_dim=obs_dim,
        profile=profile,
        valid_df=valid_df,
        smoke=smoke,
    )
    valid_metrics = selection["best_metrics"]
    print(
        f"[valid:{profile.name}] return={valid_metrics['return']:.1f}% "
        f"sharpe={valid_metrics['sharpe']:.2f} maxdd={valid_metrics['max_dd']:.1f}% "
        f"best_score={selection['best_score']:.2f}"
    )

    model_path = persist_actor_bundle(profile, actor, obs_dim=obs_dim, seed=seed, effective_tag=effective_tag)
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
        execution_config=profile.execution,
        asset_idx=0,
        action_levels=action_levels_for_profile(profile),
        observation_window=profile.model.observation_window,
        hard_floor=profile.model.hard_floor,
    )
    promotion_report = build_promotion_report(profile, slice_result, test_proxy)
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
    )
    bt_result = backtest_profile(actor, test_df, price_series, position_cap_config=profile.position_cap, asset_idx=0)
    bt_result["valid"] = valid_metrics
    bt_result["selection_history"] = selection["history"]
    bt_result["model_path"] = str(model_path)
    bt_result["summary_path"] = str(summary_path)
    bt_result["slices"] = slice_result
    bt_result["test_proxy"] = test_proxy
    bt_result["promotion"] = promotion_report
    return bt_result

__all__ = [
    "build_multi_asset_sources",
    "run_profile",
    "set_global_seed",
    "train_with_selection",
]
