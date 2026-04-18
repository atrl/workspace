from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from qlib.rl.trainer.api import train as qlib_train

from ..backtest import backtest_profile
from ..data import (
    init_qlib,
    refresh_qlib_binary,
    segment_frame,
)
from ..policy.evaluation import evaluate_split
from ..policy.environment import PositionAction, SoxlSimulator, SoxlState, build_initial_states
from ..policy.observation import observation_dim_for_profile
from ..policy.policy_builder import build_policy
from ..policy.positioning import action_levels_for_profile
from ..policy.reward import ContinuousReward
from ..settings import (
    CSV_PATH,
    ETF_UNIVERSE,
    QLIB_DATA_PATH,
    ROOT,
)
from .artifacts import artifact_tag_suffix, persist_actor_bundle, persist_summary_bundle
from .evaluate import slice_metrics, validation_score
from .promote import build_promotion_report

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


def _build_multi_asset_sources(multi_asset_dir: Path, verbose: bool = True):
    multi_asset_dir.mkdir(parents=True, exist_ok=True)
    multi_asset_sources = []
    for asset_idx, (etf, meta) in enumerate(ETF_UNIVERSE.items()):
        underlying = meta["underlying"]
        leverage = meta["leverage"]
        csv_path = ROOT / "data" / "generalization" / f"{etf.lower()}_with_underlying_vix.csv"
        if not csv_path.exists():
            if verbose:
                print(f"[warn] missing {csv_path}, skipping {etf}")
            continue
        qlib_dir = multi_asset_dir / f"qlib_{etf.lower()}"
        refresh_qlib_binary(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            leveraged_etf=etf,
            underlying_etf=underlying,
            leverage_ratio=leverage,
        )
        init_qlib(qlib_dir)
        ds = build_dataset(
            leveraged_etf=etf,
            underlying_etf=underlying,
            leverage_ratio=leverage,
        )
        etf_train_df = segment_frame(ds, "train")
        pkl_path = multi_asset_dir / f"{etf}_train.pkl"
        etf_train_df.to_pickle(pkl_path)
        multi_asset_sources.append((pkl_path, etf_train_df, asset_idx))
        if verbose:
            print(f"[multi-asset] {etf} train={len(etf_train_df)} rows -> {pkl_path.name}")
    if not multi_asset_sources:
        raise RuntimeError("no multi-asset sources available")
    if verbose:
        print(f"[multi-asset] total pooled episodes will be generated from {len(multi_asset_sources)} ETFs")
    init_qlib(QLIB_DATA_PATH)
    return multi_asset_sources


def _build_action_interpreter(profile):
    return PositionAction(
        n_actions=len(action_levels_for_profile(profile)) if not profile.model.use_hybrid_overlay else len(profile.model.delta_levels),
        position_cap_config=profile.position_cap,
        action_levels=action_levels_for_profile(profile),
        hard_floor=profile.model.hard_floor,
        use_hybrid_overlay=profile.model.use_hybrid_overlay,
        delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
    )


def train_with_selection(policy, actor, initial_states, obs_dim: int, profile, valid_df, smoke: bool) -> Dict[str, object]:
    if not smoke and not profile.selection_score.staged_training:
        baseline_metrics = evaluate_split(
            actor,
            valid_df,
            position_cap_config=profile.position_cap,
            execution_config=profile.execution,
            stress_threshold=profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
        )
        baseline_score = validation_score(baseline_metrics, profile)
        baseline_state = copy.deepcopy(actor.state_dict())
        qlib_train(
            simulator_fn=SoxlSimulator,
            state_interpreter=SoxlState(obs_dim),
            action_interpreter=_build_action_interpreter(profile),
            initial_states=initial_states,
            policy=policy,
            reward=ContinuousReward(profile),
            vessel_kwargs={
                "episode_per_iter": profile.episode_per_iter,
                "update_kwargs": {"batch_size": profile.update_batch_size, "repeat": profile.update_repeat},
            },
            trainer_kwargs={"finite_env_type": "dummy", "concurrency": 1, "max_iters": profile.max_iters},
        )
        final_metrics = evaluate_split(
            actor,
            valid_df,
            position_cap_config=profile.position_cap,
            execution_config=profile.execution,
            stress_threshold=profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
        )
        final_score = validation_score(final_metrics, profile)
        history = [
            {"iter": 0, "score": baseline_score, **baseline_metrics},
            {"iter": profile.max_iters, "score": final_score, **final_metrics},
        ]
        print(
            f"[valid:{profile.name}@{profile.max_iters}] return={final_metrics['return']:.1f}% "
            f"sharpe={final_metrics['sharpe']:.2f} maxdd={final_metrics['max_dd']:.1f}% score={final_score:.2f}"
        )
        if baseline_score > final_score:
            actor.load_state_dict(baseline_state)
            return {"best_metrics": baseline_metrics, "best_score": baseline_score, "history": history}
        return {"best_metrics": final_metrics, "best_score": final_score, "history": history}

    stage_iters = profile.max_iters if smoke else min(profile.selection_score.stage_iters, profile.max_iters)
    completed = 0
    best_state = copy.deepcopy(actor.state_dict())
    best_metrics = evaluate_split(
        actor,
        valid_df,
        position_cap_config=profile.position_cap,
        execution_config=profile.execution,
        asset_idx=0,
        stress_threshold=profile.selection_score.stress_threshold,
        action_levels=action_levels_for_profile(profile),
        observation_window=profile.model.observation_window,
        hard_floor=profile.model.hard_floor,
    )
    best_score = validation_score(best_metrics, profile)
    history = [{"iter": 0, "score": best_score, **best_metrics}]

    while completed < profile.max_iters:
        current_iters = min(stage_iters, profile.max_iters - completed)
        qlib_train(
            simulator_fn=SoxlSimulator,
            state_interpreter=SoxlState(obs_dim),
            action_interpreter=_build_action_interpreter(profile),
            initial_states=initial_states,
            policy=policy,
            reward=ContinuousReward(profile),
            vessel_kwargs={
                "episode_per_iter": profile.episode_per_iter,
                "update_kwargs": {"batch_size": profile.update_batch_size, "repeat": profile.update_repeat},
            },
            trainer_kwargs={"finite_env_type": "dummy", "concurrency": 1, "max_iters": current_iters},
        )
        completed += current_iters
        metrics = evaluate_split(
            actor,
            valid_df,
            position_cap_config=profile.position_cap,
            execution_config=profile.execution,
            asset_idx=0,
            stress_threshold=profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
        )
        score = validation_score(metrics, profile)
        history.append({"iter": completed, "score": score, **metrics})
        print(
            f"[valid:{profile.name}@{completed}] return={metrics['return']:.1f}% "
            f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_dd']:.1f}% score={score:.2f}"
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = copy.deepcopy(actor.state_dict())

    actor.load_state_dict(best_state)
    return {"best_metrics": best_metrics, "best_score": best_score, "history": history}


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
    "_build_multi_asset_sources",
    "run_profile",
    "set_global_seed",
    "train_with_selection",
]
