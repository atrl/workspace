from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from qlib.rl.trainer.api import train as qlib_train

from ..backtest import backtest_profile
from ..compat import patch_gym_compat
from ..data import (
    FEATURE_COLUMNS,
    build_dataset,
    feature_schema_signature,
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
    EXTRA_OBS_DIM,
    MODEL_DIR,
    N_ASSETS,
    QLIB_DATA_PATH,
    ROOT,
    TEST_SEG,
    load_profile,
    model_artifact_path,
    summary_artifact_path,
)
from .evaluate import compact_result, describe_metrics, slice_frame, slice_metrics, validation_score
from .promote import build_promotion_report

WalkForwardWindow = Tuple[str, Tuple[str, str], Tuple[str, str], Tuple[str, str]]

WALK_FORWARD_WINDOWS: Tuple[WalkForwardWindow, ...] = (
    ("wf_2020", ("2010-03-11", "2018-12-31"), ("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31")),
    ("wf_2022", ("2010-03-11", "2020-12-31"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")),
    ("wf_2023_2024", ("2010-03-11", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")),
)

WORKER_RESULT_DIR = ROOT / "runtime" / "soxl" / "wf_workers"


def serialize_position_cap(profile) -> Dict[str, float]:
    return {
        "start_drawdown": profile.position_cap.start_drawdown,
        "full_drawdown": profile.position_cap.full_drawdown,
        "min_cap": profile.position_cap.min_cap,
        "curve_power": profile.position_cap.curve_power,
        "bull_min_cap": profile.position_cap.bull_min_cap,
        "bull_vol_threshold": profile.position_cap.bull_vol_threshold,
        "bull_anchor_threshold": profile.position_cap.bull_anchor_threshold,
        "bear_min_cap": profile.position_cap.bear_min_cap,
        "bear_regime_threshold": profile.position_cap.bear_regime_threshold,
    }


def serialize_selection_score(profile) -> Dict[str, float | bool]:
    return {
        "return_weight": profile.selection_score.return_weight,
        "sharpe_weight": profile.selection_score.sharpe_weight,
        "max_dd_penalty": profile.selection_score.max_dd_penalty,
        "avg_pos_penalty": profile.selection_score.avg_pos_penalty,
        "stress_avg_pos_penalty": profile.selection_score.stress_avg_pos_penalty,
        "stress_threshold": profile.selection_score.stress_threshold,
        "staged_training": profile.selection_score.staged_training,
        "stage_iters": profile.selection_score.stage_iters,
    }


def serialize_model_config(profile) -> Dict[str, object]:
    return {
        "observation_window": int(profile.model.observation_window),
        "action_levels": [float(v) for v in profile.model.action_levels],
        "use_regime_router": bool(profile.model.use_regime_router),
        "hard_regime_routing": bool(profile.model.hard_regime_routing),
        "router_experts": int(profile.model.router_experts),
        "router_top_k": int(profile.model.router_top_k),
        "router_noise_std": float(profile.model.router_noise_std),
    }


def serialize_sampling_config(profile) -> Dict[str, object]:
    return {
        "regime_stratified": bool(profile.sampling.regime_stratified),
        "phase_balanced": bool(profile.sampling.phase_balanced),
        "mild_risk_threshold": float(profile.sampling.mild_risk_threshold),
        "severe_risk_threshold": float(profile.sampling.severe_risk_threshold),
        "mild_drawdown_threshold": float(profile.sampling.mild_drawdown_threshold),
        "severe_drawdown_threshold": float(profile.sampling.severe_drawdown_threshold),
        "mild_repeat": int(profile.sampling.mild_repeat),
        "severe_repeat": int(profile.sampling.severe_repeat),
        "persistent_stress_days": float(profile.sampling.persistent_stress_days),
        "trend_repeat": int(profile.sampling.trend_repeat),
        "rebound_repeat": int(profile.sampling.rebound_repeat),
        "episode_stride": int(profile.sampling.episode_stride),
        "bull_anchor_threshold": float(profile.sampling.bull_anchor_threshold),
        "bull_trend_threshold": float(profile.sampling.bull_trend_threshold),
        "bull_vol_threshold": float(profile.sampling.bull_vol_threshold),
        "bull_risk_threshold": float(profile.sampling.bull_risk_threshold),
        "rebound_anchor_ceiling": float(profile.sampling.rebound_anchor_ceiling),
        "rebound_trend_floor": float(profile.sampling.rebound_trend_floor),
        "rebound_risk_ceiling": float(profile.sampling.rebound_risk_ceiling),
        "rebound_risk_release": float(profile.sampling.rebound_risk_release),
    }


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

    moe_tag = "moe_novix" if (use_moe and ablate_vix) else ("moe" if use_moe else "")
    effective_tag = f"{artifact_tag}_{moe_tag}" if (artifact_tag and moe_tag) else (artifact_tag or moe_tag)

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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_artifact_path(profile.name, effective_tag)
    summary_path = summary_artifact_path(profile.name, effective_tag)
    torch.save(
        {
            "profile": profile.name,
            "seed": seed,
            "hidden_sizes": list(profile.hidden_sizes),
            "obs_dim": obs_dim,
            "n_actions": int(getattr(actor, "output_dim", 0) or 11),
            "feature_schema": {
                "signature": feature_schema_signature(),
                "feature_count": len(FEATURE_COLUMNS),
                "extra_obs_dim": EXTRA_OBS_DIM,
                "n_assets": N_ASSETS,
            },
            "position_cap": serialize_position_cap(profile),
            "selection_score": serialize_selection_score(profile),
            "model_config": serialize_model_config(profile),
            "sampling_config": serialize_sampling_config(profile),
            "state_dict": actor.state_dict(),
        },
        model_path,
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
        execution_config=profile.execution,
        asset_idx=0,
        action_levels=action_levels_for_profile(profile),
        observation_window=profile.model.observation_window,
        hard_floor=profile.model.hard_floor,
    )
    promotion_report = build_promotion_report(profile, slice_result, test_proxy)
    summary_path.write_text(
        json.dumps(
            {
                "profile": profile.name,
                "seed": seed,
                "feature_schema": {
                    "signature": feature_schema_signature(),
                    "feature_count": len(FEATURE_COLUMNS),
                    "extra_obs_dim": EXTRA_OBS_DIM,
                    "n_assets": N_ASSETS,
                },
                "position_cap": serialize_position_cap(profile),
                "selection_score": serialize_selection_score(profile),
                "model_config": serialize_model_config(profile),
                "sampling_config": serialize_sampling_config(profile),
                "model_path": str(model_path),
                "valid": valid_metrics,
                "best_score": selection["best_score"],
                "selection_history": selection["history"],
                "slices": slice_result,
                "test_proxy": test_proxy,
                "promotion": promotion_report,
            },
            indent=2,
        ),
        encoding="utf-8",
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


def run_walk_forward_isolated(
    profile,
    smoke: bool,
    seed: int,
    skip_data_refresh: bool,
    multi_asset: bool = False,
    use_moe: bool = False,
    ablate_vix: bool = False,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    WORKER_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name, train_seg, valid_seg, test_seg in WALK_FORWARD_WINDOWS:
        print(
            f"\n[walk-forward:{profile.name}:{name}:isolated] "
            f"train={train_seg[0]}->{train_seg[1]} valid={valid_seg[0]}->{valid_seg[1]} test={test_seg[0]}->{test_seg[1]}"
        )
        worker_output = WORKER_RESULT_DIR / f"{profile.name}-{name}-seed{seed}.json"
        if worker_output.exists():
            worker_output.unlink()
        cmd = [
            sys.executable,
            str(ROOT / "soxl_qlib_strategy_v2.py"),
            "--profiles",
            profile.name,
            "--seed",
            str(seed),
            "--worker-run-profile",
            profile.name,
            "--worker-window-name",
            name,
            "--worker-train-start",
            train_seg[0],
            "--worker-train-end",
            train_seg[1],
            "--worker-valid-start",
            valid_seg[0],
            "--worker-valid-end",
            valid_seg[1],
            "--worker-test-start",
            test_seg[0],
            "--worker-test-end",
            test_seg[1],
            "--worker-output",
            str(worker_output),
        ]
        if smoke:
            cmd.append("--smoke")
        if skip_data_refresh:
            cmd.append("--skip-data-refresh")
        if multi_asset:
            cmd.append("--multi-asset")
        if use_moe:
            cmd.append("--moe")
        if ablate_vix:
            cmd.append("--ablate-vix")
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if not worker_output.exists():
            raise RuntimeError(f"walk-forward worker failed before writing result: {name} exit={proc.returncode}")
        worker_payload = json.loads(worker_output.read_text(encoding="utf-8"))
        results[name] = worker_payload
        print(describe_metrics(f"walk-forward:{profile.name}:{name}", worker_payload))
        if smoke:
            break
    return results


def run_worker_profile(args: argparse.Namespace) -> None:
    patch_gym_compat()
    set_global_seed(args.seed)
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(ablate_vix=args.ablate_vix)
    full_df = pd.concat(
        [
            segment_frame(dataset, "train"),
            segment_frame(dataset, "valid"),
            segment_frame(dataset, "test"),
        ]
    ).sort_index()
    price_frame = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()
    profile = load_profile(args.worker_run_profile)
    if args.smoke or args.worker_run_profile == "smoke":
        profile = load_profile("smoke")
    multi_asset_sources = None
    if args.multi_asset:
        multi_asset_sources = _build_multi_asset_sources(ROOT / "data" / "multi_asset", verbose=False)

    result = run_profile(
        profile=profile,
        train_df=slice_frame(full_df, (args.worker_train_start, args.worker_train_end)),
        valid_df=slice_frame(full_df, (args.worker_valid_start, args.worker_valid_end)),
        test_df=slice_frame(full_df, (args.worker_test_start, args.worker_test_end)),
        price_series=price_frame.loc[args.worker_test_start : args.worker_test_end],
        smoke=args.smoke,
        seed=args.seed,
        artifact_tag=args.worker_window_name,
        multi_asset_sources=multi_asset_sources,
        use_moe=args.moe,
        ablate_vix=args.ablate_vix,
    )
    output_path = Path(args.worker_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compact_result(result), indent=2), encoding="utf-8")
    os._exit(0)


__all__ = [
    "WALK_FORWARD_WINDOWS",
    "WORKER_RESULT_DIR",
    "_build_multi_asset_sources",
    "run_profile",
    "run_walk_forward_isolated",
    "run_worker_profile",
    "set_global_seed",
    "train_with_selection",
]
