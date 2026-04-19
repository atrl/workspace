from __future__ import annotations

import copy
from typing import Dict

import numpy as np
from qlib.rl.trainer.api import train as qlib_train

from ..policy.evaluation import evaluate_split
from ..policy.environment import PositionAction, SoxlSimulator, SoxlState
from ..policy.positioning import action_levels_for_profile
from ..policy.reward import ContinuousReward
from .evaluate import validation_score


def build_action_interpreter(profile):
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
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
        )
        baseline_score = validation_score(baseline_metrics, profile)
        baseline_state = copy.deepcopy(actor.state_dict())
        qlib_train(
            simulator_fn=SoxlSimulator,
            state_interpreter=SoxlState(obs_dim),
            action_interpreter=build_action_interpreter(profile),
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
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
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
        use_hybrid_overlay=profile.model.use_hybrid_overlay,
        delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
    )
    best_score = validation_score(best_metrics, profile)
    history = [{"iter": 0, "score": best_score, **best_metrics}]

    while completed < profile.max_iters:
        current_iters = min(stage_iters, profile.max_iters - completed)
        qlib_train(
            simulator_fn=SoxlSimulator,
            state_interpreter=SoxlState(obs_dim),
            action_interpreter=build_action_interpreter(profile),
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
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
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


__all__ = ["build_action_interpreter", "train_with_selection"]
