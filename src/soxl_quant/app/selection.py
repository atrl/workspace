from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import pandas as pd
from qlib.rl.trainer.api import train as qlib_train

from ..policy.evaluation import evaluate_split
from ..policy.environment import PositionAction, SoxlSimulator, SoxlState
from ..policy.positioning import action_levels_for_profile
from ..policy.reward import ContinuousReward
from .evaluate import (
    evaluate_all_slices,
    pareto_front_indices,
    pareto_objective_values,
    validation_score,
)


def build_action_interpreter(profile):
    return PositionAction(
        n_actions=len(action_levels_for_profile(profile)) if not profile.model.use_hybrid_overlay else len(profile.model.delta_levels),
        position_cap_config=profile.position_cap,
        position_budget_config=profile.position_budget,
        action_levels=action_levels_for_profile(profile),
        hard_floor=profile.model.hard_floor,
        use_hybrid_overlay=profile.model.use_hybrid_overlay,
        delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
        mode_router_config=profile.mode_router,
        position_mapping=profile.model.position_mapping,
    )


def train_with_selection(
    policy,
    actor,
    initial_states,
    obs_dim: int,
    profile,
    valid_df,
    smoke: bool,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    if not smoke and not profile.selection_score.staged_training:
        baseline_metrics = evaluate_split(
            actor,
            valid_df,
            position_cap_config=profile.position_cap,
            position_budget_config=profile.position_budget,
            execution_config=profile.execution,
            stress_threshold=profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
            mode_router_config=profile.mode_router,
            position_mapping=profile.model.position_mapping,
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
            position_budget_config=profile.position_budget,
            execution_config=profile.execution,
            stress_threshold=profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
            mode_router_config=profile.mode_router,
            position_mapping=profile.model.position_mapping,
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
    use_pareto = bool(profile.selection_score.use_pareto) and test_df is not None
    pareto_objectives = tuple(profile.selection_score.pareto_objectives)

    def _eval_valid():
        return evaluate_split(
            actor,
            valid_df,
            position_cap_config=profile.position_cap,
            position_budget_config=profile.position_budget,
            execution_config=profile.execution,
            asset_idx=0,
            stress_threshold=profile.selection_score.stress_threshold,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
            mode_router_config=profile.mode_router,
            position_mapping=profile.model.position_mapping,
        )

    baseline_metrics = _eval_valid()
    baseline_score = validation_score(baseline_metrics, profile)
    best_state = copy.deepcopy(actor.state_dict())
    best_metrics = baseline_metrics
    best_score = baseline_score
    history = [{"iter": 0, "score": baseline_score, **baseline_metrics}]

    # Pareto candidate ledger: each entry is (state_dict, valid_metrics, valid_score, objectives)
    pareto_entries: list[dict] = []
    if use_pareto:
        baseline_slices = evaluate_all_slices(actor, test_df, profile)
        baseline_objectives = pareto_objective_values(baseline_slices, pareto_objectives)
        pareto_entries.append(
            {
                "iter": 0,
                "state": copy.deepcopy(actor.state_dict()),
                "valid_metrics": baseline_metrics,
                "valid_score": baseline_score,
                "objectives": baseline_objectives,
            }
        )
        history[-1]["objectives"] = baseline_objectives

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
        metrics = _eval_valid()
        score = validation_score(metrics, profile)
        entry = {"iter": completed, "score": score, **metrics}

        if use_pareto:
            slices = evaluate_all_slices(actor, test_df, profile)
            objectives = pareto_objective_values(slices, pareto_objectives)
            entry["objectives"] = objectives
            pareto_entries.append(
                {
                    "iter": completed,
                    "state": copy.deepcopy(actor.state_dict()),
                    "valid_metrics": metrics,
                    "valid_score": score,
                    "objectives": objectives,
                }
            )
            obj_str = " ".join(f"{k}={v:.3f}" for k, v in objectives.items())
            print(
                f"[valid:{profile.name}@{completed}] return={metrics['return']:.1f}% "
                f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_dd']:.1f}% "
                f"score={score:.2f} pareto[{obj_str}]"
            )
        else:
            print(
                f"[valid:{profile.name}@{completed}] return={metrics['return']:.1f}% "
                f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_dd']:.1f}% score={score:.2f}"
            )
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_state = copy.deepcopy(actor.state_dict())

        history.append(entry)

    if use_pareto and pareto_entries:
        objective_dicts = [e["objectives"] for e in pareto_entries]
        front = pareto_front_indices(objective_dicts, pareto_objectives)
        # Tiebreak: among Pareto-optimal checkpoints, pick the one with the highest
        # scalar validation_score. This keeps backward compatibility with any
        # profile whose validation_score is well-calibrated while still culling
        # the dominated stages.
        winner_idx = max(front, key=lambda i: pareto_entries[i]["valid_score"])
        winner = pareto_entries[winner_idx]
        best_state = winner["state"]
        best_metrics = winner["valid_metrics"]
        best_score = winner["valid_score"]
        print(
            f"[pareto:{profile.name}] front={[pareto_entries[i]['iter'] for i in front]} "
            f"winner_iter={winner['iter']} objectives={winner['objectives']}"
        )

    actor.load_state_dict(best_state)
    return {"best_metrics": best_metrics, "best_score": best_score, "history": history}


__all__ = ["build_action_interpreter", "train_with_selection"]
