from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH

from ..policy.evaluation import evaluate_split
from ..policy.positioning import action_levels_for_profile


def validation_score(metrics: Dict[str, float], profile) -> float:
    cfg = profile.selection_score
    return float(
        metrics["return"] * cfg.return_weight
        + metrics["sharpe"] * cfg.sharpe_weight
        - metrics["max_dd"] * cfg.max_dd_penalty
        - metrics.get("avg_pos", 0.0) * cfg.avg_pos_penalty
        - metrics.get("stress_avg_pos", 0.0) * cfg.stress_avg_pos_penalty
    )


def slice_frame(df, seg: Tuple[str, str]):
    if isinstance(df, DatasetH):
        from ..data import segment_frame
        # Use empty string as segment name if seg isn't a known segment
        return segment_frame(df, seg[0])
    return df.loc[seg[0] : seg[1]].copy()


def describe_metrics(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"[{name}] return={metrics['return']:.1f}% "
        f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_dd']:.1f}%"
    )


def compact_result(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "return": float(result["return"]),
        "sharpe": float(result["sharpe"]),
        "max_dd": float(result["max_dd"]),
        "trades": int(result["trades"]),
        "valid": result.get("valid", {}),
        "slices": result.get("slices", {}),
        "test_proxy": result.get("test_proxy", {}),
        "promotion": result.get("promotion", {}),
        "model_path": result.get("model_path", ""),
        "summary_path": result.get("summary_path", ""),
    }


def slice_metrics(actor, df, slices: Dict[str, Tuple[str, str]], profile) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, seg in slices.items():
        segment = slice_frame(df, seg)
        if len(segment) <= 120:
            continue
        metrics[name] = evaluate_split(
            actor,
            segment,
            position_cap_config=profile.position_cap,
            position_budget_config=profile.position_budget,
            execution_config=profile.execution,
            asset_idx=0,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.use_hybrid_overlay else None,
            mode_router_config=profile.mode_router,
            position_mapping=profile.model.position_mapping,
        )
    return metrics


PARETO_SLICES: Dict[str, Tuple[str, str]] = {
    "test_proxy": ("2020-01-01", "2026-03-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


def evaluate_all_slices(actor, test_df, profile) -> Dict[str, Dict[str, float]]:
    """Compute the slice metrics needed for Pareto checkpoint selection.

    Returns a dict keyed by slice name with the full evaluate_split payload.
    Kept in the `app` layer so both `selection.py` and the post-train review
    pipeline can reuse it with identical semantics.
    """
    return slice_metrics(actor, test_df, PARETO_SLICES, profile)


def pareto_objective_values(
    slice_result: Dict[str, Dict[str, float]],
    objectives: Tuple[str, ...],
) -> Dict[str, float]:
    """Map slice metrics onto the config's ordered Pareto objectives.

    Known keys (all in "higher-is-better" space after sign conversion inside
    `pareto_dominates`):
      - test_2022_max_dd: lower is better (so we store raw and dominate on <=)
      - test_proxy_sharpe: higher is better
      - test_2023_2024_return: higher is better

    Missing slices (e.g. too short) default to sentinel values that lose every
    comparison so they never artificially gate out other checkpoints.
    """
    values: Dict[str, float] = {}
    for obj in objectives:
        if obj == "test_2022_max_dd":
            values[obj] = float(slice_result.get("test_2022", {}).get("max_dd", float("inf")))
        elif obj == "test_proxy_sharpe":
            values[obj] = float(slice_result.get("test_proxy", {}).get("sharpe", float("-inf")))
        elif obj == "test_2023_2024_return":
            values[obj] = float(slice_result.get("test_2023_2024", {}).get("return", float("-inf")))
        elif obj.endswith("_return"):
            slice_name = obj.rsplit("_return", 1)[0]
            values[obj] = float(slice_result.get(slice_name, {}).get("return", float("-inf")))
        elif obj.endswith("_sharpe"):
            slice_name = obj.rsplit("_sharpe", 1)[0]
            values[obj] = float(slice_result.get(slice_name, {}).get("sharpe", float("-inf")))
        elif obj.endswith("_max_dd"):
            slice_name = obj.rsplit("_max_dd", 1)[0]
            values[obj] = float(slice_result.get(slice_name, {}).get("max_dd", float("inf")))
        else:
            values[obj] = float("-inf")
    return values


def _objective_is_minimized(name: str) -> bool:
    return name.endswith("_max_dd") or name.endswith("_drawdown") or name.endswith("_avg_pos")


def pareto_dominates(
    a: Dict[str, float],
    b: Dict[str, float],
    objectives: Tuple[str, ...],
) -> bool:
    """Return True if ``a`` dominates ``b`` across all listed objectives.

    Objectives matching the min/max heuristic (`_max_dd` / `_drawdown` / `_avg_pos`
    are minimized, everything else maximized).
    """
    strictly_better = False
    for obj in objectives:
        av, bv = a[obj], b[obj]
        if _objective_is_minimized(obj):
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
        else:
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
    return strictly_better


def pareto_front_indices(
    candidates: list[Dict[str, float]],
    objectives: Tuple[str, ...],
) -> list[int]:
    """Return the indices of the Pareto-optimal candidates."""
    front: list[int] = []
    for i, ci in enumerate(candidates):
        dominated = False
        for j, cj in enumerate(candidates):
            if i == j:
                continue
            if pareto_dominates(cj, ci, objectives):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


__all__ = [
    "PARETO_SLICES",
    "compact_result",
    "describe_metrics",
    "evaluate_all_slices",
    "pareto_dominates",
    "pareto_front_indices",
    "pareto_objective_values",
    "slice_frame",
    "slice_metrics",
    "validation_score",
]
