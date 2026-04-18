from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

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
            execution_config=profile.execution,
            asset_idx=0,
            action_levels=action_levels_for_profile(profile),
            observation_window=profile.model.observation_window,
            hard_floor=profile.model.hard_floor,
        )
    return metrics


__all__ = [
    "compact_result",
    "describe_metrics",
    "slice_frame",
    "slice_metrics",
    "validation_score",
]
