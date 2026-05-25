from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from src.soxl_quant.app.oos_diagnostics import (
    attribute_gate_gaps,
    summarize_gate_gap_attribution,
)


@dataclass(frozen=True)
class WalkForwardObjectiveConfig:
    target_relative_return: float = 30.0
    target_max_dd: float = 15.0
    relative_gap_weight: float = 1.0
    max_dd_gap_weight: float = 10.0
    window_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "test_proxy": 0.75,
            "strict_oos": 0.75,
            "main": 0.75,
            "valid": 0.25,
            "test_2020": 0.75,
            "test_2022": 1.0,
            "test_2023_2024": 0.75,
            "wf_2020": 1.0,
            "wf_2022": 1.25,
            "wf_2023_2024": 1.0,
        }
    )


def _window_weight(window: str, config: WalkForwardObjectiveConfig) -> float:
    if window in config.window_weights:
        return float(config.window_weights[window])
    if window.startswith("wf_"):
        return 1.0
    return 0.75


def _next_mechanism(dominant_failure_type: str) -> str:
    if dominant_failure_type == "pass":
        return "promote_or_replay_candidate"
    if dominant_failure_type in {"drawdown_control", "near_miss_drawdown"}:
        return "drawdown_state_guard_or_sequence_risk_feature"
    if dominant_failure_type in {"underparticipation", "trend_capture"}:
        return "state_conditioned_participation_objective"
    return "walk_forward_weighted_multi_objective_training"


def score_walkforward_candidates(
    metrics_by_candidate: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    config: WalkForwardObjectiveConfig = WalkForwardObjectiveConfig(),
) -> pd.DataFrame:
    """Rank candidates by walk-forward weighted +30pp / <=15% objective gaps."""

    gate_gaps = attribute_gate_gaps(
        metrics_by_candidate,
        target_relative_return=config.target_relative_return,
        target_max_dd=config.target_max_dd,
    )
    if gate_gaps.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "window_count",
                "gate_pass_count",
                "full_gate",
                "weighted_objective_loss",
                "mean_relative_return",
                "min_relative_return",
                "max_max_dd",
                "dominant_failure_type",
                "worst_window",
                "next_mechanism",
            ]
        )

    rows: list[dict[str, object]] = []
    profile_summary = summarize_gate_gap_attribution(gate_gaps).set_index("profile")
    for candidate, sub in gate_gaps.groupby("profile", sort=True):
        weighted_losses = []
        for row in sub.itertuples(index=False):
            weight = _window_weight(str(row.window), config)
            weighted_losses.append(
                weight
                * (
                    float(row.relative_gap) * float(config.relative_gap_weight)
                    + float(row.max_dd_gap) * float(config.max_dd_gap_weight)
                )
            )
        full_gate = bool(sub["gate_pass"].all())
        if candidate in profile_summary.index:
            dominant = str(profile_summary.loc[candidate, "dominant_failure_type"])
            worst_window = str(profile_summary.loc[candidate, "worst_window"])
        else:
            dominant = "pass" if full_gate else "unknown"
            worst_window = str(sub.sort_values("severity_score", ascending=False).iloc[0]["window"])
        rows.append(
            {
                "candidate": str(candidate),
                "window_count": int(len(sub)),
                "gate_pass_count": int(sub["gate_pass"].sum()),
                "full_gate": full_gate,
                "weighted_objective_loss": float(np.sum(weighted_losses)),
                "mean_relative_return": float(sub["relative_return"].mean()),
                "min_relative_return": float(sub["relative_return"].min()),
                "max_max_dd": float(sub["max_dd"].max()),
                "mean_avg_pos": float(sub["avg_pos"].mean()),
                "dominant_failure_type": dominant,
                "worst_window": worst_window,
                "next_mechanism": _next_mechanism(dominant),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["full_gate", "weighted_objective_loss", "candidate"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def summarize_walkforward_selection(scored: pd.DataFrame) -> dict[str, object]:
    """Summarize a scored candidate table into a promotion/next-loop decision."""

    if scored.empty:
        return {
            "decision": "no_candidates",
            "selected_candidate": "",
            "next_mechanism": "collect_walk_forward_metrics",
        }
    top = scored.iloc[0]
    if bool(top["full_gate"]):
        decision = "select_full_gate_candidate"
        next_mechanism = str(top["next_mechanism"])
    else:
        decision = "no_full_gate_candidate"
        next_mechanism = "walk_forward_weighted_multi_objective_training"
    return {
        "decision": decision,
        "selected_candidate": str(top["candidate"]),
        "selected_weighted_objective_loss": float(top["weighted_objective_loss"]),
        "selected_dominant_failure_type": str(top["dominant_failure_type"]),
        "next_mechanism": next_mechanism,
    }


def render_walkforward_selection_review(
    scored: pd.DataFrame,
    summary: Mapping[str, object],
) -> str:
    lines = [
        "# Walk-Forward Weighted Objective Selection",
        "",
        "## Decision",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- selected_candidate: `{summary.get('selected_candidate', '')}`",
        f"- next_mechanism: `{summary.get('next_mechanism', '')}`",
        "",
        "## Candidate Ranking",
        "",
        "| candidate | full | loss | pass/windows | min rel | max dd | failure | next |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    if scored.empty:
        lines.append("| none | false | 0.00 | 0/0 | 0.00 | 0.00 | none | collect |")
    else:
        for row in scored.itertuples(index=False):
            lines.append(
                "| "
                f"{row.candidate} | {bool(row.full_gate)} | {float(row.weighted_objective_loss):.2f} | "
                f"{int(row.gate_pass_count)}/{int(row.window_count)} | "
                f"{float(row.min_relative_return):.2f}pp | {float(row.max_max_dd):.2f}% | "
                f"{row.dominant_failure_type} | {row.next_mechanism} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- A full-gate candidate must satisfy every supplied window under +30pp and <=15% MaxDD.",
            "- Lower weighted loss is diagnostic only when no candidate fully passes.",
            "- Keep this selector out of runtime auto-selection until a notebook promotes it with replay evidence.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "WalkForwardObjectiveConfig",
    "render_walkforward_selection_review",
    "score_walkforward_candidates",
    "summarize_walkforward_selection",
]
