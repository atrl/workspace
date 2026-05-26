from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstrainedCheckpointSelectorConfig:
    target_relative_return_pp: float = 30.0
    target_max_dd_pct: float = 15.0
    relative_gap_weight: float = 1.0
    max_dd_gap_weight: float = 10.0
    failed_window_weight: float = 100.0
    relative_drift_weight: float = 0.25
    max_dd_drift_weight: float = 2.0
    alignment_loss_weight: float = 1.0
    checkpoint_objective_weight: float = 0.25
    missing_target_ratio_weight: float = 100.0
    coverage_failure_penalty: float = 25.0


def _candidate_col(frame: pd.DataFrame) -> str:
    if "candidate" in frame.columns:
        return "candidate"
    if "profile" in frame.columns:
        return "profile"
    raise ValueError("frame must include `candidate` or `profile`")


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"gate_metrics must include one of {names} for `{label}`")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return default
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return default


def _normalise_gate_metrics(
    gate_metrics: pd.DataFrame,
    config: ConstrainedCheckpointSelectorConfig,
) -> pd.DataFrame:
    if gate_metrics.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "window",
                "relative_return",
                "max_dd",
                "relative_gap",
                "max_dd_gap",
                "relative_pass",
                "maxdd_pass",
                "window_gate_pass",
            ]
        )
    candidate_col = _candidate_col(gate_metrics)
    window_col = _first_existing(gate_metrics, ("window", "fold"), "window")
    relative_col = _first_existing(
        gate_metrics,
        ("relative_return", "official_warm_relative_return", "relative_return_pp"),
        "relative_return",
    )
    max_dd_col = _first_existing(
        gate_metrics,
        ("max_dd", "official_warm_max_dd", "max_dd_pct"),
        "max_dd",
    )

    out = pd.DataFrame(
        {
            "candidate": gate_metrics[candidate_col].astype(str),
            "window": gate_metrics[window_col].astype(str),
            "relative_return": _numeric(gate_metrics, relative_col),
            "max_dd": _numeric(gate_metrics, max_dd_col),
        }
    )
    rel = out["relative_return"].to_numpy(dtype=float)
    max_dd = out["max_dd"].to_numpy(dtype=float)
    valid_rel = np.isfinite(rel)
    valid_dd = np.isfinite(max_dd)
    out["relative_pass"] = valid_rel & (rel >= float(config.target_relative_return_pp))
    out["maxdd_pass"] = valid_dd & (max_dd <= float(config.target_max_dd_pct))
    out["relative_gap"] = np.where(
        valid_rel,
        np.maximum(0.0, float(config.target_relative_return_pp) - rel),
        float(config.target_relative_return_pp),
    )
    out["max_dd_gap"] = np.where(
        valid_dd,
        np.maximum(0.0, max_dd - float(config.target_max_dd_pct)),
        float(config.target_max_dd_pct),
    )
    out["window_gate_pass"] = out["relative_pass"] & out["maxdd_pass"]
    return out


def _checkpoint_lookup(checkpoint_scores: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if checkpoint_scores is None or checkpoint_scores.empty:
        return {}
    candidate_col = _candidate_col(checkpoint_scores)
    lookup: dict[str, dict[str, object]] = {}
    for row in checkpoint_scores.to_dict("records"):
        lookup[str(row[candidate_col])] = dict(row)
    return lookup


def _reference_by_window(frame: pd.DataFrame, reference_candidate: str) -> dict[str, dict[str, float]]:
    if not reference_candidate:
        return {}
    ref = frame[frame["candidate"].astype(str).eq(str(reference_candidate))]
    if ref.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in ref.to_dict("records"):
        out[str(row["window"])] = {
            "relative_return": _float(row.get("relative_return")),
            "max_dd": _float(row.get("max_dd")),
        }
    return out


def score_constrained_checkpoints(
    gate_metrics: pd.DataFrame,
    *,
    checkpoint_scores: pd.DataFrame | None = None,
    reference_candidate: str = "",
    config: ConstrainedCheckpointSelectorConfig = ConstrainedCheckpointSelectorConfig(),
) -> pd.DataFrame:
    """Rank checkpoint candidates with hard walk-forward return/drawdown constraints.

    The selector is intentionally conservative: a candidate that clears every
    target window is always ranked before candidates with any window failure.
    Objective, alignment, coverage, and reference-drift terms only break ties
    within the pass/fail groups.
    """

    frame = _normalise_gate_metrics(gate_metrics, config)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "selector_gate_pass",
                "constrained_selector_loss",
                "window_count",
                "gate_pass_count",
            ]
        )

    score_by_candidate = _checkpoint_lookup(checkpoint_scores)
    reference = _reference_by_window(frame, reference_candidate)
    rows: list[dict[str, object]] = []

    for candidate, sub in frame.groupby("candidate", sort=True):
        window_count = int(len(sub))
        gate_pass_count = int(sub["window_gate_pass"].sum())
        failed_window_count = window_count - gate_pass_count
        relative_failure_count = int((~sub["relative_pass"]).sum())
        maxdd_failure_count = int((~sub["maxdd_pass"]).sum())
        relative_gap_sum = float(sub["relative_gap"].sum())
        max_dd_gap_sum = float(sub["max_dd_gap"].sum())
        constraint_gap_loss = (
            relative_gap_sum * float(config.relative_gap_weight)
            + max_dd_gap_sum * float(config.max_dd_gap_weight)
        )
        failed_window_loss = failed_window_count * float(config.failed_window_weight)

        relative_drift_gap = 0.0
        max_dd_drift_gap = 0.0
        for row in sub.to_dict("records"):
            ref = reference.get(str(row["window"]))
            if not ref or str(candidate) == str(reference_candidate):
                continue
            relative_drift_gap += max(0.0, ref["relative_return"] - _float(row.get("relative_return")))
            max_dd_drift_gap += max(0.0, _float(row.get("max_dd")) - ref["max_dd"])
        drift_loss = (
            relative_drift_gap * float(config.relative_drift_weight)
            + max_dd_drift_gap * float(config.max_dd_drift_weight)
        )

        score_row = score_by_candidate.get(str(candidate), {})
        alignment_loss = _float(score_row.get("alignment_loss"))
        checkpoint_objective_loss = _float(score_row.get("checkpoint_objective_loss"))
        missing_target_ratio = _float(score_row.get("missing_target_ratio"))
        coverage_gate = _bool(score_row.get("coverage_gate", True), default=True)
        score_loss = (
            alignment_loss * float(config.alignment_loss_weight)
            + checkpoint_objective_loss * float(config.checkpoint_objective_weight)
            + missing_target_ratio * float(config.missing_target_ratio_weight)
            + (0.0 if coverage_gate else float(config.coverage_failure_penalty))
        )

        selector_gate_pass = bool(sub["window_gate_pass"].all())
        constrained_selector_loss = constraint_gap_loss + failed_window_loss + drift_loss + score_loss
        rows.append(
            {
                "candidate": str(candidate),
                "selector_gate_pass": selector_gate_pass,
                "decision": (
                    "constrained_checkpoint_candidate"
                    if selector_gate_pass
                    else "constrained_checkpoint_no_promotion"
                ),
                "constrained_selector_loss": float(constrained_selector_loss),
                "constraint_gap_loss": float(constraint_gap_loss),
                "failed_window_loss": float(failed_window_loss),
                "drift_loss": float(drift_loss),
                "score_loss": float(score_loss),
                "window_count": window_count,
                "gate_pass_count": gate_pass_count,
                "failed_window_count": failed_window_count,
                "relative_failure_count": relative_failure_count,
                "maxdd_failure_count": maxdd_failure_count,
                "relative_gap_sum": relative_gap_sum,
                "max_dd_gap_sum": max_dd_gap_sum,
                "max_relative_gap": float(sub["relative_gap"].max()),
                "max_dd_gap": float(sub["max_dd_gap"].max()),
                "min_relative_return": float(sub["relative_return"].min()),
                "max_max_dd": float(sub["max_dd"].max()),
                "relative_drift_gap": float(relative_drift_gap),
                "max_dd_drift_gap": float(max_dd_drift_gap),
                "alignment_loss": float(alignment_loss),
                "checkpoint_objective_loss": float(checkpoint_objective_loss),
                "missing_target_ratio": float(missing_target_ratio),
                "coverage_gate": bool(coverage_gate),
                "target_relative_return_pp": float(config.target_relative_return_pp),
                "target_max_dd_pct": float(config.target_max_dd_pct),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "selector_gate_pass",
            "constrained_selector_loss",
            "failed_window_count",
            "max_dd_gap_sum",
            "relative_gap_sum",
            "candidate",
        ],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)


def summarize_constrained_checkpoint_selection(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "decision": "constrained_checkpoint_selector_no_candidates",
            "selected_candidate": "",
            "promotion_allowed": False,
            "replay_review_required": False,
            "next_mechanism": "collect_walkforward_gate_metrics",
        }
    top = scored.iloc[0]
    selector_gate_pass = bool(top["selector_gate_pass"])
    return {
        "decision": (
            "constrained_checkpoint_candidate_ready_for_replay_review"
            if selector_gate_pass
            else "constrained_checkpoint_selector_no_promotion"
        ),
        "selected_candidate": str(top["candidate"]),
        "selected_constrained_selector_loss": float(top["constrained_selector_loss"]),
        "selected_constraint_gap_loss": float(top["constraint_gap_loss"]),
        "selected_failed_window_count": int(top["failed_window_count"]),
        "selected_relative_failure_count": int(top["relative_failure_count"]),
        "selected_maxdd_failure_count": int(top["maxdd_failure_count"]),
        "selected_min_relative_return": float(top["min_relative_return"]),
        "selected_max_max_dd": float(top["max_max_dd"]),
        "selector_gate_pass": selector_gate_pass,
        "promotion_allowed": False,
        "replay_review_required": selector_gate_pass,
        "next_mechanism": (
            "full_fixed_split_and_walkforward_replay_review"
            if selector_gate_pass
            else "generate_reward_constrained_short_training_candidate"
        ),
        "acceptance_criteria": {
            "target_relative_return_pp": float(top.get("target_relative_return_pp", 30.0)),
            "target_max_dd_pct": float(top.get("target_max_dd_pct", 15.0)),
            "all_windows_must_pass": True,
        },
    }


def render_constrained_checkpoint_selection_review(
    scored: pd.DataFrame,
    summary: Mapping[str, object],
) -> str:
    lines = [
        "# Constrained Checkpoint Selector",
        "",
        "## Decision",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- selected_candidate: `{summary.get('selected_candidate', '')}`",
        f"- selector_gate_pass: `{summary.get('selector_gate_pass', False)}`",
        f"- promotion_allowed: `{summary.get('promotion_allowed', False)}`",
        f"- next_mechanism: `{summary.get('next_mechanism', '')}`",
        "",
        "## Candidate Ranking",
        "",
        "| candidate | gate | loss | pass/windows | min rel | max dd | rel fails | dd fails |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if scored.empty:
        lines.append("| none | false | 0.00 | 0/0 | 0.00pp | 0.00% | 0 | 0 |")
    else:
        for row in scored.itertuples(index=False):
            lines.append(
                "| "
                f"{row.candidate} | {bool(row.selector_gate_pass)} | "
                f"{float(row.constrained_selector_loss):.2f} | "
                f"{int(row.gate_pass_count)}/{int(row.window_count)} | "
                f"{float(row.min_relative_return):.2f}pp | {float(row.max_max_dd):.2f}% | "
                f"{int(row.relative_failure_count)} | {int(row.maxdd_failure_count)} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- A full-gate candidate still requires replay review before promotion.",
            "- If no candidate clears every window, the selector only identifies the closest diagnostic candidate.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ConstrainedCheckpointSelectorConfig",
    "render_constrained_checkpoint_selection_review",
    "score_constrained_checkpoints",
    "summarize_constrained_checkpoint_selection",
]
