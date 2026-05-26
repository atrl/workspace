from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RewardCheckpointScreenConfig:
    target_relative_return: float = 30.0
    target_max_dd: float = 15.0
    relative_gap_weight: float = 1.0
    max_dd_gap_weight: float = 10.0
    max_dd_drift_weight: float = 5.0
    relative_drift_weight: float = 0.25
    missing_short_volshock_penalty: float = 25.0
    short_volshock_accuracy_floor: float = 0.60
    min_short_volshock_samples: int = 10


def _candidate_col(frame: pd.DataFrame) -> str:
    if "candidate" in frame.columns:
        return "candidate"
    if "profile" in frame.columns:
        return "profile"
    raise ValueError("frame must include `candidate` or `profile`")


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _reference_by_window(gate_gaps: pd.DataFrame, reference_candidate: str) -> dict[str, dict[str, float]]:
    if not reference_candidate:
        return {}
    ref = gate_gaps[gate_gaps["profile"].astype(str).eq(str(reference_candidate))].copy()
    if ref.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in ref.to_dict("records"):
        out[str(row.get("window", ""))] = {
            "relative_return": float(row.get("relative_return", 0.0) or 0.0),
            "max_dd": float(row.get("max_dd", 0.0) or 0.0),
        }
    return out


def _slice_lookup(slice_scores: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if slice_scores is None or slice_scores.empty:
        return {}
    col = _candidate_col(slice_scores)
    out: dict[str, dict[str, float]] = {}
    for row in slice_scores.to_dict("records"):
        out[str(row[col])] = {
            "mean_short_volshock_accuracy": float(row.get("mean_short_volshock_accuracy", np.nan)),
            "short_volshock_target_samples": float(row.get("short_volshock_target_samples", 0.0) or 0.0),
            "slice_gate_pass_count": float(row.get("slice_gate_pass_count", 0.0) or 0.0),
        }
    return out


def _failure_mix(values: pd.Series) -> str:
    if values.empty:
        return ""
    counts = values.astype(str).value_counts()
    return ",".join(f"{name}:{int(count)}" for name, count in counts.items())


def _feature_group_set(feature_groups: pd.DataFrame | None) -> set[str]:
    if feature_groups is None or feature_groups.empty or "feature_group" not in feature_groups.columns:
        return set()
    return {str(value) for value in feature_groups["feature_group"].dropna().tolist()}


def _has_underparticipation_clusters(poor_window_clusters: pd.DataFrame | None) -> bool:
    if poor_window_clusters is None or poor_window_clusters.empty:
        return False
    if "dominant_failure_type" not in poor_window_clusters.columns:
        return False
    return bool(poor_window_clusters["dominant_failure_type"].astype(str).eq("underparticipation").any())


def _required_terms(
    *,
    failure_types: set[str],
    feature_groups: set[str],
    has_underparticipation_clusters: bool,
    has_short_volshock_targets: bool,
) -> list[str]:
    terms = ["checkpoint_objective_loss"]
    if failure_types & {"drawdown_control", "joint_tradeoff", "near_miss_drawdown"}:
        terms.append("drawdown_drift_penalty")
    if has_short_volshock_targets:
        terms.append("short_volshock_state_deleverage_consistency")
    if (
        has_underparticipation_clusters
        or failure_types & {"joint_tradeoff", "underparticipation", "trend_capture", "selection_bias"}
        or feature_groups & {"trend_quality", "trend_momentum", "relative_strength_spread"}
    ):
        terms.append("trend_preservation_floor")
    return terms


def score_reward_checkpoint_candidates(
    gate_gaps: pd.DataFrame,
    *,
    slice_scores: pd.DataFrame | None = None,
    poor_window_clusters: pd.DataFrame | None = None,
    feature_groups: pd.DataFrame | None = None,
    reference_candidate: str = "",
    config: RewardCheckpointScreenConfig = RewardCheckpointScreenConfig(),
) -> pd.DataFrame:
    """Rank reward/checkpoint candidates before allowing PPO continuation."""

    required = {"profile", "window", "relative_gap", "max_dd_gap", "relative_return", "max_dd", "gate_pass"}
    missing = sorted(required - set(gate_gaps.columns))
    if missing:
        raise ValueError(f"gate_gaps missing columns: {missing}")
    if gate_gaps.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "full_gate",
                "reward_checkpoint_loss",
                "required_objective_terms",
            ]
        )

    frame = gate_gaps.copy()
    frame["profile"] = frame["profile"].astype(str)
    frame["window"] = frame["window"].astype(str)
    for column in ("relative_gap", "max_dd_gap", "relative_return", "max_dd"):
        frame[column] = _numeric(frame, column)
    frame["gate_pass"] = frame["gate_pass"].fillna(False).astype(bool)
    if "failure_type" not in frame.columns:
        frame["failure_type"] = np.where(frame["gate_pass"], "pass", "unknown")

    reference = _reference_by_window(frame, reference_candidate)
    slice_by_candidate = _slice_lookup(slice_scores)
    slice_scope_enabled = slice_scores is not None and not slice_scores.empty
    groups = _feature_group_set(feature_groups)
    under_clusters = _has_underparticipation_clusters(poor_window_clusters)
    rows: list[dict[str, object]] = []

    for candidate, sub in frame.groupby("profile", sort=True):
        objective_loss = 0.0
        relative_drift_loss = 0.0
        max_dd_drift_loss = 0.0
        for row in sub.to_dict("records"):
            objective_loss += float(row["relative_gap"]) * float(config.relative_gap_weight)
            objective_loss += float(row["max_dd_gap"]) * float(config.max_dd_gap_weight)
            ref = reference.get(str(row.get("window", "")))
            if ref and str(candidate) != str(reference_candidate):
                relative_drift_loss += max(0.0, ref["relative_return"] - float(row["relative_return"]))
                max_dd_drift_loss += max(0.0, float(row["max_dd"]) - ref["max_dd"])

        slice_row = slice_by_candidate.get(str(candidate), {})
        short_samples = float(slice_row.get("short_volshock_target_samples", 0.0) or 0.0)
        short_accuracy = float(slice_row.get("mean_short_volshock_accuracy", np.nan))
        short_loss = 0.0
        needs_short_volshock_slice = "short_volshock" in str(candidate) or short_samples > 0
        if slice_scope_enabled and needs_short_volshock_slice:
            if short_samples < int(config.min_short_volshock_samples):
                short_loss += float(config.missing_short_volshock_penalty)
            if np.isfinite(short_accuracy):
                short_loss += max(0.0, float(config.short_volshock_accuracy_floor) - short_accuracy) * 10.0

        loss = (
            objective_loss
            + relative_drift_loss * float(config.relative_drift_weight)
            + max_dd_drift_loss * float(config.max_dd_drift_weight)
            + short_loss
        )
        failure_types = set(sub["failure_type"].astype(str).tolist())
        required_terms = _required_terms(
            failure_types=failure_types,
            feature_groups=groups,
            has_underparticipation_clusters=under_clusters,
            has_short_volshock_targets=short_samples >= int(config.min_short_volshock_samples),
        )
        rows.append(
            {
                "candidate": str(candidate),
                "window_count": int(len(sub)),
                "gate_pass_count": int(sub["gate_pass"].sum()),
                "full_gate": bool(sub["gate_pass"].all()),
                "reward_checkpoint_loss": float(loss),
                "gate_gap_loss": float(objective_loss),
                "relative_drift_loss": float(relative_drift_loss),
                "max_dd_drift_loss": float(max_dd_drift_loss),
                "short_volshock_slice_loss": float(short_loss),
                "min_relative_return": float(sub["relative_return"].min()),
                "max_max_dd": float(sub["max_dd"].max()),
                "failure_mix": _failure_mix(sub["failure_type"]),
                "dominant_failure_type": str(sub["failure_type"].astype(str).value_counts().index[0]),
                "required_objective_terms": json.dumps(required_terms, sort_keys=True),
                "short_volshock_target_samples": int(short_samples),
                "mean_short_volshock_accuracy": short_accuracy if np.isfinite(short_accuracy) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["full_gate", "reward_checkpoint_loss", "candidate"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def summarize_reward_checkpoint_screen(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "decision": "reward_checkpoint_screen_no_candidates",
            "selected_candidate": "",
            "promotion_allowed": False,
            "ppo_allowed_next": False,
            "required_objective_terms": [],
            "next_mechanism": "collect_gate_gap_and_slice_scores",
        }
    top = scored.iloc[0]
    terms = json.loads(str(top.get("required_objective_terms", "[]")))
    if bool(top["full_gate"]):
        decision = "reward_checkpoint_promotion_review"
        ppo_allowed_next = True
        next_mechanism = "full_fixed_split_and_walkforward_replay_review"
    else:
        decision = "reward_checkpoint_screen_no_promotion"
        ppo_allowed_next = False
        next_mechanism = "train_drawdown_drift_reward_or_checkpoint_recipe"
    return {
        "decision": decision,
        "selected_candidate": str(top["candidate"]),
        "selected_reward_checkpoint_loss": float(top["reward_checkpoint_loss"]),
        "selected_gate_gap_loss": float(top.get("gate_gap_loss", 0.0)),
        "selected_failure_mix": str(top.get("failure_mix", "")),
        "selected_min_relative_return": float(top.get("min_relative_return", 0.0)),
        "selected_max_max_dd": float(top.get("max_max_dd", 0.0)),
        "promotion_allowed": False,
        "ppo_allowed_next": bool(ppo_allowed_next),
        "required_objective_terms": terms,
        "next_mechanism": next_mechanism,
    }


def render_reward_checkpoint_screen_review(
    scored: pd.DataFrame,
    summary: Mapping[str, object],
) -> str:
    lines = [
        "# Reward / Checkpoint Screen",
        "",
        "## Decision",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- selected_candidate: `{summary.get('selected_candidate', '')}`",
        f"- next_mechanism: `{summary.get('next_mechanism', '')}`",
        "",
        "## Required Objective Terms",
        "",
    ]
    for term in summary.get("required_objective_terms", []):
        lines.append(f"- `{term}`")
    lines.extend(
        [
            "",
            "## Candidate Ranking",
            "",
            "| candidate | full | loss | pass/windows | min rel | max dd | failures |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if scored.empty:
        lines.append("| none | false | 0.00 | 0/0 | 0.00 | 0.00 | none |")
    else:
        for row in scored.itertuples(index=False):
            lines.append(
                "| "
                f"{row.candidate} | {bool(row.full_gate)} | {float(row.reward_checkpoint_loss):.2f} | "
                f"{int(row.gate_pass_count)}/{int(row.window_count)} | "
                f"{float(row.min_relative_return):.2f}pp | {float(row.max_max_dd):.2f}% | "
                f"{row.failure_mix} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This screen is a pre-PPO reward/checkpoint diagnostic, not runtime promotion.",
            "- If no full-gate candidate exists, use the required terms as the next reward/checkpoint recipe.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "RewardCheckpointScreenConfig",
    "render_reward_checkpoint_screen_review",
    "score_reward_checkpoint_candidates",
    "summarize_reward_checkpoint_screen",
]
