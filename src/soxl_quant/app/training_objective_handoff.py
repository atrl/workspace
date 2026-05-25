from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


TARGET_RELATIVE_RETURN = 30.0
TARGET_MAX_DD = 15.0


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.to_dict(orient="records")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _top_walkforward_row(walkforward_scores: pd.DataFrame) -> dict[str, Any]:
    if walkforward_scores.empty:
        return {}
    if "full_gate" in walkforward_scores.columns and "weighted_objective_loss" in walkforward_scores.columns:
        sorted_scores = walkforward_scores.sort_values(
            ["full_gate", "weighted_objective_loss"],
            ascending=[False, True],
        )
        return dict(sorted_scores.iloc[0])
    return dict(walkforward_scores.iloc[0])


def _top_underparticipation_cluster(poor_window_clusters: pd.DataFrame | None) -> dict[str, Any]:
    if poor_window_clusters is None or poor_window_clusters.empty:
        return {}
    clusters = poor_window_clusters.copy()
    if "dominant_failure_type" in clusters.columns:
        clusters = clusters[clusters["dominant_failure_type"] == "underparticipation"]
    if clusters.empty:
        return {}
    score_col = "mean_excess_return" if "mean_excess_return" in clusters.columns else None
    if score_col:
        clusters = clusters.sort_values(score_col, ascending=True)
    return dict(clusters.iloc[0])


def _feature_groups(feature_groups: pd.DataFrame | None, *, top_n: int = 2) -> list[str]:
    if feature_groups is None or feature_groups.empty or "feature_group" not in feature_groups.columns:
        return []
    ranked = feature_groups.copy()
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "top_mentions" in ranked.columns:
        sort_cols.append("top_mentions")
        ascending.append(False)
    if "mean_abs_delta_z" in ranked.columns:
        sort_cols.append("mean_abs_delta_z")
        ascending.append(False)
    if sort_cols:
        ranked = ranked.sort_values(sort_cols, ascending=ascending)
    return [str(value) for value in ranked["feature_group"].head(top_n).tolist()]


def _router_failed_after_cluster_improvement(router_selection: Mapping[str, Any] | None) -> bool:
    if not router_selection:
        return False
    if int(router_selection.get("full_pass_count", 0) or 0) > 0:
        return False
    decision = str(router_selection.get("decision", ""))
    if "no_promotion" in decision or "diagnostic" in decision:
        return True
    conclusions = " ".join(str(item) for item in router_selection.get("conclusion", []))
    return "No " in conclusions and "gate" in conclusions


def _max_cluster_delta(router_selection: Mapping[str, Any] | None) -> float:
    if not router_selection:
        return 0.0
    comparisons = router_selection.get("selected_cluster_comparison", [])
    if not isinstance(comparisons, Sequence):
        return 0.0
    deltas = []
    for row in comparisons:
        if isinstance(row, Mapping):
            deltas.append(float(row.get("excess_delta_vs_base", 0.0) or 0.0))
    return float(max(deltas, default=0.0))


def _risk_gap(top: Mapping[str, Any]) -> dict[str, Any]:
    max_dd = float(top.get("max_max_dd", top.get("max_dd", 0.0)) or 0.0)
    min_relative = float(top.get("min_relative_return", top.get("relative_return", 0.0)) or 0.0)
    return {
        "target_max_dd": TARGET_MAX_DD,
        "observed_max_dd": round(max_dd, 2),
        "max_dd_gap": round(max(0.0, max_dd - TARGET_MAX_DD), 2),
        "target_relative_return": TARGET_RELATIVE_RETURN,
        "observed_min_relative_return": round(min_relative, 2),
        "relative_gap": round(max(0.0, TARGET_RELATIVE_RETURN - min_relative), 2),
        "worst_window": str(top.get("worst_window", "")),
    }


def derive_training_objective_handoff(
    *,
    walkforward_scores: pd.DataFrame,
    poor_window_clusters: pd.DataFrame,
    feature_groups: pd.DataFrame | None = None,
    router_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine selection and poor-window evidence into the next training objective."""

    top = _top_walkforward_row(walkforward_scores)
    if not top:
        return _json_safe(
            {
            "decision": "collect_walkforward_evidence",
            "promotion": "blocked",
            "selected_diagnostic_candidate": "",
            "objective_terms": ["walkforward_metric_collection"],
            "stop_mechanisms": [],
            }
        )

    selected = str(top.get("candidate", top.get("profile", "")))
    if bool(top.get("full_gate", False)):
        return _json_safe(
            {
            "decision": "candidate_promotion_review",
            "promotion": "review_required",
            "selected_diagnostic_candidate": selected,
            "risk_gap": _risk_gap(top),
            "objective_terms": [],
            "stop_mechanisms": [],
            }
        )

    under_cluster = _top_underparticipation_cluster(poor_window_clusters)
    router_failed = _router_failed_after_cluster_improvement(router_selection)
    dominant_failure = str(top.get("dominant_failure_type", ""))
    state_groups = _feature_groups(feature_groups)

    stop_mechanisms: list[str] = []
    if router_failed:
        stop_mechanisms.append("path_level_threshold_router_search")

    objective_terms = [
        "walkforward_relative_return_gap_penalty",
        "walkforward_drawdown_gap_penalty",
    ]
    if under_cluster:
        objective_terms.append("high_vol_rebound_participation_floor")
    if dominant_failure in {"drawdown_control", "near_miss_drawdown", "joint_tradeoff"}:
        objective_terms.append("downtrend_drawdown_state_cap")

    if under_cluster and dominant_failure in {"drawdown_control", "joint_tradeoff", "near_miss_drawdown"}:
        decision = "train_state_conditioned_multi_objective"
    elif under_cluster:
        decision = "train_rebound_participation_objective"
    elif dominant_failure in {"drawdown_control", "near_miss_drawdown"}:
        decision = "train_drawdown_control_objective"
    else:
        decision = "train_walkforward_weighted_objective"

    return _json_safe(
        {
        "decision": decision,
        "promotion": "blocked",
        "selected_diagnostic_candidate": selected,
        "selected_dominant_failure_type": dominant_failure,
        "risk_gap": _risk_gap(top),
        "primary_underparticipation_cluster": under_cluster,
        "state_feature_groups": state_groups,
        "objective_terms": objective_terms,
        "stop_mechanisms": stop_mechanisms,
        "router_cluster_delta_max": round(_max_cluster_delta(router_selection), 2),
        "input_counts": {
            "walkforward_candidates": int(len(walkforward_scores)),
            "poor_window_clusters": int(len(poor_window_clusters)),
            "feature_groups": int(0 if feature_groups is None else len(feature_groups)),
        },
        }
    )


def render_training_objective_handoff(handoff: Mapping[str, Any]) -> str:
    cluster = handoff.get("primary_underparticipation_cluster", {})
    if not isinstance(cluster, Mapping):
        cluster = {}
    lines = [
        "# Training Objective Handoff",
        "",
        "## Decision",
        "",
        f"- decision: `{handoff.get('decision', '')}`",
        f"- promotion: `{handoff.get('promotion', '')}`",
        f"- selected_diagnostic_candidate: `{handoff.get('selected_diagnostic_candidate', '')}`",
        "",
        "## Objective Terms",
        "",
    ]
    for term in handoff.get("objective_terms", []):
        lines.append(f"- `{term}`")
    lines.extend(["", "## Risk Gap", ""])
    risk_gap = handoff.get("risk_gap", {})
    if isinstance(risk_gap, Mapping):
        lines.append(
            "- "
            f"max_dd_gap={risk_gap.get('max_dd_gap', 0)}pp, "
            f"relative_gap={risk_gap.get('relative_gap', 0)}pp, "
            f"worst_window=`{risk_gap.get('worst_window', '')}`"
        )
    if cluster:
        lines.extend(
            [
                "",
                "## Primary Underparticipation Cluster",
                "",
                "- "
                f"cluster=`{cluster.get('cluster_id', cluster.get('cluster', ''))}`, "
                f"start={cluster.get('cluster_start', cluster.get('start', ''))}, "
                f"end={cluster.get('cluster_end', cluster.get('end', ''))}, "
                f"mean_excess={float(cluster.get('mean_excess_return', cluster.get('median_excess', 0.0))):.2f}pp",
            ]
        )
    stop = handoff.get("stop_mechanisms", [])
    if stop:
        lines.extend(["", "## Stop Mechanisms", ""])
        for mechanism in stop:
            lines.append(f"- `{mechanism}`")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "derive_training_objective_handoff",
    "render_training_objective_handoff",
]
