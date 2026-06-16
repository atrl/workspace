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


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def build_reward_checkpoint_recipe_contract(
    screen_summary: Mapping[str, object],
    source_recipe: Mapping[str, object],
) -> dict[str, object]:
    """Build a machine-readable handoff before any reward/checkpoint training.

    The contract intentionally keeps training blocked. It records exactly what
    must be consumed and re-audited before a near-miss teacher can become a
    fold-local target or checkpoint objective.
    """

    target_contract_raw = source_recipe.get("target_contract")
    target_contract = dict(target_contract_raw) if isinstance(target_contract_raw, Mapping) else {}
    leakage_guard_raw = source_recipe.get("leakage_guard")
    leakage_guard = dict(leakage_guard_raw) if isinstance(leakage_guard_raw, Mapping) else {}
    final_windows = _string_list(
        screen_summary.get("final_required_windows") or target_contract.get("final_required_windows")
    )
    missing_windows = _string_list(
        screen_summary.get("missing_final_required_windows")
        or target_contract.get("missing_final_required_windows")
    )
    required_before_training = [
        "consume_reward_checkpoint_recipe_in_code",
        "fold_local_target_scope_audit",
    ]
    if missing_windows:
        required_before_training.append("add_wf_2020_official_replay_coverage")
    required_before_training.append("rerun_strict_final_walkforward_retention_gate")

    return {
        "decision": "reward_checkpoint_recipe_contract_ready",
        "source_teacher_decision": str(screen_summary.get("source_teacher_decision") or ""),
        "screen_decision": str(screen_summary.get("decision") or ""),
        "selected_candidate": str(screen_summary.get("selected_candidate") or ""),
        "source_screen": str(source_recipe.get("source_screen") or ""),
        "source_variant": str(source_recipe.get("source_variant") or ""),
        "source_candidate": str(source_recipe.get("source_candidate") or ""),
        "teacher_usage": str(source_recipe.get("teacher_usage") or ""),
        "training_allowed": False,
        "ppo_allowed_next": False,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "required_objective_terms": _string_list(screen_summary.get("required_objective_terms")),
        "final_required_windows": final_windows,
        "missing_final_required_windows": missing_windows,
        "required_before_training": required_before_training,
        "blocked_actions": [
            "do_not_promote_near_miss_candidate",
            "do_not_run_ppo_from_near_miss_without_scope_audit",
            "do_not_copy_eval_window_positions_into_training_labels",
            "do_not_skip_wf_2020_coverage",
        ],
        "reference_profile": str(target_contract.get("reference_profile") or ""),
        "target_contract": target_contract,
        "leakage_guard": leakage_guard,
    }


def _split_window_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = value.replace(",", ";").split(";")
        return [piece.strip() for piece in pieces if piece.strip()]
    return _string_list(value)


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed"}
    return bool(value)


def _final_window_key(value: object) -> str:
    text = str(value or "").strip()
    for suffix in ("_train", "_target", "_eval", "_test"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _fold_manifest_lookup(fold_local_target_manifest: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if fold_local_target_manifest is None or fold_local_target_manifest.empty or "fold" not in fold_local_target_manifest.columns:
        return {}
    out: dict[str, dict[str, object]] = {}
    for row in fold_local_target_manifest.to_dict("records"):
        for field in ("fold", "window", "target_fold"):
            raw = str(row.get(field, "") or "")
            if not raw:
                continue
            out.setdefault(raw, row)
            out.setdefault(_final_window_key(raw), row)
    return out


def audit_reward_checkpoint_recipe_scope(
    contract: Mapping[str, object],
    *,
    teacher_window_coverage: pd.DataFrame | None,
    selected_fold_metrics: pd.DataFrame,
    fold_local_target_manifest: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Audit whether a reward/checkpoint recipe may be consumed for training."""

    target_contract_raw = contract.get("target_contract")
    target_contract = dict(target_contract_raw) if isinstance(target_contract_raw, Mapping) else {}
    final_windows = _split_window_list(
        contract.get("final_required_windows") or target_contract.get("final_required_windows")
    )
    if not final_windows:
        final_windows = sorted({str(value) for value in selected_fold_metrics.get("fold", pd.Series(dtype=str)).dropna()})
    final_windows = [_final_window_key(value) for value in final_windows]
    explicit_missing = set(
        _final_window_key(value)
        for value in _split_window_list(
            contract.get("missing_final_required_windows")
            or target_contract.get("missing_final_required_windows")
        )
    )
    present_windows: set[str] = set()
    if "fold" in selected_fold_metrics.columns:
        present_windows.update(_final_window_key(value) for value in selected_fold_metrics["fold"].dropna().tolist())
    if teacher_window_coverage is not None and not teacher_window_coverage.empty:
        if "present_windows" in teacher_window_coverage.columns:
            for value in teacher_window_coverage["present_windows"].dropna().tolist():
                present_windows.update(_final_window_key(item) for item in _split_window_list(value))
        if "missing_final_required_windows" in teacher_window_coverage.columns:
            for value in teacher_window_coverage["missing_final_required_windows"].dropna().tolist():
                explicit_missing.update(_final_window_key(item) for item in _split_window_list(value))

    metric_by_fold: dict[str, dict[str, object]] = {}
    for row in selected_fold_metrics.to_dict("records"):
        fold = str(row.get("fold", "") or "")
        if not fold:
            continue
        metric_by_fold.setdefault(fold, row)
        metric_by_fold.setdefault(_final_window_key(fold), row)
    manifest_by_fold = _fold_manifest_lookup(fold_local_target_manifest)
    required_eval_overlap = int(
        dict(contract.get("leakage_guard", {})).get("eval_overlap_rows_required", 0)
        if isinstance(contract.get("leakage_guard"), Mapping)
        else 0
    )

    target_ready_by_window: dict[str, bool] = {}
    for window in final_windows:
        manifest_row = manifest_by_fold.get(window, {})
        if not manifest_row:
            target_ready_by_window[window] = False
            continue
        eval_overlap_rows = int(float(manifest_row.get("eval_overlap_rows", -1)))
        target_ready_by_window[window] = bool(manifest_row.get("target_path", "")) and eval_overlap_rows == required_eval_overlap

    coverage_rows = []
    official_missing_windows: list[str] = []
    for window in final_windows:
        row = metric_by_fold.get(window, {})
        official_present = window in present_windows and window not in explicit_missing
        target_present = bool(target_ready_by_window.get(window, False))
        present = official_present or target_present
        if not official_present:
            official_missing_windows.append(window)
        coverage_rows.append(
            {
                "window": window,
                "status": "present" if official_present else "target_ready" if target_present else "missing",
                "present": bool(present),
                "official_present": bool(official_present),
                "target_present": bool(target_present),
                "teacher_status": str(row.get("teacher_status", "missing" if not present else "")),
                "full_gate": bool(_bool_value(row.get("full_gate", False))) if row else False,
                "daily_path": str(row.get("daily_path", "")) if row else "",
            }
        )
    coverage_frame = pd.DataFrame(coverage_rows)
    missing_windows = coverage_frame.loc[~coverage_frame["present"], "window"].astype(str).tolist()
    coverage_gate = len(missing_windows) == 0
    official_replay_coverage_gate = len(official_missing_windows) == 0

    scope_rows = []
    for fold in final_windows:
        row = metric_by_fold.get(fold, {})
        teacher_usage = str(row.get("teacher_usage", ""))
        manifest_row = manifest_by_fold.get(fold, {})
        reference_only = (
            "reference_only" in teacher_usage
            or "not_training_label" in teacher_usage
            or (not manifest_row and not teacher_usage)
        )
        manifest_row = manifest_by_fold.get(fold, {})
        eval_overlap_rows = int(float(manifest_row.get("eval_overlap_rows", -1))) if manifest_row else -1
        target_path = str(manifest_row.get("target_path", "")) if manifest_row else ""
        target_ready = bool(target_path) and eval_overlap_rows == required_eval_overlap
        scope_rows.append(
            {
                "fold": fold,
                "daily_path": str(row.get("daily_path", "")),
                "teacher_status": str(row.get("teacher_status", "")),
                "usage": "reference_only" if reference_only else "fold_local_target_candidate",
                "eval_overlap_rows": eval_overlap_rows if manifest_row else None,
                "target_path": target_path,
                "target_ready": bool(target_ready),
            }
        )
    scope_frame = pd.DataFrame(scope_rows)
    ready_folds = set(scope_frame.loc[scope_frame["target_ready"], "fold"].astype(str).tolist()) if not scope_frame.empty else set()
    target_coverage_gate = set(final_windows).issubset(ready_folds)
    scope_gate = coverage_gate and target_coverage_gate

    consume_gate = str(contract.get("decision") or "") == "reward_checkpoint_recipe_contract_ready"
    gate_rows = [
        {
            "gate": "consume_reward_checkpoint_recipe_in_code",
            "status": "passed" if consume_gate else "blocked",
            "reason": "contract consumed by audit helper" if consume_gate else "missing ready recipe contract",
        },
        {
            "gate": "fold_local_target_scope_audit",
            "status": "passed" if scope_gate else "blocked",
            "reason": "all required folds have zero-overlap fold-local targets"
            if scope_gate
            else "selected daily paths are reference-only or fold-local target manifest is incomplete",
        },
    ]
    if official_missing_windows:
        gate_rows.append(
            {
                "gate": "add_wf_2020_official_replay_coverage",
                "status": "blocked",
                "reason": "missing official replay coverage for final windows: " + ",".join(official_missing_windows),
            }
        )
    gate_rows.append(
        {
            "gate": "rerun_strict_final_walkforward_retention_gate",
            "status": "passed" if consume_gate and official_replay_coverage_gate and scope_gate else "blocked",
            "reason": "ready to rerun strict retention gate"
            if consume_gate and official_replay_coverage_gate and scope_gate
            else "strict final gate must wait for recipe scope and official replay coverage gates",
        }
    )
    gate_frame = pd.DataFrame(gate_rows)
    training_allowed = consume_gate and coverage_gate and scope_gate
    promotion_allowed = False
    return {
        "summary": {
            "decision": "recipe_scope_audit_training_ready" if training_allowed else "recipe_scope_audit_blocks_training",
            "selected_candidate": str(contract.get("selected_candidate") or ""),
            "training_allowed": bool(training_allowed),
            "promotion_allowed": bool(promotion_allowed),
            "coverage_gate": bool(coverage_gate),
            "official_replay_coverage_gate": bool(official_replay_coverage_gate),
            "target_coverage_gate": bool(target_coverage_gate),
            "scope_gate": bool(scope_gate),
            "missing_final_required_windows": missing_windows,
            "official_missing_final_required_windows": official_missing_windows,
            "required_window_count": len(final_windows),
            "target_ready_window_count": len(ready_folds),
        },
        "coverage_rows": coverage_frame,
        "scope_rows": scope_frame,
        "required_gate_rows": gate_frame,
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
    "audit_reward_checkpoint_recipe_scope",
    "build_reward_checkpoint_recipe_contract",
    "render_reward_checkpoint_screen_review",
    "score_reward_checkpoint_candidates",
    "summarize_reward_checkpoint_screen",
]
