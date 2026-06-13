from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TARGET_RELATIVE_RETURN_PP = 30.0
DEFAULT_TARGET_MAX_DD_PCT = 20.0


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _required_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def _failure_type(
    *,
    fold: str,
    selected_gate: bool,
    selected_relative_gap_pp: float,
    selected_drawdown_gap_pct: float,
    selected_return: float,
    selected_buyhold_return: float,
) -> str:
    if selected_gate or (selected_relative_gap_pp <= 0.0 and selected_drawdown_gap_pct <= 0.0):
        return "pass"
    if selected_relative_gap_pp > 0.0 and selected_drawdown_gap_pct > 0.0:
        return "joint_tradeoff"
    if selected_drawdown_gap_pct > 0.0:
        return "drawdown_control"
    if fold == "wf_2020" or (selected_buyhold_return > 0.0 and selected_return < selected_buyhold_return):
        return "bull_underparticipation"
    return "relative_return_gap"


def _next_training_target(failure_type: str) -> str:
    if failure_type == "pass":
        return "none"
    if failure_type == "bull_underparticipation":
        return "bull_rebound_participation_lift_with_drawdown_budget"
    if failure_type == "drawdown_control":
        return "drawdown_controller_or_sequence_risk_feature"
    if failure_type == "joint_tradeoff":
        return "walkforward_multi_objective_return_drawdown_tradeoff"
    return "relative_return_lift_without_drawdown_regression"


def classify_walkforward_checkpoint_failures(
    candidates: pd.DataFrame,
    selected_by_fold: pd.DataFrame,
    *,
    target_relative_return_pp: float = DEFAULT_TARGET_RELATIVE_RETURN_PP,
    target_max_dd_pct: float = DEFAULT_TARGET_MAX_DD_PCT,
) -> pd.DataFrame:
    """Classify why selected walk-forward checkpoint candidates miss retention gates."""
    _required_columns(
        candidates,
        {
            "fold",
            "candidate",
            "test_return",
            "test_buy_and_hold_return",
            "test_relative_return",
            "test_max_dd",
        },
        "candidates",
    )
    _required_columns(
        selected_by_fold,
        {
            "fold",
            "selected_candidate",
            "selected_relative_return",
            "selected_max_dd",
            "selected_candidate_gate",
        },
        "selected_by_fold",
    )

    rows: list[dict[str, Any]] = []
    for _, selected in selected_by_fold.sort_values("fold").iterrows():
        fold = str(selected["fold"])
        fold_candidates = candidates[candidates["fold"].astype(str).eq(fold)].copy()
        if fold_candidates.empty:
            selected_relative = _as_float(selected.get("selected_relative_return"))
            selected_max_dd = _as_float(selected.get("selected_max_dd"))
            rows.append(
                {
                    "fold": fold,
                    "selected_candidate": str(selected["selected_candidate"]),
                    "selected_candidate_gate": False,
                    "failure_type": "missing_candidate_metrics",
                    "selected_relative_return_pp": _rounded(selected_relative),
                    "selected_relative_gap_pp": _rounded(max(float(target_relative_return_pp) - selected_relative, 0.0)),
                    "selected_max_dd_pct": _rounded(selected_max_dd),
                    "selected_drawdown_gap_pct": _rounded(max(selected_max_dd - float(target_max_dd_pct), 0.0)),
                    "selected_test_return_pct": 0.0,
                    "selected_buyhold_return_pct": 0.0,
                    "selected_return_vs_buyhold_gap_pp": 0.0,
                    "best_relative_candidate": "",
                    "best_relative_return_pp": 0.0,
                    "best_relative_gap_pp": _rounded(float(target_relative_return_pp)),
                    "best_relative_max_dd_pct": 0.0,
                    "best_relative_max_dd_gap_pct": 0.0,
                    "best_relative_test_return_pct": 0.0,
                    "candidate_gate_pass_count": 0,
                    "checkpoint_selection_exhausted": True,
                    "next_training_target": "regenerate_candidate_metrics",
                }
            )
            continue

        selected_candidate = str(selected["selected_candidate"])
        selected_rows = fold_candidates[fold_candidates["candidate"].astype(str).eq(selected_candidate)]
        selected_metrics = selected_rows.iloc[0] if not selected_rows.empty else fold_candidates.iloc[0]
        best_relative = fold_candidates.sort_values("test_relative_return", ascending=False).iloc[0]
        gate_candidates = fold_candidates[
            (fold_candidates["test_relative_return"].astype(float) >= float(target_relative_return_pp))
            & (fold_candidates["test_max_dd"].astype(float) < float(target_max_dd_pct))
        ]

        selected_gate = _as_bool(selected["selected_candidate_gate"])
        selected_relative = _as_float(selected["selected_relative_return"])
        selected_max_dd = _as_float(selected["selected_max_dd"])
        selected_return = _as_float(selected_metrics.get("test_return"))
        selected_buyhold_return = _as_float(selected_metrics.get("test_buy_and_hold_return"))
        selected_relative_gap = max(float(target_relative_return_pp) - selected_relative, 0.0)
        selected_drawdown_gap = max(selected_max_dd - float(target_max_dd_pct), 0.0)
        best_relative_return = _as_float(best_relative["test_relative_return"])
        best_relative_max_dd = _as_float(best_relative["test_max_dd"])
        best_relative_gap = max(float(target_relative_return_pp) - best_relative_return, 0.0)
        best_relative_drawdown_gap = max(best_relative_max_dd - float(target_max_dd_pct), 0.0)
        failure_type = _failure_type(
            fold=fold,
            selected_gate=selected_gate,
            selected_relative_gap_pp=selected_relative_gap,
            selected_drawdown_gap_pct=selected_drawdown_gap,
            selected_return=selected_return,
            selected_buyhold_return=selected_buyhold_return,
        )

        rows.append(
            {
                "fold": fold,
                "selected_candidate": selected_candidate,
                "selected_candidate_gate": bool(selected_gate),
                "failure_type": failure_type,
                "selected_relative_return_pp": _rounded(selected_relative),
                "selected_relative_gap_pp": _rounded(selected_relative_gap),
                "selected_max_dd_pct": _rounded(selected_max_dd),
                "selected_drawdown_gap_pct": _rounded(selected_drawdown_gap),
                "selected_test_return_pct": _rounded(selected_return),
                "selected_buyhold_return_pct": _rounded(selected_buyhold_return),
                "selected_return_vs_buyhold_gap_pp": _rounded(selected_return - selected_buyhold_return),
                "best_relative_candidate": str(best_relative["candidate"]),
                "best_relative_return_pp": _rounded(best_relative_return),
                "best_relative_gap_pp": _rounded(best_relative_gap),
                "best_relative_max_dd_pct": _rounded(best_relative_max_dd),
                "best_relative_max_dd_gap_pct": _rounded(best_relative_drawdown_gap),
                "best_relative_test_return_pct": _rounded(_as_float(best_relative.get("test_return"))),
                "candidate_gate_pass_count": int(len(gate_candidates)),
                "checkpoint_selection_exhausted": bool(gate_candidates.empty and failure_type != "pass"),
                "next_training_target": _next_training_target(failure_type),
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("selected_candidate_gate", "checkpoint_selection_exhausted"):
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: True if value else False).astype(object)
    return frame


def summarize_walkforward_failure_attribution(attribution: pd.DataFrame) -> dict[str, Any]:
    _required_columns(
        attribution,
        {
            "fold",
            "failure_type",
            "selected_candidate_gate",
            "checkpoint_selection_exhausted",
            "selected_relative_gap_pp",
            "selected_drawdown_gap_pct",
        },
        "attribution",
    )
    failed = attribution[attribution["failure_type"].astype(str).ne("pass")].copy()
    failed_folds = failed["fold"].astype(str).tolist()
    promotion_allowed = bool(failed.empty and attribution["selected_candidate_gate"].map(_as_bool).all())
    primary_failure_type = (
        Counter(failed["failure_type"].astype(str)).most_common(1)[0][0]
        if not failed.empty
        else "pass"
    )
    exhausted_failed_count = int(failed["checkpoint_selection_exhausted"].map(_as_bool).sum()) if not failed.empty else 0
    max_relative_gap = float(failed["selected_relative_gap_pp"].max()) if not failed.empty else 0.0
    max_drawdown_gap = float(failed["selected_drawdown_gap_pct"].max()) if not failed.empty else 0.0
    if primary_failure_type == "bull_underparticipation":
        next_mechanism = "rebuild_wf2020_bull_participation_target_before_full_retention_training"
    elif primary_failure_type == "drawdown_control":
        next_mechanism = "add_drawdown_controller_before_full_retention_training"
    elif primary_failure_type == "joint_tradeoff":
        next_mechanism = "train_walkforward_weighted_multi_objective_actor"
    elif primary_failure_type == "pass":
        next_mechanism = "evaluate_single_actor_ytd_retention_alignment"
    else:
        next_mechanism = "raise_relative_return_without_drawdown_regression"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promotion_allowed": promotion_allowed,
        "failed_folds": failed_folds,
        "failed_fold_count": int(len(failed)),
        "primary_failure_type": primary_failure_type,
        "checkpoint_selection_exhausted_failed_count": exhausted_failed_count,
        "max_selected_relative_gap_pp": _rounded(max_relative_gap),
        "max_selected_drawdown_gap_pct": _rounded(max_drawdown_gap),
        "next_mechanism": next_mechanism,
    }


def write_walkforward_failure_attribution(
    *,
    candidates_path: Path,
    selected_by_fold_path: Path,
    output_dir: Path,
    target_relative_return_pp: float = DEFAULT_TARGET_RELATIVE_RETURN_PP,
    target_max_dd_pct: float = DEFAULT_TARGET_MAX_DD_PCT,
) -> dict[str, Any]:
    candidates = pd.read_csv(candidates_path)
    selected_by_fold = pd.read_csv(selected_by_fold_path)
    attribution = classify_walkforward_checkpoint_failures(
        candidates,
        selected_by_fold,
        target_relative_return_pp=target_relative_return_pp,
        target_max_dd_pct=target_max_dd_pct,
    )
    summary = summarize_walkforward_failure_attribution(attribution)
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution_path = output_dir / "failure_attribution.csv"
    summary_path = output_dir / "summary.json"
    attribution.to_csv(attribution_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "attribution": attribution,
        "summary": summary,
        "attribution_path": attribution_path,
        "summary_path": summary_path,
    }


__all__ = [
    "DEFAULT_TARGET_MAX_DD_PCT",
    "DEFAULT_TARGET_RELATIVE_RETURN_PP",
    "classify_walkforward_checkpoint_failures",
    "summarize_walkforward_failure_attribution",
    "write_walkforward_failure_attribution",
]
