from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.soxl_quant.config import StrategyConfig


def resolve_local_checkpoint_path(path_value: str | Path, *, source_model_card: Path) -> Path:
    """Resolve stale persisted actor paths against the current workspace card."""

    raw = Path(str(path_value))
    if raw.exists():
        return raw
    local = Path(source_model_card) / raw.name
    return local if local.exists() else raw


def apply_short_training_patch(
    source_profile: StrategyConfig,
    *,
    training_profile_name: str,
    reward_patch: Mapping[str, Any],
    checkpoint_path: str | Path,
    max_iters: int = 4,
    stage_iters: int = 2,
) -> StrategyConfig:
    """Return an in-memory diagnostic training profile without mutating source."""

    reward_payload = asdict(source_profile.reward)
    unknown = sorted(str(key) for key in reward_patch if str(key) not in reward_payload)
    if unknown:
        raise ValueError(f"unknown reward patch field(s): {unknown}")
    reward_payload.update(dict(reward_patch))
    reward = replace(source_profile.reward, **reward_payload)
    selection_score = replace(
        source_profile.selection_score,
        staged_training=True,
        stage_iters=int(stage_iters),
    )
    return replace(
        source_profile,
        name=str(training_profile_name),
        reward=reward,
        selection_score=selection_score,
        checkpoint_path=str(checkpoint_path),
        max_iters=int(max_iters),
    )


def _summary_path(source_model_card: Path, source_profile_name: str, fold: str) -> Path:
    return source_model_card / f"soxl_{source_profile_name}_{fold}_summary.json"


def _best_checkpoint_from_summary(path: Path) -> str:
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = payload.get("retained_checkpoints") or []
    if not isinstance(checkpoints, list) or not checkpoints:
        return ""
    ranked = sorted(
        (row for row in checkpoints if isinstance(row, Mapping)),
        key=lambda row: (float(row.get("score", float("-inf")) or float("-inf")), int(row.get("iter", -1) or -1)),
        reverse=True,
    )
    return str(ranked[0].get("path") or "") if ranked else ""


def _fallback_checkpoint(source_model_card: Path, source_profile_name: str, fold: str) -> Path:
    patterns = [
        f"soxl_{source_profile_name}_{fold}_checkpoint_iter*_actor.pt",
        f"soxl_{source_profile_name}_{fold}_actor.pt",
    ]
    for pattern in patterns:
        matches = sorted(source_model_card.glob(pattern))
        if matches:
            return matches[-1]
    return source_model_card / f"soxl_{source_profile_name}_{fold}_actor.pt"


def build_short_foldlocal_training_plan(
    *,
    source_profile: StrategyConfig,
    source_model_card: Path,
    training_profile_name: str,
    reward_patch: Mapping[str, Any],
    required_folds: Sequence[str],
    max_iters: int = 4,
    stage_iters: int = 2,
) -> pd.DataFrame:
    """Build the diagnostic fold-local training plan before running PPO."""

    rows: list[dict[str, Any]] = []
    for fold in required_folds:
        summary_path = _summary_path(Path(source_model_card), source_profile.name, str(fold))
        checkpoint = _best_checkpoint_from_summary(summary_path)
        if checkpoint:
            source_checkpoint = resolve_local_checkpoint_path(checkpoint, source_model_card=Path(source_model_card))
        else:
            source_checkpoint = _fallback_checkpoint(Path(source_model_card), source_profile.name, str(fold))
        rows.append(
            {
                "source_profile": source_profile.name,
                "training_profile": str(training_profile_name),
                "fold": str(fold),
                "artifact_tag": str(fold),
                "source_summary_path": str(summary_path),
                "source_checkpoint_path": str(source_checkpoint),
                "source_checkpoint_exists": bool(source_checkpoint.exists()),
                "max_iters": int(max_iters),
                "stage_iters": int(stage_iters),
                "reward_patch_json": json.dumps(dict(reward_patch), sort_keys=True),
                "promotion_allowed": False,
                "actor_retention_allowed": False,
                "screen_role": "diagnostic_short_foldlocal_training",
            }
        )
    return pd.DataFrame(rows)


def summarize_short_foldlocal_training_plan(plan: pd.DataFrame) -> dict[str, Any]:
    missing = []
    if not plan.empty and "source_checkpoint_exists" in plan.columns:
        missing = plan.loc[~plan["source_checkpoint_exists"].fillna(False).astype(bool), "fold"].astype(str).tolist()
    training_allowed = bool(not plan.empty and not missing)
    return {
        "decision": (
            "short_foldlocal_training_plan_ready_no_promotion"
            if training_allowed
            else "short_foldlocal_training_plan_blocked_missing_sources"
        ),
        "training_allowed": training_allowed,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "fold_count": int(len(plan)),
        "missing_source_folds": missing,
        "next_mechanism": (
            "run_short_foldlocal_ppo_screen"
            if training_allowed
            else "repair_checkpoint_source_manifest_before_training"
        ),
    }


def render_short_foldlocal_training_plan(plan: pd.DataFrame, summary: Mapping[str, Any]) -> str:
    lines = [
        "# Short Fold-Local Training Plan",
        "",
        "## Decision",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- training_allowed: `{summary.get('training_allowed')}`",
        f"- promotion_allowed: `{summary.get('promotion_allowed')}`",
        f"- next_mechanism: `{summary.get('next_mechanism', '')}`",
        "",
        "## Fold Sources",
        "",
        "| fold | source checkpoint | exists | max iters | stage iters |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    if plan.empty:
        lines.append("| none | none | false | 0 | 0 |")
    else:
        for row in plan.itertuples(index=False):
            lines.append(
                "| "
                f"{row.fold} | `{Path(str(row.source_checkpoint_path)).name}` | "
                f"{bool(row.source_checkpoint_exists)} | {int(row.max_iters)} | {int(row.stage_iters)} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This plan prepares a diagnostic short fold-local training screen only.",
            "- It does not permit promotion or actor retention; official walk-forward gates remain required.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "apply_short_training_patch",
    "build_short_foldlocal_training_plan",
    "render_short_foldlocal_training_plan",
    "resolve_local_checkpoint_path",
    "summarize_short_foldlocal_training_plan",
]
