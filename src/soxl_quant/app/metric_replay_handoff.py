from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REQUIRED_FOLDS = ("wf_2020", "wf_2022", "wf_2023_2024")


@dataclass(frozen=True)
class MetricReplayHandoffAudit:
    summary: dict[str, Any]
    fold_metrics: pd.DataFrame


def _notebook_uses_metric_replay_fields(text: str) -> bool:
    required_tokens = (
        "backtest_profile(",
        "bnh_return",
        "buy_and_hold_return",
        "relative_outperformance",
        "relative_return",
    )
    return all(token in text for token in required_tokens)


def _required_folds(replay_summary: dict[str, Any]) -> list[str]:
    thresholds = replay_summary.get("reference_thresholds")
    if isinstance(thresholds, dict):
        folds = thresholds.get("required_wf_windows")
        if isinstance(folds, list | tuple) and folds:
            return [str(fold) for fold in folds]
    return list(DEFAULT_REQUIRED_FOLDS)


def _source_requires_metric_replay(source_summary: dict[str, Any]) -> bool:
    next_mechanism = str(source_summary.get("next_mechanism", ""))
    return bool(
        source_summary.get("mixed_metric_source_detected")
        and "backtest_profile" in next_mechanism
        and "bnh_return" in next_mechanism
        and "relative_outperformance" in next_mechanism
    )


def _fold_metrics(replay_summary: dict[str, Any], required_folds: list[str]) -> pd.DataFrame:
    selected = replay_summary.get("selected_by_fold")
    if not isinstance(selected, dict):
        return pd.DataFrame(
            columns=[
                "fold",
                "selected_candidate",
                "selected_candidate_gate",
                "selected_max_dd",
                "selected_relative_return",
            ]
        )
    rows = []
    for fold in required_folds:
        payload = selected.get(fold)
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "fold": str(fold),
                "selected_candidate": str(payload.get("selected_candidate", "")),
                "selected_candidate_gate": bool(payload.get("selected_candidate_gate", False)),
                "selected_max_dd": _float(payload.get("selected_max_dd")),
                "selected_relative_return": _float(payload.get("selected_relative_return")),
            }
        )
    return pd.DataFrame(rows)


def _float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable_selected_by_fold(replay_summary: dict[str, Any], required_folds: list[str]) -> dict[str, dict[str, Any]]:
    frame = _fold_metrics(replay_summary, required_folds)
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        fold = str(row.pop("fold"))
        out[fold] = row
    return out


def build_metric_replay_handoff_audit(
    *,
    source_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    replay_notebook_text: str,
    experiment: str = "cap10_metric_replay_handoff_audit_v1",
) -> MetricReplayHandoffAudit:
    required_folds = _required_folds(replay_summary)
    fold_metrics = _fold_metrics(replay_summary, required_folds)
    folds_evaluated = fold_metrics["fold"].tolist() if "fold" in fold_metrics else []
    missing_folds = [fold for fold in required_folds if fold not in set(folds_evaluated)]
    failed_folds = (
        fold_metrics.loc[~fold_metrics["selected_candidate_gate"].astype(bool), "fold"].astype(str).tolist()
        if not fold_metrics.empty
        else required_folds
    )
    metric_fields_present = _notebook_uses_metric_replay_fields(replay_notebook_text)
    source_requires_replay = _source_requires_metric_replay(source_summary)
    replay_fold_coverage = bool(replay_summary.get("required_fold_coverage")) and not missing_folds
    handoff_completed = bool(source_requires_replay and metric_fields_present and replay_fold_coverage)

    if handoff_completed:
        decision = "metric_replay_handoff_completed_no_promotion"
        next_mechanism = str(replay_summary.get("next_mechanism", ""))
    else:
        decision = "metric_replay_handoff_incomplete_no_promotion"
        next_mechanism = str(source_summary.get("next_mechanism", ""))

    summary = {
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "training_allowed_from_this_card": False,
        "role": "diagnose",
        "changed_layer": ["model_selection", "metric_reconciliation", "lineage"],
        "source_experiment": str(source_summary.get("experiment", "")),
        "direct_comparison_baseline": str(replay_summary.get("experiment", "")),
        "metric_replay_experiment": str(replay_summary.get("experiment", "")),
        "source_requires_metric_replay": source_requires_replay,
        "metric_replay_source_fields_present": metric_fields_present,
        "metric_replay_fold_coverage": replay_fold_coverage,
        "metric_replay_handoff_completed": handoff_completed,
        "required_folds": required_folds,
        "folds_evaluated": folds_evaluated,
        "missing_folds": missing_folds,
        "failed_folds": failed_folds,
        "selected_by_fold": _jsonable_selected_by_fold(replay_summary, required_folds),
        "reference_baseline": str(
            replay_summary.get("reference_baseline") or source_summary.get("reference_baseline") or ""
        ),
        "target_relative_return_pp": _float(
            replay_summary.get("target_relative_return_pp") or source_summary.get("target_relative_return_pp")
        ),
        "target_max_dd_pct": _float(
            replay_summary.get("target_max_dd_pct") or source_summary.get("target_max_dd_pct")
        ),
        "next_mechanism": next_mechanism,
    }
    return MetricReplayHandoffAudit(summary=summary, fold_metrics=fold_metrics)


def write_metric_replay_handoff_audit(
    *,
    source_card_dir: Path,
    replay_card_dir: Path,
    output_dir: Path,
) -> MetricReplayHandoffAudit:
    source_summary = _load_summary(source_card_dir / "summary.json")
    replay_summary = _load_summary(replay_card_dir / "summary.json")
    notebook_path = replay_card_dir / f"{replay_card_dir.name}.ipynb"
    audit = build_metric_replay_handoff_audit(
        source_summary=source_summary,
        replay_summary=replay_summary,
        replay_notebook_text=notebook_path.read_text(encoding="utf-8"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(audit.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selection.json").write_text(
        json.dumps(audit.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit.fold_metrics.to_csv(output_dir / "fold_metric_replay_handoff.csv", index=False)
    (output_dir / "conclusion.md").write_text(render_metric_replay_handoff_conclusion(audit.summary), encoding="utf-8")
    return audit


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else payload


def render_metric_replay_handoff_conclusion(summary: dict[str, Any]) -> str:
    folds = ", ".join(summary.get("folds_evaluated", []))
    failed = ", ".join(summary.get("failed_folds", [])) or "none"
    return (
        "# Cap10 Metric Replay Handoff Audit\n\n"
        f"- Decision: `{summary.get('decision')}`.\n"
        f"- Source diagnostic: `{summary.get('source_experiment')}`.\n"
        f"- Metric replay: `{summary.get('metric_replay_experiment')}`.\n"
        f"- Handoff completed: `{summary.get('metric_replay_handoff_completed')}`.\n"
        f"- Evaluated folds: `{folds}`.\n"
        f"- Failed folds after metric replay: `{failed}`.\n"
        f"- Next action: `{summary.get('next_mechanism')}`.\n\n"
        "No actor is promoted or retained by this audit.\n"
    )


__all__ = [
    "MetricReplayHandoffAudit",
    "build_metric_replay_handoff_audit",
    "render_metric_replay_handoff_conclusion",
    "write_metric_replay_handoff_audit",
]
