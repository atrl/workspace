from __future__ import annotations

"""Frontier review for the next SOXL model-retention training run."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


MODEL_SELECTION_CARD_ROOT = "runtime/model_selection/card"
RETENTION_GATE_SUMMARY_PATH = (
    "runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json"
)
DEFAULT_REQUIRED_WF_WINDOWS = ("wf_2020", "wf_2022", "wf_2023_2024")
DEFAULT_TARGET_RELATIVE_RETURN_PP = 30.0
EXCLUDED_CARD_NAMES = {"model_retention_walkforward_gate_v1", "model_retention_frontier_v1"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else payload


def _selection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection")
    return selection if isinstance(selection, dict) else payload


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def _string_list(values: Iterable[str]) -> str:
    return ",".join(sorted(str(value) for value in values if value))


def _coerce_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(key) for key in value] + [str(item) for item in value.values()]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _source_references(payload: dict[str, Any], known_experiments: set[str]) -> list[str]:
    summary = _summary_payload(payload)
    selection = _selection_payload(summary)
    values: list[str] = []
    for container in (summary, selection):
        for field in (
            "source_decisions",
            "direct_comparison_baseline",
            "source_card",
            "source_experiment",
            "source_selection",
            "source_bc_decision",
        ):
            values.extend(_coerce_text_values(container.get(field)))
    references: set[str] = set()
    for text in values:
        for experiment in known_experiments:
            if experiment and experiment in text:
                references.add(experiment)
    return sorted(references)


def _sidecar_manifest_references(card_dir: Path, known_experiments: set[str]) -> list[str]:
    references: set[str] = set()
    for path in sorted(card_dir.glob("*manifest*.csv")):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        for column in frame.columns:
            if column not in {"path", "source_path", "source", "source_label", "candidate", "mechanism", "variant"}:
                continue
            for value in frame[column].dropna().astype(str):
                for experiment in known_experiments:
                    if experiment and experiment in value:
                        references.add(experiment)
    return sorted(references)


def _selection_sidecar_references(card_dir: Path, known_experiments: set[str]) -> list[str]:
    path = card_dir / "selection.json"
    if not path.exists():
        return []
    try:
        payload = _load_json(path)
    except (json.JSONDecodeError, OSError):
        return []
    return _source_references(payload, known_experiments)


def _load_gate_summary(root: Path) -> dict[str, Any]:
    path = root / RETENTION_GATE_SUMMARY_PATH
    if not path.exists():
        return {
            "required_best_return_pct": None,
            "reference_return_pct": None,
            "max_walkforward_drawdown_pct": 20.0,
            "required_wf_windows": list(DEFAULT_REQUIRED_WF_WINDOWS),
        }
    return _summary_payload(_load_json(path))


def _wf_metrics_from_selection(summary: dict[str, Any], selection: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    max_dd: dict[str, float] = {}
    relative: dict[str, float] = {}

    wf_max_dd = selection.get("wf_max_dd_pct")
    if isinstance(wf_max_dd, dict):
        max_dd.update({str(key): float(value) for key, value in wf_max_dd.items() if _number(value) is not None})

    selected_by_fold = summary.get("selected_by_fold")
    if isinstance(selected_by_fold, dict):
        for fold, payload in selected_by_fold.items():
            if not isinstance(payload, dict):
                continue
            dd = _number(payload.get("selected_max_dd"))
            rel = _number(payload.get("selected_relative_return"))
            if dd is not None:
                max_dd[str(fold)] = dd
            if rel is not None:
                relative[str(fold)] = rel

    metrics = summary.get("metrics")
    if isinstance(metrics, list):
        for payload in metrics:
            if not isinstance(payload, dict):
                continue
            window = str(payload.get("window", ""))
            dd = _number(payload.get("max_dd_pct"))
            rel = _number(payload.get("relative_return_pct"))
            if window and dd is not None:
                max_dd.setdefault(window, dd)
            if window and rel is not None:
                relative.setdefault(window, rel)

    return max_dd, relative


def _primary_blocker(
    *,
    promotion_allowed: bool,
    training_contract_ready: bool,
    missing_windows: list[str],
    wf_drawdown_gap_pct: float | None,
    relative_return_gap_to_target_pp: float | None,
    ytd_gap_to_best_pct: float | None,
) -> str:
    if promotion_allowed:
        return "promotion_ready"
    if training_contract_ready:
        return "training_contract_ready"
    if missing_windows:
        return "missing_walkforward_evidence"
    if (wf_drawdown_gap_pct or 0.0) > 0 and (ytd_gap_to_best_pct or 0.0) > 0:
        return "joint_drawdown_return_gap"
    if (relative_return_gap_to_target_pp or 0.0) > 0:
        return "walkforward_relative_return_gap"
    if (wf_drawdown_gap_pct or 0.0) > 0:
        return "walkforward_drawdown_gap"
    if (ytd_gap_to_best_pct or 0.0) > 0:
        return "return_alignment_gap"
    return "diagnostic_no_promotion"


def _frontier_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    blocker_priority = {
        "promotion_ready": -1.0,
        "training_contract_ready": 0.0,
        "walkforward_relative_return_gap": 10.0,
        "joint_drawdown_return_gap": 20.0,
        "walkforward_drawdown_gap": 30.0,
        "return_alignment_gap": 40.0,
        "missing_walkforward_evidence": 50.0,
        "diagnostic_no_promotion": 60.0,
        "completed_handoff": 90.0,
    }
    gap = float(row.get("wf_drawdown_gap_pct") or 0.0)
    gap += float(row.get("relative_return_gap_to_target_pp") or 0.0) / 100.0
    gap += float(row.get("ytd_gap_to_best_pct") or 0.0) / 1000.0
    primary_blocker = str(row.get("primary_blocker"))
    needs_next_action = primary_blocker not in {"promotion_ready", "training_contract_ready"}
    action_penalty = 25.0 if needs_next_action and not str(row.get("next_action") or "").strip() else 0.0
    return (blocker_priority.get(primary_blocker, 99.0) + action_penalty + gap, str(row.get("experiment")))


def _row_from_summary(path: Path, payload: dict[str, Any], gate_summary: dict[str, Any]) -> dict[str, Any]:
    summary = _summary_payload(payload)
    selection = _selection_payload(summary)
    experiment = str(selection.get("experiment") or summary.get("experiment") or path.parent.name)
    decision = str(selection.get("decision") or summary.get("decision") or "")
    promotion_allowed = bool(selection.get("promotion_allowed", summary.get("promotion_allowed", False)))
    next_action = str(selection.get("next_mechanism") or summary.get("next_mechanism") or "")
    required_windows = [str(item) for item in gate_summary.get("required_wf_windows", DEFAULT_REQUIRED_WF_WINDOWS)]
    target_relative = float(selection.get("target_relative_return_pp") or summary.get("target_relative_return_pp") or DEFAULT_TARGET_RELATIVE_RETURN_PP)
    max_dd_threshold = float(gate_summary.get("max_walkforward_drawdown_pct") or 20.0)
    required_best = _number(selection.get("required_best_return_pct")) or _number(gate_summary.get("required_best_return_pct"))
    reference_return = (
        _number(selection.get("required_soxlbacktest_return_pct"))
        or _number(gate_summary.get("reference_return_pct"))
    )

    wf_max_dd, wf_relative = _wf_metrics_from_selection(summary, selection)
    present_windows = sorted(window for window in required_windows if window in wf_max_dd or window in wf_relative)
    missing_windows = [window for window in required_windows if window not in present_windows]
    failed_windows = sorted(
        window
        for window in present_windows
        if (
            (window in wf_max_dd and wf_max_dd[window] >= max_dd_threshold)
            or (window in wf_relative and wf_relative[window] < target_relative)
        )
    )

    ytd_return = _number(selection.get("ytd_return_pct") or summary.get("ytd_return_pct"))
    worst_dd = max(wf_max_dd.values()) if wf_max_dd else None
    min_relative = min(wf_relative.values()) if wf_relative else None
    wf_drawdown_gap = max(0.0, worst_dd - max_dd_threshold) if worst_dd is not None else None
    relative_gap = max(0.0, target_relative - min_relative) if min_relative is not None else None
    ytd_best_gap = max(0.0, required_best - ytd_return) if ytd_return is not None and required_best is not None else None
    ytd_reference_gap = (
        max(0.0, reference_return - ytd_return)
        if ytd_return is not None and reference_return is not None
        else None
    )
    training_contract_ready = bool(
        (selection.get("target_contract_gate") or summary.get("target_contract_gate"))
        and (selection.get("bc_readiness_gate") or summary.get("bc_readiness_gate"))
        and (selection.get("bc_screen_allowed_next") or summary.get("bc_screen_allowed_next") or next_action)
    )
    primary_blocker = _primary_blocker(
        promotion_allowed=promotion_allowed,
        training_contract_ready=training_contract_ready,
        missing_windows=missing_windows,
        wf_drawdown_gap_pct=wf_drawdown_gap,
        relative_return_gap_to_target_pp=relative_gap,
        ytd_gap_to_best_pct=ytd_best_gap,
    )
    return {
        "frontier_rank": None,
        "experiment": experiment,
        "decision": decision,
        "promotion_allowed": promotion_allowed,
        "reference_baseline": str(selection.get("reference_baseline") or summary.get("reference_baseline") or ""),
        "primary_blocker": primary_blocker,
        "next_action": next_action,
        "source_path": str(path),
        "handoff_completed_by": "",
        "wf_windows_present": _string_list(present_windows),
        "missing_wf_windows": _string_list(missing_windows),
        "failed_wf_windows": _string_list(failed_windows),
        "worst_wf_max_drawdown_pct": _round(worst_dd),
        "wf_drawdown_gap_pct": _round(wf_drawdown_gap),
        "min_wf_relative_return_pp": _round(min_relative),
        "relative_return_gap_to_target_pp": _round(relative_gap),
        "ytd_return_pct": _round(ytd_return),
        "ytd_gap_to_soxlbacktest_pct": _round(ytd_reference_gap),
        "ytd_gap_to_best_pct": _round(ytd_best_gap),
    }


def scan_model_retention_frontier(*, root: Path, card_names: Iterable[str] | None = None) -> dict[str, Any]:
    root = Path(root)
    gate_summary = _load_gate_summary(root)
    card_root = root / MODEL_SELECTION_CARD_ROOT
    selected_names = {str(name) for name in card_names} if card_names is not None else None
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(card_root.glob("*/summary.json")):
        if path.parent.name in EXCLUDED_CARD_NAMES:
            continue
        if selected_names is not None and path.parent.name not in selected_names:
            continue
        try:
            payloads.append((path, _load_json(path)))
        except (json.JSONDecodeError, OSError) as exc:
            payloads.append((path, {"experiment": path.parent.name, "load_error": str(exc)}))
    known_experiments = {
        str(_selection_payload(_summary_payload(payload)).get("experiment") or _summary_payload(payload).get("experiment") or path.parent.name)
        for path, payload in payloads
    }
    downstream_by_source: dict[str, list[str]] = {}
    for path, payload in payloads:
        downstream = str(
            _selection_payload(_summary_payload(payload)).get("experiment")
            or _summary_payload(payload).get("experiment")
            or path.parent.name
        )
        sources = set(_source_references(payload, known_experiments))
        sources.update(_sidecar_manifest_references(path.parent, known_experiments))
        sources.update(_selection_sidecar_references(path.parent, known_experiments))
        for source in sorted(sources):
            if source != downstream:
                downstream_by_source.setdefault(source, []).append(downstream)

    rows: list[dict[str, Any]] = []
    for path, payload in payloads:
        try:
            row = _row_from_summary(path.relative_to(root), payload, gate_summary)
            completed_by = sorted(set(downstream_by_source.get(str(row["experiment"]), [])))
            if completed_by:
                row["handoff_completed_by"] = _string_list(completed_by)
                if not bool(row["promotion_allowed"]):
                    row["primary_blocker"] = "completed_handoff"
            rows.append(row)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            rows.append(
                {
                    "frontier_rank": None,
                    "experiment": path.parent.name,
                    "decision": "",
                    "promotion_allowed": False,
                    "reference_baseline": "",
                    "primary_blocker": "unreadable_summary",
                    "next_action": "",
                    "source_path": str(path.relative_to(root)),
                    "handoff_completed_by": "",
                    "wf_windows_present": "",
                    "missing_wf_windows": "",
                    "failed_wf_windows": "",
                    "worst_wf_max_drawdown_pct": None,
                    "wf_drawdown_gap_pct": None,
                    "min_wf_relative_return_pp": None,
                    "relative_return_gap_to_target_pp": None,
                    "ytd_return_pct": None,
                    "ytd_gap_to_soxlbacktest_pct": None,
                    "ytd_gap_to_best_pct": None,
                    "error": str(exc),
                }
            )

    rows = sorted(rows, key=_frontier_sort_key)
    for idx, row in enumerate(rows, start=1):
        row["frontier_rank"] = idx

    recommended = next((row for row in rows if row["primary_blocker"] != "unreadable_summary"), None)
    return {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows_scanned": len(rows),
            "completed_handoff_count": sum(1 for row in rows if row.get("primary_blocker") == "completed_handoff"),
            "recommended_next_source": recommended.get("experiment") if recommended else None,
            "recommended_next_action": recommended.get("next_action") if recommended else None,
            "recommended_primary_blocker": recommended.get("primary_blocker") if recommended else None,
            "required_best_return_pct": gate_summary.get("required_best_return_pct"),
            "reference_return_pct": gate_summary.get("reference_return_pct"),
            "max_walkforward_drawdown_pct": gate_summary.get("max_walkforward_drawdown_pct"),
            "required_wf_windows": gate_summary.get("required_wf_windows", list(DEFAULT_REQUIRED_WF_WINDOWS)),
        },
        "rows": rows,
    }


def _csv_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _sentence_line(label: str, value: Any) -> str:
    text = str(value)
    suffix = "" if text.endswith((".", "!", "?")) else "."
    return f"- {label}: `{text}`{suffix}"


def write_retention_frontier_outputs(result: dict[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_csv_rows(result["rows"])).to_csv(output_dir / "retention_frontier.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    conclusion = [
        "# Model Retention Frontier",
        "",
        f"- Rows scanned: `{summary['rows_scanned']}`.",
        f"- Completed handoffs: `{summary.get('completed_handoff_count', 0)}`.",
        _sentence_line("Recommended next source", summary.get("recommended_next_source")),
        _sentence_line("Recommended next action", summary.get("recommended_next_action")),
        _sentence_line("Primary blocker", summary.get("recommended_primary_blocker")),
        f"- Retention gate: MaxDD `< {summary.get('max_walkforward_drawdown_pct')}` and YTD return floor `{summary.get('required_best_return_pct')}` versus SOXLBacktest reference `{summary.get('reference_return_pct')}`.",
        "",
        "This card ranks model-selection evidence for the next training run. It does not promote or retain actors.",
    ]
    (output_dir / "conclusion.md").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
