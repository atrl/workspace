#!/usr/bin/env python3
from __future__ import annotations

"""Audit backtest/model replay paths against the reimplemented SOXL rule policy.

The cleanup rule is intentionally narrow:

- keep the new framework reimplementation as the primary executable rule policy
- keep any YTD replay that still beats that policy on the common window
- protect production runtime paths and runtime components even when their narrow
  YTD replay return is lower
- prune only explicitly identified duplicate ablation executables while keeping
  notes and summaries as idea evidence
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

PRIMARY_PROFILE = "soxl_backtest_policy_reimplementation_v1"
PRIMARY_BASELINE_WINDOW = "ytd_2026"
COMPARISON_WINDOW = "ytd_2026_common"
V22_CONTINUED_PROFILE = "balanced_rule_residual_v22_continued_retention_v1"

DEFAULT_BASELINE_METRICS_PATH = (
    ROOT
    / "runtime/production/card/soxl_backtest_policy_reimplementation_v1/legacy_comparison_metrics.csv"
)
DEFAULT_COMPARISON_METRICS_PATH = (
    ROOT / "runtime/production/card/soxl_backtest_jan_june_windows_v1/all_model_window_metrics.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "runtime/production/card/underperforming_backtest_cleanup_v1"

PRODUCTION_PROTECTED_PROFILES = {
    "v19reg_v22_budget_usage_bull_guard_v1",
}
RUNTIME_COMPONENT_PROTECTED_PROFILES = {
    "balanced_rule_residual_v22",
}

IDEAS_TO_KEEP = {
    PRIMARY_PROFILE: (
        "Fresh framework rule-policy port of the non-qlib soxl_backtest panic-buy, "
        "profit-taking, and post-sale re-entry state machine."
    ),
    "smoke": "High-return canonical replay remains useful as an upside participation reference.",
    "balanced_macro_v1": "Macro feature family remains useful because the common-window YTD replay beats the primary rule policy.",
    "balanced_rule_residual_v22": (
        "Do not use as a standalone main-line replay versus the new rule policy on this window; "
        "keep as canonical runtime component and fast leg for the budget-usage composer."
    ),
    V22_CONTINUED_PROFILE: (
        "Continuation-from-v22 idea failed: extra retained checkpoints did not improve the usable v22 path, "
        "worsened stress risk, and now trails the primary rule policy on the YTD common window."
    ),
    "v19reg_v22_budget_usage_bull_guard_v1": (
        "Production scheduler composer remains a protected execution/risk-control path; compare separately from "
        "research return replays because the narrow 2026 YTD result reflects its defensive runtime posture."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit underperforming SOXL backtest artifacts.")
    parser.add_argument("--apply", action="store_true", help="delete paths explicitly marked safe to prune")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS_PATH)
    parser.add_argument("--comparison-metrics", type=Path, default=DEFAULT_COMPARISON_METRICS_PATH)
    return parser.parse_args()


def _relative_path(root: Path, path: Path) -> Path:
    return path.resolve().relative_to(root.resolve())


def load_primary_return_pct(baseline_metrics_path: Path) -> float:
    frame = pd.read_csv(baseline_metrics_path)
    matches = frame[frame["window"].eq(PRIMARY_BASELINE_WINDOW)]
    if matches.empty:
        raise ValueError(f"{baseline_metrics_path} does not contain {PRIMARY_BASELINE_WINDOW}")
    return float(matches.iloc[0]["new_return_pct"])


def candidate_delete_paths(root: Path, profile: str) -> list[Path]:
    """Return repo-relative paths that are executable artifacts for a pruned idea.

    Notes, reviews, and summary JSON are intentionally retained as the idea record.
    """

    if profile != V22_CONTINUED_PROFILE:
        return []
    profile_dir = root / "runtime/model/card" / profile
    if not profile_dir.exists():
        return []
    paths: list[Path] = []
    profile_yaml = profile_dir / "profile.yaml"
    if profile_yaml.exists():
        paths.append(_relative_path(root, profile_yaml))
    for actor_path in sorted(profile_dir.glob("*.pt")):
        paths.append(_relative_path(root, actor_path))
    return paths


def load_registry_statuses(root: Path) -> dict[str, str]:
    registry_path = root / "runtime/harness/card/registry/profile.yaml"
    if not registry_path.exists():
        return {}
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return {str(item["name"]): str(item.get("status", "")) for item in payload.get("profiles", [])}


def _decision_for_row(
    root: Path,
    row: pd.Series,
    primary_return_pct: float,
    registry_statuses: dict[str, str],
) -> dict[str, Any]:
    profile = str(row["profile"])
    status = registry_statuses.get(profile, str(row.get("status", "")))
    model_kind = str(row.get("model_kind", ""))
    return_pct = float(row["model_return_pct"])
    underperforms = return_pct < primary_return_pct

    if profile in PRODUCTION_PROTECTED_PROFILES or status == "production_composer" or model_kind == "runtime_composer":
        decision = "protect_production_runtime"
        cleanup_action = "none"
        protected_reason = "production scheduler/runtime path; not removed by return-only cleanup"
    elif profile in RUNTIME_COMPONENT_PROTECTED_PROFILES:
        decision = "protect_runtime_component"
        cleanup_action = "none"
        protected_reason = "canonical runtime component used by the protected composer"
    elif underperforms:
        decision = "archive_idea_only"
        cleanup_action = "review_manually"
        protected_reason = ""
        if profile == V22_CONTINUED_PROFILE:
            cleanup_action = "delete_profile_and_actor_checkpoints"
    else:
        decision = "keep_executable"
        cleanup_action = "none"
        protected_reason = "common-window YTD return is not worse than the primary rule policy"

    delete_paths = candidate_delete_paths(root, profile) if cleanup_action.startswith("delete_") else []
    return {
        "profile": profile,
        "status": status,
        "model_kind": model_kind,
        "window": COMPARISON_WINDOW,
        "ytd_return_pct": return_pct,
        "primary_return_pct": primary_return_pct,
        "return_delta_vs_primary_pct": return_pct - primary_return_pct,
        "decision": decision,
        "cleanup_action": cleanup_action,
        "protected_reason": protected_reason,
        "idea_to_keep": IDEAS_TO_KEEP.get(profile, "Retain existing notebook/notes summary as idea evidence."),
        "source_path": str(row.get("source_path", "")),
        "delete_paths": delete_paths,
    }


def build_cleanup_decisions(
    *,
    root: Path = ROOT,
    baseline_metrics_path: Path = DEFAULT_BASELINE_METRICS_PATH,
    comparison_metrics_path: Path = DEFAULT_COMPARISON_METRICS_PATH,
) -> list[dict[str, Any]]:
    primary_return_pct = load_primary_return_pct(baseline_metrics_path)
    comparison = pd.read_csv(comparison_metrics_path)
    rows = comparison[comparison["window"].eq(COMPARISON_WINDOW)].copy()
    if rows.empty:
        raise ValueError(f"{comparison_metrics_path} does not contain {COMPARISON_WINDOW}")
    rows = rows.sort_values("model_return_pct", ascending=False)
    registry_statuses = load_registry_statuses(root)

    decisions: list[dict[str, Any]] = [
        {
            "profile": PRIMARY_PROFILE,
            "status": "primary_reimplementation",
            "model_kind": "rule_policy",
            "window": PRIMARY_BASELINE_WINDOW,
            "ytd_return_pct": primary_return_pct,
            "primary_return_pct": primary_return_pct,
            "return_delta_vs_primary_pct": 0.0,
            "decision": "keep_primary",
            "cleanup_action": "none",
            "protected_reason": "validated to match the legacy non-qlib soxl_backtest trade timing and return",
            "idea_to_keep": IDEAS_TO_KEEP[PRIMARY_PROFILE],
            "source_path": (
                "runtime/production/card/soxl_backtest_policy_reimplementation_v1/"
                "legacy_comparison_metrics.csv"
            ),
            "delete_paths": [],
        }
    ]
    decisions.extend(
        _decision_for_row(root, row, primary_return_pct, registry_statuses) for _, row in rows.iterrows()
    )
    return decisions


def _dataframe_for_csv(decisions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in decisions:
        row = dict(item)
        row["delete_paths"] = ";".join(str(path) for path in row["delete_paths"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_idea_retention(output_dir: Path, decisions: list[dict[str, Any]]) -> Path:
    path = output_dir / "retained_ideas.md"
    lines = [
        "# Underperforming Backtest Cleanup Ideas",
        "",
        "Primary benchmark: `soxl_backtest_policy_reimplementation_v1` on the 2026 YTD common window.",
        "",
    ]
    for item in decisions:
        if item["decision"] not in {
            "archive_idea_only",
            "protect_runtime_component",
            "protect_production_runtime",
        }:
            continue
        lines.extend(
            [
                f"## {item['profile']}",
                "",
                f"- Decision: `{item['decision']}`",
                f"- YTD return: `{item['ytd_return_pct']:.6f}%`",
                f"- Delta vs primary: `{item['return_delta_vs_primary_pct']:.6f}pp`",
                f"- Keep idea: {item['idea_to_keep']}",
                f"- Cleanup action: `{item['cleanup_action']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def apply_cleanup(root: Path, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    deleted: list[str] = []
    missing: list[str] = []
    freed_bytes = 0
    for item in decisions:
        for rel_path in item["delete_paths"]:
            path = root / rel_path
            if not path.exists():
                missing.append(str(rel_path))
                continue
            if not path.is_file():
                raise ValueError(f"cleanup path is not a file: {rel_path}")
            freed_bytes += path.stat().st_size
            path.unlink()
            deleted.append(str(rel_path))
    return {
        "deleted": deleted,
        "missing": missing,
        "deleted_count": len(deleted),
        "freed_bytes": freed_bytes,
    }


def write_outputs(
    *,
    root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    baseline_metrics_path: Path = DEFAULT_BASELINE_METRICS_PATH,
    comparison_metrics_path: Path = DEFAULT_COMPARISON_METRICS_PATH,
    apply: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = build_cleanup_decisions(
        root=root,
        baseline_metrics_path=baseline_metrics_path,
        comparison_metrics_path=comparison_metrics_path,
    )
    decisions_csv = output_dir / "cleanup_decisions.csv"
    _dataframe_for_csv(decisions).to_csv(decisions_csv, index=False)

    delete_rows = []
    for item in decisions:
        for rel_path in item["delete_paths"]:
            delete_rows.append(
                {
                    "profile": item["profile"],
                    "decision": item["decision"],
                    "cleanup_action": item["cleanup_action"],
                    "path": str(rel_path),
                }
            )
    delete_manifest = output_dir / "delete_manifest.csv"
    pd.DataFrame(delete_rows, columns=["profile", "decision", "cleanup_action", "path"]).to_csv(
        delete_manifest, index=False
    )
    ideas_path = write_idea_retention(output_dir, decisions)

    applied = {"deleted": [], "missing": [], "deleted_count": 0, "freed_bytes": 0}
    if apply:
        applied = apply_cleanup(root, decisions)
        (output_dir / "applied_cleanup.json").write_text(
            json.dumps(applied, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_profile": PRIMARY_PROFILE,
        "primary_return_pct": decisions[0]["primary_return_pct"],
        "comparison_window": COMPARISON_WINDOW,
        "decision_counts": _dataframe_for_csv(decisions)["decision"].value_counts().to_dict(),
        "cleanup_action_counts": _dataframe_for_csv(decisions)["cleanup_action"].value_counts().to_dict(),
        "applied": applied,
        "outputs": {
            "cleanup_decisions_csv": str(_relative_path(root, decisions_csv)),
            "delete_manifest_csv": str(_relative_path(root, delete_manifest)),
            "retained_ideas_md": str(_relative_path(root, ideas_path)),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    baseline_metrics = args.baseline_metrics
    if not baseline_metrics.is_absolute():
        baseline_metrics = ROOT / baseline_metrics
    comparison_metrics = args.comparison_metrics
    if not comparison_metrics.is_absolute():
        comparison_metrics = ROOT / comparison_metrics
    summary = write_outputs(
        root=ROOT,
        output_dir=output_dir,
        baseline_metrics_path=baseline_metrics,
        comparison_metrics_path=comparison_metrics,
        apply=args.apply,
    )
    print(
        "[underperforming-backtests] "
        f"primary_return={summary['primary_return_pct']:.6f}% "
        f"decisions={summary['decision_counts']} "
        f"deleted={summary['applied']['deleted_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
