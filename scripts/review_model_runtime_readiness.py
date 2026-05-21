#!/usr/bin/env python3
from __future__ import annotations

"""Build a current model-runtime readiness review from retained evidence."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CURRENT_PROFILE = "v19reg_v22_budget_usage_bull_guard_v1"
DEFAULT_OUTPUT_DIR = ROOT / "runtime/model/card/model_runtime_readiness_v1"
DEFAULT_COMPOSER_REVIEW = (
    ROOT / "runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.json"
)
DEFAULT_PRODUCTION_GOAL_GATES = ROOT / "dashboard_data/production_goal_gates.json"
DEFAULT_V22_CONTINUATION_SUMMARY = (
    ROOT
    / "runtime/model/card/balanced_rule_residual_v22_continued_retention_v1"
    / "soxl_balanced_rule_residual_v22_continued_retention_v1_summary.json"
)
DEFAULT_DIRECT_SOXX_SUMMARY = (
    ROOT / "runtime/model/card/balanced_rule_residual_v22_soxx/soxl_balanced_rule_residual_v22_soxx_summary.json"
)
DEFAULT_SOXX_WALKFORWARD_REVIEW = (
    ROOT / "runtime/execution/card/soxx_relative_weakness_walkforward_v1/soxx_relative_walkforward_results.json"
)
REQUIRED_WINDOWS = ("wf_2020", "wf_2022", "wf_2023_2024")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(passed: bool, detail: str, **values: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"passed": bool(passed), "detail": detail}
    payload.update(values)
    return payload


def _gate(production_goal_gates: dict[str, Any], name: str) -> dict[str, Any]:
    gates = production_goal_gates.get("gates") or {}
    value = gates.get(name)
    return value if isinstance(value, dict) else {}


def _production_blockers(production_goal_gates: dict[str, Any]) -> list[str]:
    overall = production_goal_gates.get("overall") or {}
    issues = overall.get("issues") or []
    return [str(issue) for issue in issues]


def _failed_fixed_split_checks(summary: dict[str, Any]) -> list[str]:
    promotion = summary.get("promotion") or {}
    fixed_split = promotion.get("fixed_split") or {}
    checks = fixed_split.get("checks") or {}
    failed: list[str] = []
    for name, payload in checks.items():
        if isinstance(payload, dict) and payload.get("passed") is False:
            failed.append(str(name))
    return failed


def _blocked_summary_row(summary: dict[str, Any], fallback_profile: str, reason_prefix: str) -> dict[str, Any]:
    profile = str(summary.get("profile") or fallback_profile)
    promotion = summary.get("promotion") or {}
    failed_checks = _failed_fixed_split_checks(summary)
    if failed_checks:
        reason = f"{reason_prefix}: failed fixed-split checks {', '.join(failed_checks)}"
    elif promotion.get("eligible") is False:
        reason = f"{reason_prefix}: promotion sidecar is not eligible"
    else:
        reason = f"{reason_prefix}: retained as historical contrast only"
    return {
        "profile": profile,
        "decision": "blocked",
        "reason": reason,
        "promotion_eligible": bool(promotion.get("eligible", False)),
        "failed_checks": failed_checks,
    }


def _soxx_walkforward_row(payload: dict[str, Any]) -> dict[str, Any]:
    decision_payload = payload.get("decision") or {}
    if isinstance(decision_payload, dict):
        decision = str(decision_payload.get("decision") or "unknown")
        checks = decision_payload.get("checks") or {}
    else:
        decision = str(decision_payload)
        checks = {}
    failed_checks = [str(name) for name, value in checks.items() if value is False]
    if decision == "blocked" and failed_checks:
        reason = f"SOXX walk-forward overlay blocked: failed checks {', '.join(failed_checks)}"
    elif decision == "blocked":
        reason = "SOXX walk-forward overlay blocked by retained review"
    else:
        reason = "SOXX walk-forward overlay is not a runtime promotion input"
    return {
        "profile": "soxx_relative_weakness_walkforward_v1",
        "decision": decision,
        "reason": reason,
        "failed_checks": failed_checks,
    }


def _truthy_check(review: dict[str, Any], name: str) -> bool:
    checks = review.get("checks") or {}
    payload = checks.get(name)
    return bool(isinstance(payload, dict) and payload.get("passed", False))


def build_model_runtime_readiness_review(
    *,
    current_profile: str,
    composer_review: dict[str, Any],
    production_goal_gates: dict[str, Any],
    v22_continuation_summary: dict[str, Any],
    direct_soxx_summary: dict[str, Any],
    soxx_walkforward_review: dict[str, Any],
) -> dict[str, Any]:
    components = composer_review.get("components") or {}
    windows = composer_review.get("windows") or {}
    present_windows = [window for window in REQUIRED_WINDOWS if window in windows]

    composer_approved = (
        composer_review.get("profile") == current_profile
        and composer_review.get("decision") == "approved"
        and bool(composer_review.get("runtime_model_gate_allowed", False))
    )
    required_windows_present = _truthy_check(composer_review, "required_windows_present") and len(present_windows) == len(
        REQUIRED_WINDOWS
    )
    components_named = bool(components.get("slow_profile")) and bool(components.get("fast_profile"))

    model_gate = _gate(production_goal_gates, "model")
    model_gate_matches = model_gate.get("status") == "ok" and current_profile in str(model_gate.get("detail", ""))

    paper_gate = _gate(production_goal_gates, "ordinary_paper_proof")
    paper_detail = str(paper_gate.get("detail", ""))
    fresh_non_duplicate_paper = (
        paper_gate.get("status") == "ok"
        and current_profile in paper_detail
        and "duplicate_skipped" not in paper_detail
    )

    blocked_experiments = [
        _blocked_summary_row(
            v22_continuation_summary,
            "balanced_rule_residual_v22_continued_retention_v1",
            "continued v22 training is not a valid model-optimization path",
        ),
        _blocked_summary_row(
            direct_soxx_summary,
            "balanced_rule_residual_v22_soxx",
            "direct SOXX actor-feature retraining is not a valid path",
        ),
        _soxx_walkforward_row(soxx_walkforward_review),
    ]
    duplicate_paths_blocked = all(row.get("decision") in {"blocked", "drop"} for row in blocked_experiments)

    checks = {
        "composer_runtime_promotion_approved": _check(
            composer_approved,
            "current composer has an approved runtime promotion review",
            review_profile=composer_review.get("profile"),
            review_decision=composer_review.get("decision"),
        ),
        "composer_required_walkforward_windows_present": _check(
            required_windows_present,
            "composer review includes all standard walk-forward windows",
            present=present_windows,
            required=list(REQUIRED_WINDOWS),
        ),
        "composer_components_named": _check(
            components_named,
            "composer review names slow and fast component profiles",
            slow_profile=components.get("slow_profile"),
            fast_profile=components.get("fast_profile"),
        ),
        "production_model_gate_matches_current_profile": _check(
            model_gate_matches,
            "production goal gates point model readiness at the current runtime profile",
            gate_status=model_gate.get("status"),
            gate_detail=model_gate.get("detail"),
        ),
        "ordinary_paper_proof_is_fresh_non_duplicate": _check(
            fresh_non_duplicate_paper,
            "ordinary paper proof must be a fresh non-duplicate selected-model artifact",
            gate_status=paper_gate.get("status"),
            gate_detail=paper_detail,
        ),
        "duplicate_model_training_paths_blocked": _check(
            duplicate_paths_blocked,
            "known duplicate or negative model-training paths remain blocked",
            blocked_profiles=[row["profile"] for row in blocked_experiments],
        ),
    }

    runtime_model_ready = all(
        checks[name]["passed"]
        for name in (
            "composer_runtime_promotion_approved",
            "composer_required_walkforward_windows_present",
            "composer_components_named",
            "production_model_gate_matches_current_profile",
        )
    )
    production_blockers = _production_blockers(production_goal_gates)
    if not runtime_model_ready:
        decision = "blocked"
    elif fresh_non_duplicate_paper:
        decision = "ready_for_production_monitoring"
    else:
        decision = "ready_waiting_scheduled_paper_proof"

    should_open_training_branch = False
    if not runtime_model_ready and not duplicate_paths_blocked:
        should_open_training_branch = True

    if decision == "ready_waiting_scheduled_paper_proof":
        next_action = (
            "wait for the next natural scheduled post-close/non-dry paper artifact for the current composer; "
            "do not open another PPO or SOXX branch before that proof lands"
        )
    elif decision == "ready_for_production_monitoring":
        next_action = "monitor scheduled paper trading and Vercel deployment health"
    else:
        next_action = "repair the composer promotion review or production model gate before model optimization continues"

    return {
        "profile": current_profile,
        "decision": decision,
        "runtime_model_ready": runtime_model_ready,
        "should_open_training_branch": should_open_training_branch,
        "production_blockers": production_blockers,
        "checks": checks,
        "components": {
            "slow_profile": components.get("slow_profile"),
            "fast_profile": components.get("fast_profile"),
            "slow_model": components.get("slow_model"),
            "fast_model": components.get("fast_model"),
        },
        "window_deltas": {
            window: {
                "delta_return": windows[window].get("delta_return"),
                "delta_max_dd": windows[window].get("delta_max_dd"),
                "delta_avg_pos": windows[window].get("delta_avg_pos"),
            }
            for window in present_windows
        },
        "blocked_experiments": blocked_experiments,
        "next_action": next_action,
    }


def _markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Model Runtime Readiness Review",
        "",
        f"- Profile: `{review['profile']}`",
        f"- Decision: `{review['decision']}`",
        f"- Runtime model ready: `{review['runtime_model_ready']}`",
        f"- Open new training branch: `{review['should_open_training_branch']}`",
        f"- Next action: {review['next_action']}",
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for name, payload in review["checks"].items():
        lines.append(f"| `{name}` | `{payload['passed']}` | {payload['detail']} |")

    lines.extend(
        [
            "",
            "## Composer Components",
            "",
            f"- Slow profile: `{review['components'].get('slow_profile')}`",
            f"- Fast profile: `{review['components'].get('fast_profile')}`",
            "",
            "## Walk-Forward Deltas",
            "",
            "| window | delta_return | delta_max_dd | delta_avg_pos |",
            "|---|---:|---:|---:|",
        ]
    )
    for window, row in review["window_deltas"].items():
        lines.append(
            f"| `{window}` | {float(row['delta_return']):.2f}pp | "
            f"{float(row['delta_max_dd']):.2f}pp | {float(row['delta_avg_pos']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Blocked Model Paths",
            "",
            "| profile | decision | reason |",
            "|---|---|---|",
        ]
    )
    for row in review["blocked_experiments"]:
        lines.append(f"| `{row['profile']}` | `{row['decision']}` | {row['reason']} |")

    lines.extend(["", "## Production Blockers", ""])
    if review["production_blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in review["production_blockers"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_review(review: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_runtime_readiness_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "model_runtime_readiness_review.md").write_text(_markdown(review), encoding="utf-8")


def relative_output_label(output_dir: Path) -> Path:
    return output_dir.resolve().relative_to(ROOT)


def exit_code_for_review(review: dict[str, Any]) -> int:
    return 1 if review.get("decision") == "blocked" else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_CURRENT_PROFILE)
    parser.add_argument("--composer-review", type=Path, default=DEFAULT_COMPOSER_REVIEW)
    parser.add_argument("--production-goal-gates", type=Path, default=DEFAULT_PRODUCTION_GOAL_GATES)
    parser.add_argument("--v22-continuation-summary", type=Path, default=DEFAULT_V22_CONTINUATION_SUMMARY)
    parser.add_argument("--direct-soxx-summary", type=Path, default=DEFAULT_DIRECT_SOXX_SUMMARY)
    parser.add_argument("--soxx-walkforward-review", type=Path, default=DEFAULT_SOXX_WALKFORWARD_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review = build_model_runtime_readiness_review(
        current_profile=args.profile,
        composer_review=_read_json(args.composer_review),
        production_goal_gates=_read_json(args.production_goal_gates),
        v22_continuation_summary=_read_json(args.v22_continuation_summary),
        direct_soxx_summary=_read_json(args.direct_soxx_summary),
        soxx_walkforward_review=_read_json(args.soxx_walkforward_review),
    )
    write_review(review, args.output_dir)
    print(
        f"[model_runtime_readiness] wrote {relative_output_label(args.output_dir)} "
        f"decision={review['decision']}"
    )
    return exit_code_for_review(review)


if __name__ == "__main__":
    raise SystemExit(main())
