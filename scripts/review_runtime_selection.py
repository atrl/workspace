#!/usr/bin/env python3
from __future__ import annotations

"""Build a runtime-selection review card for a profile-bound overlay candidate."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.promote import build_promotion_report, persist_promotion_report
from src.soxl_quant.runtime import auto_select_profile
from src.soxl_quant.settings import load_profile


DEFAULT_PROFILE = "balanced_rule_residual_v22"
DEFAULT_MODEL_SUMMARY = (
    ROOT
    / "runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json"
)
DEFAULT_OVERLAY_REVIEW = (
    ROOT / "runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "runtime/model/card/balanced_rule_residual_v22"


def _check(passed: bool, detail: str, **values: Any) -> dict[str, Any]:
    payload = {"passed": bool(passed), "detail": detail}
    payload.update(values)
    return payload


def _approval_detail(allowed: bool) -> str:
    if allowed:
        return "profile overlay is explicitly approved for runtime auto-selection"
    return "profile overlay is not approved for runtime auto-selection"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_walk_forward_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _read_json(path)
    results: dict[str, dict[str, Any]] = {}
    for window, row in payload.items():
        if not isinstance(row, dict):
            continue
        metrics = dict(row.get("main_backtest") or {})
        if not metrics:
            continue
        if "relative_outperformance" in row:
            metrics["relative_return"] = row["relative_outperformance"]
        results[str(window)] = metrics
    return results


def effective_model_promotion(
    *,
    profile_name: str,
    model_summary: dict[str, Any],
    walk_forward_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    promotion = model_summary.get("promotion") or {}
    if promotion.get("walk_forward_evaluated", False) or not walk_forward_results:
        return promotion
    profile = load_profile(profile_name)
    return build_promotion_report(
        profile,
        model_summary.get("slices") or {},
        model_summary.get("test_proxy") or {},
        walk_forward_results,
    )


def build_runtime_selection_review(
    *,
    profile_name: str,
    model_summary: dict[str, Any],
    model_promotion: dict[str, Any] | None = None,
    overlay_review: dict[str, Any],
    profile_overlay_auto_select_allowed: bool,
    current_auto_select_profile: str = "",
) -> dict[str, Any]:
    promotion = model_promotion or model_summary.get("promotion") or {}
    fixed_split = promotion.get("fixed_split") or {}
    overlay_checks = overlay_review.get("checks") or {}
    runtime_profile_bound = overlay_checks.get("runtime_profile_bound") or {}

    checks = {
        "model_fixed_split_eligible": _check(
            bool(fixed_split.get("eligible", False)),
            "base model fixed-split promotion gate is eligible",
        ),
        "model_walk_forward_evaluated": _check(
            bool(promotion.get("walk_forward_evaluated", False)),
            "base model walk-forward promotion gate is evaluated",
        ),
        "model_promotion_eligible": _check(
            bool(promotion.get("eligible", False)),
            "base model promotion sidecar is eligible",
        ),
        "overlay_review_candidate": _check(
            overlay_review.get("decision") == "promotion_review_candidate",
            "overlay promotion review remains a candidate",
            variant=overlay_review.get("variant", ""),
        ),
        "overlay_runtime_profile_bound": _check(
            bool(runtime_profile_bound.get("passed", False)),
            "overlay is bound in the runtime profile",
        ),
        "profile_overlay_auto_select_allowed": _check(
            bool(profile_overlay_auto_select_allowed),
            _approval_detail(bool(profile_overlay_auto_select_allowed)),
        ),
    }

    blocking_reasons: list[str] = []
    if not checks["model_fixed_split_eligible"]["passed"]:
        blocking_reasons.append("model fixed-split promotion gate is not eligible")
    if not checks["model_walk_forward_evaluated"]["passed"]:
        blocking_reasons.append("model walk-forward promotion gate not evaluated")
    if not checks["model_promotion_eligible"]["passed"]:
        blocking_reasons.append("model promotion sidecar is not eligible")
    if not checks["overlay_review_candidate"]["passed"]:
        blocking_reasons.append("overlay promotion review is not a candidate")
    if not checks["overlay_runtime_profile_bound"]["passed"]:
        blocking_reasons.append("overlay is not bound in the runtime profile")
    if not checks["profile_overlay_auto_select_allowed"]["passed"]:
        blocking_reasons.append("profile overlay auto_select_allowed=false")

    runtime_auto_select_allowed = not blocking_reasons
    return {
        "profile": profile_name,
        "decision": "approved" if runtime_auto_select_allowed else "blocked",
        "runtime_auto_select_allowed": runtime_auto_select_allowed,
        "current_auto_select_profile": current_auto_select_profile,
        "blocking_reasons": blocking_reasons,
        "checks": checks,
    }


def runtime_selection_context(profile_name: str, *, family: str = "balanced") -> dict[str, Any]:
    profile = load_profile(profile_name)
    try:
        current_profile, _ = auto_select_profile(family)
    except Exception as exc:
        current_profile = f"unavailable: {exc}"
    return {
        "profile_overlay_auto_select_allowed": bool(profile.execution_overlay.auto_select_allowed),
        "current_auto_select_profile": current_profile,
    }


def _markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Runtime Selection Review",
        "",
        f"- Profile: `{review['profile']}`",
        f"- Decision: `{review['decision']}`",
        f"- Runtime auto-select allowed: `{review['runtime_auto_select_allowed']}`",
        f"- Current auto-selected profile: `{review['current_auto_select_profile']}`",
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for name, payload in review["checks"].items():
        lines.append(f"| `{name}` | `{payload['passed']}` | {payload['detail']} |")
    lines.extend(["", "## Blocking Reasons", ""])
    if review["blocking_reasons"]:
        for reason in review["blocking_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_review(review: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime_selection_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime_selection_review.md").write_text(
        _markdown(review),
        encoding="utf-8",
    )


def persist_effective_model_promotion(summary_path: Path, promotion: dict[str, Any]) -> None:
    persist_promotion_report(summary_path, promotion)


def relative_output_label(output_dir: Path) -> Path:
    return output_dir.resolve().relative_to(ROOT)


def exit_code_for_review(review: dict[str, Any]) -> int:
    del review
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--model-summary", type=Path, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--overlay-review", type=Path, default=DEFAULT_OVERLAY_REVIEW)
    parser.add_argument(
        "--walk-forward-report",
        type=Path,
        default=None,
        help="Optional corrected main-backtest walk-forward report for the base model",
    )
    parser.add_argument(
        "--persist-model-promotion",
        action="store_true",
        help="Write the effective promotion report back to the model summary sidecar",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--family", default="balanced")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_walk_forward_report = args.output_dir / f"{args.profile}_report.json"
    walk_forward_report = args.walk_forward_report or default_walk_forward_report
    model_summary = _read_json(args.model_summary)
    walk_forward_results = load_walk_forward_results(walk_forward_report)
    model_promotion = effective_model_promotion(
        profile_name=args.profile,
        model_summary=model_summary,
        walk_forward_results=walk_forward_results,
    )
    if args.persist_model_promotion:
        persist_effective_model_promotion(args.model_summary, model_promotion)
        model_summary = _read_json(args.model_summary)
    review = build_runtime_selection_review(
        profile_name=args.profile,
        model_summary=model_summary,
        model_promotion=model_promotion,
        overlay_review=_read_json(args.overlay_review),
        **runtime_selection_context(args.profile, family=args.family),
    )
    write_review(review, args.output_dir)
    print(f"[runtime_selection_review] wrote {relative_output_label(args.output_dir)}")
    return exit_code_for_review(review)


if __name__ == "__main__":
    raise SystemExit(main())
