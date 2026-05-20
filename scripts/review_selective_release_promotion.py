#!/usr/bin/env python3
from __future__ import annotations

"""Build a promotion-review card for the retained selective-release cap."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.policy.execution_overlays import POST_RELEASE_CAP_VARIANT
from src.soxl_quant.runtime import latest_model_path, promotion_gate_status
from src.soxl_quant.settings import load_profile


DEFAULT_WALKFORWARD = ROOT / "runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.csv"
DEFAULT_FIXED = ROOT / "runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_results.csv"
DEFAULT_OUTPUT_DIR = ROOT / "runtime/execution/card/selective_release_post_release_cap_v1"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row(rows: list[dict[str, Any]], **filters: str) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == value for key, value in filters.items()):
            return row
    label = ", ".join(f"{key}={value}" for key, value in filters.items())
    raise KeyError(f"missing row: {label}")


def _float(value: Any) -> float:
    return float(value)


def _int(value: Any) -> int:
    return int(float(value))


def _metric_delta(variant_row: dict[str, Any], baseline_row: dict[str, Any], metric: str) -> float:
    return _float(variant_row[metric]) - _float(baseline_row[metric])


def _check(passed: bool, detail: str, **values: Any) -> dict[str, Any]:
    payload = {"passed": bool(passed), "detail": detail}
    payload.update(values)
    return payload


def _fixed_actor_noop_check(
    fixed_rows: list[dict[str, Any]],
    *,
    variant: str,
    tolerance: float,
) -> dict[str, Any]:
    windows = sorted({str(row["window"]) for row in fixed_rows})
    deltas: dict[str, dict[str, float]] = {}
    for window in windows:
        try:
            baseline = _row(fixed_rows, window=window, variant="baseline")
            candidate = _row(fixed_rows, window=window, variant=variant)
        except KeyError:
            continue
        deltas[window] = {
            "return": _metric_delta(candidate, baseline, "return"),
            "max_dd": _metric_delta(candidate, baseline, "max_dd"),
            "trades": _metric_delta(candidate, baseline, "trades"),
        }
    passed = bool(deltas) and all(
        abs(delta) <= tolerance
        for metrics in deltas.values()
        for delta in metrics.values()
    )
    return _check(
        passed,
        "fixed full-window actor overlay grid is unchanged",
        windows=deltas,
    )


def build_promotion_review(
    walk_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    *,
    variant: str = POST_RELEASE_CAP_VARIANT,
    tolerance: float = 1e-6,
    runtime_profile_bound: bool = False,
    runtime_profile_auto_select_allowed: bool = False,
    model_promotion_gate_passed: bool = False,
    model_promotion_reason: str = "",
) -> dict[str, Any]:
    wf_2022_base = _row(walk_rows, actor_window="wf_2022", variant="baseline")
    wf_2022_variant = _row(walk_rows, actor_window="wf_2022", variant=variant)
    wf_2024_base = _row(walk_rows, actor_window="wf_2023_2024", variant="baseline")
    wf_2024_variant = _row(walk_rows, actor_window="wf_2023_2024", variant=variant)

    wf_2022_return_delta = _metric_delta(wf_2022_variant, wf_2022_base, "return")
    wf_2022_maxdd_delta = _metric_delta(wf_2022_variant, wf_2022_base, "max_dd")
    wf_2024_return_delta = _metric_delta(wf_2024_variant, wf_2024_base, "return")
    wf_2024_maxdd_delta = _metric_delta(wf_2024_variant, wf_2024_base, "max_dd")

    checks = {
        "wf_2022_unchanged": _check(
            abs(wf_2022_return_delta) <= tolerance
            and abs(wf_2022_maxdd_delta) <= tolerance
            and _int(wf_2022_variant.get("changed_days", 0)) == 0,
            "defensive walk-forward window remains unchanged",
            return_delta=round(wf_2022_return_delta, 6),
            max_dd_delta=round(wf_2022_maxdd_delta, 6),
            changed_days=_int(wf_2022_variant.get("changed_days", 0)),
        ),
        "wf_2023_2024_return_improved": _check(
            wf_2024_return_delta > 0.0,
            "recovery walk-forward return improves versus baseline",
            baseline_return=_float(wf_2024_base["return"]),
            variant_return=_float(wf_2024_variant["return"]),
            delta=round(wf_2024_return_delta, 6),
        ),
        "wf_2023_2024_maxdd_improved": _check(
            wf_2024_maxdd_delta < 0.0,
            "recovery walk-forward MaxDD improves versus baseline",
            baseline_max_dd=_float(wf_2024_base["max_dd"]),
            variant_max_dd=_float(wf_2024_variant["max_dd"]),
            delta=round(wf_2024_maxdd_delta, 6),
        ),
        "fixed_actor_noop": _fixed_actor_noop_check(
            fixed_rows,
            variant=variant,
            tolerance=tolerance,
        ),
    }
    accepted_for_review = all(item["passed"] for item in checks.values())
    checks["runtime_profile_bound"] = _check(
        runtime_profile_bound,
        (
            "profile binds the overlay to runtime execution config"
            if runtime_profile_bound
            else "overlay is not profile-bound production runtime behavior"
        ),
    )
    runtime_auto_select_allowed = (
        accepted_for_review
        and runtime_profile_bound
        and runtime_profile_auto_select_allowed
        and model_promotion_gate_passed
    )
    if not runtime_profile_bound:
        runtime_auto_select_reason = "overlay is not profile-bound production runtime behavior"
    elif not runtime_profile_auto_select_allowed and not model_promotion_gate_passed:
        runtime_auto_select_reason = (
            "profile binds overlay but keeps auto_select_allowed=false and the base actor has not passed the model promotion gate"
        )
    elif not runtime_profile_auto_select_allowed:
        runtime_auto_select_reason = "profile binds overlay but keeps auto_select_allowed=false"
    elif not model_promotion_gate_passed:
        runtime_auto_select_reason = f"base actor has not passed the model promotion gate: {model_promotion_reason}"
    else:
        runtime_auto_select_reason = "promotion review, profile flag, and model gate all allow auto-selection"
    return {
        "variant": variant,
        "decision": "promotion_review_candidate" if accepted_for_review else "blocked",
        "runtime_auto_select_allowed": runtime_auto_select_allowed,
        "runtime_auto_select_reason": runtime_auto_select_reason,
        "runtime_profile_auto_select_allowed": runtime_profile_auto_select_allowed,
        "model_promotion_gate_passed": model_promotion_gate_passed,
        "model_promotion_reason": model_promotion_reason,
        "checks": checks,
    }


def _markdown(review: dict[str, Any]) -> str:
    checks = review["checks"]
    lines = [
        "# Selective Release Post-Release Cap Promotion Review",
        "",
        f"- Variant: `{review['variant']}`",
        f"- Decision: `{review['decision']}`",
        f"- Runtime auto-select allowed: `{review['runtime_auto_select_allowed']}`",
        f"- Runtime auto-select reason: {review['runtime_auto_select_reason']}",
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for name, payload in checks.items():
        lines.append(f"| `{name}` | `{payload['passed']}` | {payload['detail']} |")
    lines.extend(["", "## Readout", ""])
    if review["runtime_auto_select_allowed"]:
        lines.extend(
            [
                "- This card is approved for scheduler auto-selection.",
                "- Keep the profile flag and model-promotion gate passing; either one should block auto-selection again if revoked.",
            ]
        )
    else:
        lines.extend(
            [
                "- This card packages the walk-forward candidate for promotion review only.",
                "- It must not be used by scheduler auto-selection until the profile flag is explicitly approved and the model-promotion gate remains passing.",
            ]
        )
    return "\n".join(lines) + "\n"


def runtime_review_context(profile_name: str, *, variant: str) -> dict[str, Any]:
    profile = load_profile(profile_name)
    runtime_profile_bound = (
        profile.execution_overlay.enabled
        and profile.execution_overlay.name == variant
    )
    try:
        model_promotion_gate_passed, model_promotion_reason = promotion_gate_status(
            latest_model_path(profile.name)
        )
    except FileNotFoundError as exc:
        model_promotion_gate_passed = False
        model_promotion_reason = str(exc)
    return {
        "runtime_profile_bound": bool(runtime_profile_bound),
        "runtime_profile_auto_select_allowed": bool(profile.execution_overlay.auto_select_allowed),
        "model_promotion_gate_passed": bool(model_promotion_gate_passed),
        "model_promotion_reason": model_promotion_reason,
    }


def write_review(
    review: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "promotion_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "promotion_review.md").write_text(_markdown(review), encoding="utf-8")


def relative_output_label(output_dir: Path) -> Path:
    return output_dir.resolve().relative_to(ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walkforward", type=Path, default=DEFAULT_WALKFORWARD)
    parser.add_argument("--fixed", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default=POST_RELEASE_CAP_VARIANT)
    parser.add_argument("--profile", default="balanced_rule_residual_v22")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review = build_promotion_review(
        _read_csv(args.walkforward),
        _read_csv(args.fixed),
        variant=args.variant,
        **runtime_review_context(args.profile, variant=args.variant),
    )
    write_review(review, args.output_dir)
    print(f"[selective_release_promotion] wrote {relative_output_label(args.output_dir)}")
    return 0 if review["decision"] == "promotion_review_candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
