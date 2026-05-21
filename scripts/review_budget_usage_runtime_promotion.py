#!/usr/bin/env python3
from __future__ import annotations

"""Build a runtime-promotion review for a budget-usage composer candidate."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PROFILE = "v19reg_v22_budget_usage_bull_guard_v1"
DEFAULT_SLOW_PROFILE = "balanced_rule_residual_v19_reg"
DEFAULT_FAST_PROFILE = "balanced_rule_residual_v22"
DEFAULT_OUTPUT_DIR = ROOT / "runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1"
DEFAULT_WINDOW_PAYLOADS = {
    "wf_2020": DEFAULT_OUTPUT_DIR
    / "walkforward/wf_2020/v19reg_v22_budget_usage_bull_guard_v1_wf_2020.json",
    "wf_2022": DEFAULT_OUTPUT_DIR
    / "walkforward/wf_2022/v19reg_v22_budget_usage_bull_guard_v1_wf_2022.json",
    "wf_2023_2024": DEFAULT_OUTPUT_DIR
    / "walkforward/wf_2023_2024/v19reg_v22_budget_usage_bull_guard_v1_wf_2023_2024.json",
}
REQUIRED_WINDOWS = ("wf_2020", "wf_2022", "wf_2023_2024")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], label: str) -> dict[str, Any]:
    return dict(payload["variants"][label]["test_proxy"])


def _check(passed: bool, detail: str, **values: Any) -> dict[str, Any]:
    payload = {"passed": bool(passed), "detail": detail}
    payload.update(values)
    return payload


def _window_rows(window_payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for window, payload in window_payloads.items():
        composed_label = str(payload["composed_label"])
        fast_label = str(payload["fast_label"])
        candidate = _metric(payload, composed_label)
        fast = _metric(payload, fast_label)
        rows[window] = {
            "candidate": candidate,
            "fast": fast,
            "delta_return": float(candidate["return"]) - float(fast["return"]),
            "delta_max_dd": float(candidate["max_dd"]) - float(fast["max_dd"]),
            "delta_avg_pos": float(candidate["avg_pos"]) - float(fast["avg_pos"]),
        }
    return rows


def build_runtime_promotion_review(
    *,
    profile: str,
    slow_profile: str,
    fast_profile: str,
    slow_model: str,
    fast_model: str,
    window_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    present_windows = tuple(sorted(window_payloads))
    missing_windows = [window for window in REQUIRED_WINDOWS if window not in window_payloads]
    rows = _window_rows(window_payloads) if not missing_windows else {}
    bull_guard_enabled = bool(window_payloads) and all(
        bool((payload.get("config") or {}).get("bull_participation_guard", {}).get("enabled", False))
        for payload in window_payloads.values()
    )

    return_beats_fast = bool(rows) and all(row["delta_return"] > 0.0 for row in rows.values())
    maxdd_not_worse = bool(rows) and all(row["delta_max_dd"] <= 0.0 for row in rows.values())
    wf_2022 = rows.get("wf_2022", {})
    wf_2022_candidate = wf_2022.get("candidate", {})
    wf_2022_defense = bool(wf_2022) and (
        float(wf_2022_candidate.get("max_dd", float("inf"))) <= 30.0
        and float(wf_2022_candidate.get("avg_pos", float("inf"))) <= 0.20
        and float(wf_2022.get("delta_return", 0.0)) > 0.0
        and float(wf_2022.get("delta_max_dd", 0.0)) < 0.0
    )

    checks = {
        "required_windows_present": _check(
            not missing_windows,
            "all required walk-forward payloads are present",
            present=list(present_windows),
            missing=missing_windows,
        ),
        "bull_guard_enabled": _check(
            bull_guard_enabled,
            "price-derived bull guard is enabled in all payloads",
        ),
        "all_windows_return_beats_fast": _check(
            return_beats_fast,
            "candidate beats the fast v22 path return in every required window",
            windows={
                window: round(float(row["delta_return"]), 6)
                for window, row in rows.items()
            },
        ),
        "all_windows_maxdd_not_worse_than_fast": _check(
            maxdd_not_worse,
            "candidate MaxDD is not worse than the fast v22 path in every required window",
            windows={
                window: round(float(row["delta_max_dd"]), 6)
                for window, row in rows.items()
            },
        ),
        "wf_2022_defense_preserved": _check(
            wf_2022_defense,
            "2022 defense remains within runtime promotion limits",
            max_dd=float(wf_2022_candidate.get("max_dd", float("nan"))) if wf_2022_candidate else None,
            avg_pos=float(wf_2022_candidate.get("avg_pos", float("nan"))) if wf_2022_candidate else None,
            delta_return=float(wf_2022.get("delta_return", float("nan"))) if wf_2022 else None,
            delta_max_dd=float(wf_2022.get("delta_max_dd", float("nan"))) if wf_2022 else None,
        ),
    }
    blocking_reasons: list[str] = []
    if not checks["required_windows_present"]["passed"]:
        blocking_reasons.append("missing required walk-forward payloads")
    if not checks["bull_guard_enabled"]["passed"]:
        blocking_reasons.append("bull participation guard is not enabled in all payloads")
    if not checks["all_windows_return_beats_fast"]["passed"]:
        blocking_reasons.append("candidate does not beat fast path return in every required window")
    if not checks["all_windows_maxdd_not_worse_than_fast"]["passed"]:
        blocking_reasons.append("candidate MaxDD is worse than fast path in at least one required window")
    if not checks["wf_2022_defense_preserved"]["passed"]:
        blocking_reasons.append("2022 defense is not preserved")

    approved = not blocking_reasons
    return {
        "profile": profile,
        "decision": "approved" if approved else "blocked",
        "runtime_model_gate_allowed": approved,
        "components": {
            "slow_profile": slow_profile,
            "fast_profile": fast_profile,
            "slow_model": slow_model,
            "fast_model": fast_model,
        },
        "blocking_reasons": blocking_reasons,
        "checks": checks,
        "windows": rows,
    }


def _markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Budget-Usage Runtime Promotion Review",
        "",
        f"- Profile: `{review['profile']}`",
        f"- Decision: `{review['decision']}`",
        f"- Runtime model gate allowed: `{review['runtime_model_gate_allowed']}`",
        f"- Slow profile: `{review['components']['slow_profile']}`",
        f"- Fast profile: `{review['components']['fast_profile']}`",
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for name, payload in review["checks"].items():
        lines.append(f"| `{name}` | `{payload['passed']}` | {payload['detail']} |")
    lines.extend(["", "## Window Deltas", "", "| window | delta_return | delta_max_dd | delta_avg_pos |"])
    lines.append("|---|---:|---:|---:|")
    for window, row in review.get("windows", {}).items():
        lines.append(
            f"| `{window}` | {float(row['delta_return']):.2f}pp | "
            f"{float(row['delta_max_dd']):.2f}pp | {float(row['delta_avg_pos']):.3f} |"
        )
    lines.extend(["", "## Blocking Reasons", ""])
    if review["blocking_reasons"]:
        lines.extend(f"- {reason}" for reason in review["blocking_reasons"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_review(review: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime_promotion_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime_promotion_review.md").write_text(_markdown(review), encoding="utf-8")


def relative_output_label(output_dir: Path) -> Path:
    return output_dir.resolve().relative_to(ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--slow-profile", default=DEFAULT_SLOW_PROFILE)
    parser.add_argument("--fast-profile", default=DEFAULT_FAST_PROFILE)
    parser.add_argument("--slow-model", default="soxl_balanced_rule_residual_v19_reg_actor.pt")
    parser.add_argument("--fast-model", default="soxl_balanced_rule_residual_v22_actor.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for window, path in DEFAULT_WINDOW_PAYLOADS.items():
        parser.add_argument(f"--{window.replace('_', '-')}", type=Path, default=path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window_payloads = {
        "wf_2020": _read_json(args.wf_2020),
        "wf_2022": _read_json(args.wf_2022),
        "wf_2023_2024": _read_json(args.wf_2023_2024),
    }
    review = build_runtime_promotion_review(
        profile=args.profile,
        slow_profile=args.slow_profile,
        fast_profile=args.fast_profile,
        slow_model=args.slow_model,
        fast_model=args.fast_model,
        window_payloads=window_payloads,
    )
    write_review(review, args.output_dir)
    print(f"[budget_usage_runtime_promotion] wrote {relative_output_label(args.output_dir)}")
    return 0 if review["decision"] == "approved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
