from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..config import PromotionGateConfig


def fixed_promotion_report(
    slices: Dict[str, Dict[str, float]],
    test_proxy: Dict[str, float],
    gate: PromotionGateConfig,
) -> Dict[str, object]:
    test_2022 = slices.get("test_2022", {})
    checks = {
        "test_2022_max_dd": {
            "passed": bool(test_2022) and float(test_2022.get("max_dd", float("inf"))) <= gate.max_test_2022_drawdown,
            "actual": float(test_2022.get("max_dd", float("inf"))),
            "limit": float(gate.max_test_2022_drawdown),
        },
        "test_2022_avg_pos": {
            "passed": bool(test_2022) and float(test_2022.get("avg_pos", float("inf"))) <= gate.max_test_2022_avg_pos,
            "actual": float(test_2022.get("avg_pos", float("inf"))),
            "limit": float(gate.max_test_2022_avg_pos),
        },
        "test_proxy_sharpe": {
            "passed": float(test_proxy.get("sharpe", float("-inf"))) >= gate.min_test_proxy_sharpe,
            "actual": float(test_proxy.get("sharpe", float("-inf"))),
            "limit": float(gate.min_test_proxy_sharpe),
        },
    }
    eligible = all(item["passed"] for item in checks.values())
    return {"eligible": eligible, "checks": checks}


def walk_forward_promotion_report(
    walk_forward_results: Dict[str, Dict[str, object]],
    gate: PromotionGateConfig,
) -> Dict[str, object]:
    wf_2023_2024 = walk_forward_results.get("wf_2023_2024", {})
    checks = {
        "wf_2023_2024_return": {
            "passed": bool(wf_2023_2024) and float(wf_2023_2024.get("return", float("-inf"))) >= gate.min_wf_2023_2024_return,
            "actual": float(wf_2023_2024.get("return", float("-inf"))),
            "limit": float(gate.min_wf_2023_2024_return),
        },
        "wf_2023_2024_sharpe": {
            "passed": bool(wf_2023_2024) and float(wf_2023_2024.get("sharpe", float("-inf"))) >= gate.min_wf_2023_2024_sharpe,
            "actual": float(wf_2023_2024.get("sharpe", float("-inf"))),
            "limit": float(gate.min_wf_2023_2024_sharpe),
        },
    }
    eligible = all(item["passed"] for item in checks.values())
    return {"eligible": eligible, "checks": checks}


def build_promotion_report(
    profile,
    slices: Dict[str, Dict[str, float]],
    test_proxy: Dict[str, float],
    walk_forward_results: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object]:
    fixed_report = fixed_promotion_report(slices, test_proxy, profile.promotion_gate)
    if walk_forward_results:
        walk_forward_report = walk_forward_promotion_report(walk_forward_results, profile.promotion_gate)
        walk_forward_evaluated = True
    else:
        walk_forward_report = {"eligible": False, "checks": {}, "status": "missing"}
        walk_forward_evaluated = False
    eligible = fixed_report["eligible"] and walk_forward_report["eligible"] if walk_forward_evaluated else False
    return {
        "eligible": eligible,
        "fixed_split": fixed_report,
        "walk_forward": walk_forward_report,
        "walk_forward_evaluated": walk_forward_evaluated,
    }


def persist_promotion_report(summary_path: Path, promotion_report: Dict[str, object]) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["promotion"] = promotion_report
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "build_promotion_report",
    "fixed_promotion_report",
    "persist_promotion_report",
    "walk_forward_promotion_report",
]
