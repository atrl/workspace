#!/usr/bin/env python3
"""Create a durable checkpoint-selection review from a saved summary JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review checkpoint-selection evidence from a summary JSON")
    parser.add_argument("--summary", required=True, help="Path to *_summary.json")
    parser.add_argument("--output", required=True, help="Path to write the Markdown review")
    return parser.parse_args()


def gate_status(summary: dict[str, object]) -> list[str]:
    checks = summary.get("promotion", {}).get("fixed_split", {}).get("checks", {})
    rendered: list[str] = []
    for name in ("test_2022_max_dd", "test_2022_avg_pos", "test_proxy_sharpe"):
        payload = checks.get(name)
        if not payload:
            continue
        rendered.append(
            f"- `{name}`: {'pass' if payload['passed'] else 'fail'} "
            f"(actual={payload['actual']:.3f}, limit={payload['limit']:.3f})"
        )
    return rendered


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    history = payload.get("selection_history", [])
    valid = payload.get("valid", {})
    slices = payload.get("slices", {})
    test_proxy = payload.get("test_proxy", {})
    promotion = payload.get("promotion", {})
    stage_iters = payload.get("selection_score", {}).get("stage_iters")

    best_iter = None
    best_score = None
    if history:
        best_row = max(history, key=lambda row: row.get("score", float("-inf")))
        best_iter = best_row.get("iter")
        best_score = best_row.get("score")

    lines = [
        f"# Checkpoint Review: {summary_path.stem}",
        "",
        "## Summary",
        f"- profile: `{payload.get('profile')}`",
        f"- seed: `{payload.get('seed')}`",
        f"- staged training: `{payload.get('selection_score', {}).get('staged_training')}`",
        f"- stage iters: `{stage_iters}`",
        f"- best checkpoint iter: `{best_iter}`",
        f"- best score: `{best_score:.2f}`" if best_score is not None else "- best score: `n/a`",
        f"- selection checkpoints captured: `{len(history)}`",
        "",
        "## Valid",
        f"- return: `{valid.get('return', 0.0):.2f}%`",
        f"- sharpe: `{valid.get('sharpe', 0.0):.3f}`",
        f"- max_dd: `{valid.get('max_dd', 0.0):.2f}%`",
        f"- avg_pos: `{valid.get('avg_pos', 0.0):.3f}`",
        "",
        "## Fixed-Split",
        f"- test_2020: `{slices.get('test_2020', {}).get('return', 0.0):.2f}% / "
        f"Sharpe {slices.get('test_2020', {}).get('sharpe', 0.0):.3f} / "
        f"MaxDD {slices.get('test_2020', {}).get('max_dd', 0.0):.2f}% / "
        f"avg_pos {slices.get('test_2020', {}).get('avg_pos', 0.0):.3f}`",
        f"- test_2022: `{slices.get('test_2022', {}).get('return', 0.0):.2f}% / "
        f"Sharpe {slices.get('test_2022', {}).get('sharpe', 0.0):.3f} / "
        f"MaxDD {slices.get('test_2022', {}).get('max_dd', 0.0):.2f}% / "
        f"avg_pos {slices.get('test_2022', {}).get('avg_pos', 0.0):.3f}`",
        f"- test_2023_2024: `{slices.get('test_2023_2024', {}).get('return', 0.0):.2f}% / "
        f"Sharpe {slices.get('test_2023_2024', {}).get('sharpe', 0.0):.3f} / "
        f"MaxDD {slices.get('test_2023_2024', {}).get('max_dd', 0.0):.2f}% / "
        f"avg_pos {slices.get('test_2023_2024', {}).get('avg_pos', 0.0):.3f}`",
        f"- test_proxy: `{test_proxy.get('return', 0.0):.2f}% / "
        f"Sharpe {test_proxy.get('sharpe', 0.0):.3f} / "
        f"MaxDD {test_proxy.get('max_dd', 0.0):.2f}% / "
        f"avg_pos {test_proxy.get('avg_pos', 0.0):.3f}`",
        "",
        "## Gate Checks",
        *gate_status(payload),
        "",
        "## Selection History",
    ]
    if history:
        for row in history:
            lines.append(
                f"- iter `{row.get('iter')}`: score `{row.get('score', 0.0):.2f}`, "
                f"return `{row.get('return', 0.0):.2f}%`, sharpe `{row.get('sharpe', 0.0):.3f}`, "
                f"max_dd `{row.get('max_dd', 0.0):.2f}%`, avg_pos `{row.get('avg_pos', 0.0):.3f}`"
            )
    else:
        lines.append("- no selection history present")

    lines.extend(
        [
            "",
            "## Decision",
            (
                "- selection evidence is useful only if the best iter is earlier than max_iters "
                "and gate quality is preserved."
            ),
            (
                f"- promotion eligible: "
                f"`{promotion.get('eligible')}`"
            ),
        ]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
