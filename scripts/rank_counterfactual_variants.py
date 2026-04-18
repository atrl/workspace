#!/usr/bin/env python3
"""Rank fixed-actor counterfactual variants by promotion-gate quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank fixed-actor counterfactual variants")
    parser.add_argument("--input", required=True, help="Path to counterfactual JSON")
    parser.add_argument("--output", required=True, help="Path to write Markdown ranking")
    return parser.parse_args()


def score_variant(name: str, payload: dict[str, object]) -> tuple[int, float, float, float, str]:
    proxy = payload["test_proxy"]
    test_2022 = payload["slices"]["test_2022"]
    gate_passes = int(test_2022["max_dd"] <= 35.0) + int(test_2022["avg_pos"] <= 0.35) + int(proxy["sharpe"] >= 0.75)
    return (
        gate_passes,
        -test_2022["max_dd"],
        -test_2022["avg_pos"],
        proxy["sharpe"],
        name,
    )


def gate_text(proxy: dict[str, float], test_2022: dict[str, float]) -> str:
    statuses = [
        f"2022_maxdd={'pass' if test_2022['max_dd'] <= 35.0 else 'fail'}",
        f"2022_avg_pos={'pass' if test_2022['avg_pos'] <= 0.35 else 'fail'}",
        f"proxy_sharpe={'pass' if proxy['sharpe'] >= 0.75 else 'fail'}",
    ]
    return ", ".join(statuses)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    variants = payload.get("variants", {})
    ranked = sorted(
        ((name, row, score_variant(name, row)) for name, row in variants.items()),
        key=lambda item: item[2],
        reverse=True,
    )

    lines = [
        f"# Counterfactual Ranking: {input_path.stem}",
        "",
        f"- profile: `{payload.get('profile')}`",
        f"- model: `{payload.get('model')}`",
        f"- variants: `{len(variants)}`",
        "",
        "## Ranking",
    ]
    for idx, (name, row, sort_key) in enumerate(ranked, start=1):
        proxy = row["test_proxy"]
        test_2022 = row["slices"]["test_2022"]
        test_2023_2024 = row["slices"]["test_2023_2024"]
        lines.append(
            f"{idx}. `{name}`: gate_passes={sort_key[0]} | "
            f"2022 `{test_2022['return']:.2f}% / Sharpe {test_2022['sharpe']:.3f} / "
            f"MaxDD {test_2022['max_dd']:.2f}% / avg_pos {test_2022['avg_pos']:.3f}` | "
            f"proxy `Sharpe {proxy['sharpe']:.3f} / return {proxy['return']:.2f}% / maxDD {proxy['max_dd']:.2f}%` | "
            f"2023-2024 `{test_2023_2024['return']:.2f}% / Sharpe {test_2023_2024['sharpe']:.3f}` | "
            f"{gate_text(proxy, test_2022)}"
        )

    lines.extend(
        [
            "",
            "## Top Candidates",
        ]
    )
    for name, row, sort_key in ranked[:3]:
        lines.append(
            f"- `{name}` with gate_passes={sort_key[0]} and `2022 max_dd={row['slices']['test_2022']['max_dd']:.2f}%`, "
            f"`2022 avg_pos={row['slices']['test_2022']['avg_pos']:.3f}`, "
            f"`proxy sharpe={row['test_proxy']['sharpe']:.3f}`"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
