#!/usr/bin/env python3
"""Generate a trade-quality / Kelly audit report from repo artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct_share(part: float, whole: float, digits: int = 1) -> str:
    if not whole:
        return "n/a"
    return f"{part / whole * 100:.{digits}f}%"


def bucket_name(buckets: dict[str, Any], key: str) -> dict[str, Any]:
    return buckets.get(key) or {}


def render_report(summary: dict[str, Any], diagnosis: dict[str, Any], profile: str) -> str:
    trade = diagnosis.get("trade_analysis") or {}
    overall = trade.get("overall") or {}
    by_holding = trade.get("by_holding_period") or {}
    by_pos = trade.get("by_pos_change") or {}
    friction = trade.get("friction_estimate") or {}

    short_bucket = bucket_name(by_holding, "short_<3d")
    mid_bucket = bucket_name(by_holding, "mid_3_10d")
    long_bucket = bucket_name(by_holding, "long_10_30d")
    very_long_bucket = bucket_name(by_holding, "very_long_30d+")

    micro_bucket = bucket_name(by_pos, "micro_<5%")
    small_bucket = bucket_name(by_pos, "small_5_15%")
    large_bucket = bucket_name(by_pos, "large_15%+")

    total_trades = float(overall.get("total", 0) or 0)
    micro_trades = float(micro_bucket.get("total", 0) or 0)

    test_proxy = summary.get("test_proxy") or {}
    test_2022 = (summary.get("slices") or {}).get("test_2022") or {}
    promotion = summary.get("promotion") or {}
    fixed_split = promotion.get("fixed_split") or {}

    lines: list[str] = []
    lines.append(f"# Trade Quality Audit: {profile}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- profile: `{profile}`")
    lines.append(
        f"- portfolio-level retained result: `test_proxy = {fmt_num(test_proxy.get('return'))}% / "
        f"Sharpe {fmt_num(test_proxy.get('sharpe'), 3)} / MaxDD {fmt_num(test_proxy.get('max_dd'))}%`"
    )
    lines.append(
        f"- `test_2022 = {fmt_num(test_2022.get('return'))}% / Sharpe {fmt_num(test_2022.get('sharpe'), 3)} / "
        f"MaxDD {fmt_num(test_2022.get('max_dd'))}% / avg_pos {fmt_num(test_2022.get('avg_pos'), 3)}`"
    )
    lines.append(f"- fixed-split promotion gate: `{'pass' if fixed_split.get('eligible') else 'fail'}`")
    lines.append("")
    lines.append(
        "The actor remains usable at the portfolio level, but the trade-quality breakdown shows that "
        "short-horizon T behavior is not where the edge comes from."
    )
    lines.append("")
    lines.append("## Kelly By Holding Period")
    lines.append("")
    lines.append(
        f"- `short_<3d`: `{int(short_bucket.get('total', 0) or 0)}` trades | "
        f"`Kelly {fmt_num(short_bucket.get('kelly'), 4)}` | "
        f"`Expectancy {fmt_num(short_bucket.get('expectancy'))}%`"
    )
    lines.append(
        f"- `mid_3_10d`: `{int(mid_bucket.get('total', 0) or 0)}` trades | "
        f"`Kelly {fmt_num(mid_bucket.get('kelly'), 4)}` | "
        f"`Expectancy {fmt_num(mid_bucket.get('expectancy'))}%`"
    )
    lines.append(
        f"- `long_10_30d`: `{int(long_bucket.get('total', 0) or 0)}` trades | "
        f"`Kelly {fmt_num(long_bucket.get('kelly'), 4)}` | "
        f"`Expectancy {fmt_num(long_bucket.get('expectancy'))}%`"
    )
    lines.append(
        f"- `very_long_30d+`: `{int(very_long_bucket.get('total', 0) or 0)}` trades | "
        f"`Kelly {fmt_num(very_long_bucket.get('kelly'), 4)}` | "
        f"`Expectancy {fmt_num(very_long_bucket.get('expectancy'))}%`"
    )
    lines.append("")
    lines.append("## Kelly By Position Change")
    lines.append("")
    lines.append(
        f"- `micro_<5%`: `{int(micro_trades)}` trades (`{pct_share(micro_trades, total_trades)}` of all trades) | "
        f"`Kelly {fmt_num(micro_bucket.get('kelly'), 4)}` | "
        f"`Expectancy {fmt_num(micro_bucket.get('expectancy'))}%`"
    )
    lines.append(
        f"- `small_5_15%`: `{int(small_bucket.get('total', 0) or 0)}` trades | "
        f"`Kelly {fmt_num(small_bucket.get('kelly'), 4)}` | "
        f"`Expectancy {fmt_num(small_bucket.get('expectancy'))}%`"
    )
    lines.append(
        f"- `large_15%+`: `{int(large_bucket.get('total', 0) or 0)}` trades | "
        f"`Kelly {fmt_num(large_bucket.get('kelly'), 4)}` | "
        f"`Expectancy {fmt_num(large_bucket.get('expectancy'))}%`"
    )
    lines.append("")
    lines.append("## Friction And Interpretation")
    lines.append("")
    lines.append(
        f"- total trade actions: `{int(friction.get('total_trade_actions', 0) or 0)}`"
    )
    lines.append(
        f"- rough friction estimate: `{fmt_num(friction.get('friction_cost_pct'))}%`"
    )
    lines.append(
        f"- removing `micro_<5%` would save about `{fmt_num(friction.get('estimated_friction_saved_by_removing_micro'))}%` of friction"
    )
    lines.append("")
    lines.append("## What The Model Learned")
    lines.append("")
    lines.append(
        "The current reward is dominated by step-level portfolio return, turnover/smoothness cost, "
        "drawdown/downside/risk-budget control, and inactivity/opportunity-cost shaping."
    )
    lines.append(
        "That means the actor is mainly optimized for aggregate path quality, not for per-trade round-trip expectancy."
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        f"- `mid_3_10d` is the clearest negative-expectancy bucket: "
        f"`Kelly {fmt_num(mid_bucket.get('kelly'), 4)} / Expectancy {fmt_num(mid_bucket.get('expectancy'))}%`"
    )
    lines.append(
        f"- `small_5_15%` is also negative: "
        f"`Kelly {fmt_num(small_bucket.get('kelly'), 4)} / Expectancy {fmt_num(small_bucket.get('expectancy'))}%`"
    )
    lines.append(
        f"- `micro_<5%` is near-flat but too frequent: "
        f"`Kelly {fmt_num(micro_bucket.get('kelly'), 4)} / Expectancy {fmt_num(micro_bucket.get('expectancy'))}%`"
    )
    lines.append(
        "- the actor still has portfolio-level value, but its T-style rebalances are partly a side-effect "
        "of path control rather than a clean alpha source"
    )
    lines.append("")
    lines.append("## Next Action Order")
    lines.append("")
    lines.append("- raise the effective minimum rebalance threshold to suppress `micro_<5%` churn")
    lines.append("- add or tighten `trade_lock` / `cooldown` for `3-10d` churn")
    lines.append("- test fixed-actor low-micro replays before changing reward or architecture")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to *_summary.json")
    parser.add_argument("--diagnosis", required=True, help="Path to phase1_phase2_combined.json")
    parser.add_argument("--profile", required=True, help="Profile name for labeling")
    parser.add_argument("--output", required=True, help="Markdown output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = load_json(args.summary)
    diagnosis = load_json(args.diagnosis)
    report = render_report(summary, diagnosis, args.profile)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Trade-quality audit written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
