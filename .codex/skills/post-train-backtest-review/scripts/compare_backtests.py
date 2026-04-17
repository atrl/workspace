#!/usr/bin/env python3
"""Compare two SOXL training summary artifacts and emit a Markdown review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BETTER_HIGHER = {"return", "sharpe", "score", "best_score"}
BETTER_LOWER = {"max_dd"}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_delta(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}"


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def delta(latest: dict[str, Any], previous: dict[str, Any], *keys: str) -> float | None:
    a = get_nested(latest, *keys)
    b = get_nested(previous, *keys)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def compare_metric(metric: str, latest_value: Any, previous_value: Any) -> tuple[str, float | None]:
    if latest_value is None or previous_value is None:
        return "unknown", None
    d = float(latest_value) - float(previous_value)
    if metric in BETTER_HIGHER:
        if d > 0:
            return "better", d
        if d < 0:
            return "worse", d
        return "flat", d
    if metric in BETTER_LOWER:
        if d < 0:
            return "better", d
        if d > 0:
            return "worse", d
        return "flat", d
    return "changed", d


def window_assessment(latest_slice: dict[str, Any], previous_slice: dict[str, Any]) -> tuple[str, list[str]]:
    votes = 0
    notes: list[str] = []
    for metric in ("return", "sharpe", "max_dd"):
        state, d = compare_metric(metric, latest_slice.get(metric), previous_slice.get(metric))
        if state == "better":
            votes += 1
            notes.append(f"{metric} improved ({fmt_delta(d)})")
        elif state == "worse":
            votes -= 1
            notes.append(f"{metric} worsened ({fmt_delta(d)})")
    if votes >= 2:
        return "good", notes
    if votes <= -2:
        return "bad", notes
    return "mixed", notes


def infer_mechanisms(latest: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    findings: list[str] = []

    d_2022_ret = delta(latest, previous, "slices", "test_2022", "return")
    d_2022_dd = delta(latest, previous, "slices", "test_2022", "max_dd")
    d_2022_pos = delta(latest, previous, "slices", "test_2022", "avg_pos")
    d_2324_ret = delta(latest, previous, "slices", "test_2023_2024", "return")
    d_2324_pos = delta(latest, previous, "slices", "test_2023_2024", "avg_pos")
    d_proxy_ret = delta(latest, previous, "test_proxy", "return")
    d_proxy_dd = delta(latest, previous, "test_proxy", "max_dd")

    if d_2022_ret is not None and d_2022_dd is not None and d_2022_pos is not None:
        if d_2022_ret < 0 and d_2022_dd > 0 and d_2022_pos > 0:
            findings.append("`2022` 回撤和收益同时恶化且平均仓位上升，说明新模型在压力窗口去风险不足。")
        if d_2022_ret > 0 and d_2022_dd < 0 and d_2022_pos < 0:
            findings.append("`2022` 改善伴随平均仓位下降，说明风险控制更积极，防守侧机制更有效。")

    if d_2324_ret is not None and d_2324_pos is not None:
        if d_2324_ret < 0 and d_2324_pos < 0:
            findings.append("`2023-2024` 收益下滑且平均仓位下降，说明新模型可能在顺风阶段过度保守。")
        if d_2324_ret > 0 and d_2324_pos > 0:
            findings.append("`2023-2024` 收益改善伴随更高持仓，说明新模型更愿意保留趋势暴露。")

    if d_proxy_ret is not None and d_proxy_dd is not None:
        if d_proxy_ret > 0 and d_proxy_dd > 0:
            findings.append("总体回报上升但总回撤加深，属于典型的收益换风险型改进，需要确认是否越过 promotion 风险边界。")
        if d_proxy_ret < 0 and d_proxy_dd < 0:
            findings.append("总体回报下降但回撤收敛，说明模型风格向保守漂移，需判断是否值得。")

    latest_hist = latest.get("selection_history") or []
    if latest_hist:
        best_iter = min(latest_hist, key=lambda item: item.get("iter", 0)) if len(latest_hist) == 1 else max(
            latest_hist, key=lambda item: float(item.get("score", float("-inf")))
        )
        last_iter = max(latest_hist, key=lambda item: item.get("iter", 0))
        best_score = float(best_iter.get("score", 0.0))
        last_score = float(last_iter.get("score", 0.0))
        if last_iter.get("iter", 0) > best_iter.get("iter", 0) and last_score + 1e-9 < best_score:
            findings.append(
                f"最新 run 的最佳 checkpoint 出现在 iter {best_iter.get('iter')}，之后分数回落到 iter {last_iter.get('iter')}，应优先考虑 early stopping 或更稳健的 checkpoint 选择。"
            )

    if not findings:
        findings.append("仅从 summary 指标看，没有单一明确机制；需要补充交易明细、特征归因或 walk-forward 结果。")

    return findings


def compare_features(latest_features: dict[str, Any], previous_features: dict[str, Any]) -> list[str]:
    findings: list[str] = []

    def top_names(blob: dict[str, Any], key: str) -> list[str]:
        rows = blob.get(key) or []
        return [row.get("feature", "unknown") for row in rows[:5] if isinstance(row, dict)]

    for key, label in (("top_shap", "SHAP"), ("top_ic", "IC")):
        latest_top = top_names(latest_features, key)
        previous_top = top_names(previous_features, key)
        if not latest_top or not previous_top:
            continue
        added = [name for name in latest_top if name not in previous_top]
        removed = [name for name in previous_top if name not in latest_top]
        shared = [name for name in latest_top if name in previous_top]
        findings.append(
            f"{label} Top5 overlap={len(shared)}/5; newly emphasized: {', '.join(added) if added else 'none'}; de-emphasized: {', '.join(removed) if removed else 'none'}."
        )

    return findings


def summarize_cost_audit(audit: dict[str, Any]) -> list[str]:
    if audit.get("status") == "failed":
        return [
            f"成本审计未完成：{audit.get('reason')} ({audit.get('message', 'unknown error')}).",
            "这不是性能结论，而是 harness / 可重放性问题；下一轮应先修复 artifact 与当前特征集的兼容性。",
        ]
    scenarios = audit.get("scenarios") or []
    if len(scenarios) < 2:
        return ["成本审计结果不足，无法判断摩擦鲁棒性。"]

    base = scenarios[0]
    findings: list[str] = []
    for row in scenarios[1:]:
        ret_delta = float(row.get("return", 0.0)) - float(base.get("return", 0.0))
        sharpe_delta = float(row.get("sharpe", 0.0)) - float(base.get("sharpe", 0.0))
        findings.append(
            f"{row.get('name')}: return Δ={fmt_delta(ret_delta)} sharpe Δ={fmt_delta(sharpe_delta, 3)} "
            f"under costs {fmt_num(row.get('open_cost'), 4)}/{fmt_num(row.get('close_cost'), 4)}."
        )

    worst = min(
        scenarios[1:],
        key=lambda row: float(row.get("return", float("inf"))) - float(base.get("return", 0.0)),
    )
    worst_ret_delta = float(worst.get("return", 0.0)) - float(base.get("return", 0.0))
    if worst_ret_delta < -20.0:
        findings.append("成本上调后收益塌陷明显，说明策略对摩擦敏感，下一轮应优先压换手或提高交易门槛。")
    elif worst_ret_delta < -10.0:
        findings.append("成本上调后收益回落较明显，建议把交易频率与成本鲁棒性纳入下一轮优化目标。")
    else:
        findings.append("成本上调后表现仍较稳，当前问题更像是风格暴露或 checkpoint 选择，而不是纯摩擦问题。")
    return findings


def build_report(args: argparse.Namespace) -> str:
    latest = load_json(args.latest)
    previous = load_json(args.previous)
    latest_label = args.latest_label or Path(args.latest).stem
    previous_label = args.previous_label or Path(args.previous).stem

    lines: list[str] = []
    lines.append(f"# Post-Train Backtest Review: {latest_label} vs {previous_label}")
    lines.append("")
    lines.append("## Summary")

    proxy_ret_state, proxy_ret_delta = compare_metric(
        "return", get_nested(latest, "test_proxy", "return"), get_nested(previous, "test_proxy", "return")
    )
    proxy_dd_state, proxy_dd_delta = compare_metric(
        "max_dd", get_nested(latest, "test_proxy", "max_dd"), get_nested(previous, "test_proxy", "max_dd")
    )
    lines.append(
        f"- Test proxy return: {fmt_num(get_nested(previous, 'test_proxy', 'return'))}% -> {fmt_num(get_nested(latest, 'test_proxy', 'return'))}% ({fmt_delta(proxy_ret_delta)}) [{proxy_ret_state}]"
    )
    lines.append(
        f"- Test proxy maxDD: {fmt_num(get_nested(previous, 'test_proxy', 'max_dd'))}% -> {fmt_num(get_nested(latest, 'test_proxy', 'max_dd'))}% ({fmt_delta(proxy_dd_delta)}) [{proxy_dd_state}]"
    )
    lines.append(
        f"- Validation score: {fmt_num(previous.get('best_score'))} -> {fmt_num(latest.get('best_score'))} ({fmt_delta(delta(latest, previous, 'best_score'))})"
    )
    lines.append(
        f"- Promotion eligible: {get_nested(previous, 'promotion', 'eligible')} -> {get_nested(latest, 'promotion', 'eligible')}"
    )
    lines.append("")

    lines.append("## Window Review")
    for name in ("test_2020", "test_2022", "test_2023_2024"):
        latest_slice = get_nested(latest, "slices", name) or {}
        previous_slice = get_nested(previous, "slices", name) or {}
        status, notes = window_assessment(latest_slice, previous_slice)
        lines.append(f"- {name}: {status}")
        lines.append(
            f"  return {fmt_num(previous_slice.get('return'))}% -> {fmt_num(latest_slice.get('return'))}% | "
            f"sharpe {fmt_num(previous_slice.get('sharpe'))} -> {fmt_num(latest_slice.get('sharpe'))} | "
            f"maxDD {fmt_num(previous_slice.get('max_dd'))}% -> {fmt_num(latest_slice.get('max_dd'))}% | "
            f"avg_pos {fmt_num(previous_slice.get('avg_pos'), 3)} -> {fmt_num(latest_slice.get('avg_pos'), 3)}"
        )
        lines.append(f"  evidence: {'; '.join(notes) if notes else 'n/a'}")
    lines.append("")

    lines.append("## Mechanism Hypotheses")
    for finding in infer_mechanisms(latest, previous):
        lines.append(f"- {finding}")
    lines.append("")

    lines.append("## Gate Checks")
    prev_checks = get_nested(previous, "promotion", "fixed_split", "checks") or {}
    latest_checks = get_nested(latest, "promotion", "fixed_split", "checks") or {}
    for name in sorted(set(prev_checks) | set(latest_checks)):
        prev_check = prev_checks.get(name, {})
        latest_check = latest_checks.get(name, {})
        lines.append(
            f"- {name}: {prev_check.get('passed')} ({fmt_num(prev_check.get('actual'))} / limit {fmt_num(prev_check.get('limit'))})"
            f" -> {latest_check.get('passed')} ({fmt_num(latest_check.get('actual'))} / limit {fmt_num(latest_check.get('limit'))})"
        )
    lines.append("")

    if args.latest_features and args.previous_features:
        latest_features = load_json(args.latest_features)
        previous_features = load_json(args.previous_features)
        lines.append("## Feature Shift")
        feature_findings = compare_features(latest_features, previous_features)
        if feature_findings:
            for finding in feature_findings:
                lines.append(f"- {finding}")
        else:
            lines.append("- Feature summaries were provided but did not contain comparable top lists.")
        lines.append("")

    if args.latest_cost_audit:
        lines.append("## Quant Reality Checks")
        for finding in summarize_cost_audit(load_json(args.latest_cost_audit)):
            lines.append(f"- {finding}")
        lines.append("")

    lines.append("## Recommended Next Actions")
    d_2022_dd = delta(latest, previous, "slices", "test_2022", "max_dd")
    d_2324_ret = delta(latest, previous, "slices", "test_2023_2024", "return")
    d_valid = delta(latest, previous, "best_score")
    if d_2022_dd is not None and d_2022_dd > 0:
        lines.append("- 优先检查压力窗口风险约束：drawdown cap、stress routing、危机态 floor，以及 `2022` 平均仓位。")
    if d_2324_ret is not None and d_2324_ret < 0:
        lines.append("- 优先检查是否过度防守：bull floor、趋势锚点特征权重、卖出阈值、持仓上限曲线。")
    if d_valid is not None and d_valid > 0 and proxy_ret_delta is not None and proxy_ret_delta < 0:
        lines.append("- validation 变强但 OOS 变弱，优先复核 checkpoint 选择和 early stopping，而不是继续加特征。")
    if len(lines) > 0 and lines[-1] == "## Recommended Next Actions":
        lines.append("- 需要更多证据后再决定下一步；建议补充 walk-forward、交易明细或 feature summary。")

    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- latest: `{args.latest}`")
    lines.append(f"- previous: `{args.previous}`")
    if args.latest_features:
        lines.append(f"- latest_features: `{args.latest_features}`")
    if args.previous_features:
        lines.append(f"- previous_features: `{args.previous_features}`")
    if args.latest_cost_audit:
        lines.append(f"- latest_cost_audit: `{args.latest_cost_audit}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two SOXL summary JSON files and emit a Markdown review.")
    parser.add_argument("--latest", required=True, help="Path to the latest summary JSON")
    parser.add_argument("--previous", required=True, help="Path to the previous/baseline summary JSON")
    parser.add_argument("--latest-label", help="Display label for the latest run")
    parser.add_argument("--previous-label", help="Display label for the previous run")
    parser.add_argument("--latest-features", help="Optional path to latest feature summary JSON")
    parser.add_argument("--previous-features", help="Optional path to previous feature summary JSON")
    parser.add_argument("--latest-cost-audit", help="Optional path to latest cost sensitivity audit JSON")
    parser.add_argument("--output", required=True, help="Path to output Markdown report")
    args = parser.parse_args()

    report = build_report(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
