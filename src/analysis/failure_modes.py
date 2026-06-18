from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


FAILURE_MODES_SCHEMA_VERSION = "analysis.failure_modes.v1"

FAILURE_MODE_COLUMNS = [
    "schema_version",
    "model",
    "baseline_model",
    "segment_type",
    "segment",
    "failure_mode",
    "severity_score",
    "relative_return_pct_delta",
    "stability_score_delta",
    "max_drawdown_pct_delta",
    "turnover_delta",
    "diagnosis",
    "review_note",
]


def _number(row: pd.Series, column: str) -> float:
    try:
        value = float(row.get(column, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(value):
        return 0.0
    return value


def _mode_and_note(row: pd.Series) -> tuple[str, str]:
    model = str(row.get("model", ""))
    baseline = str(row.get("baseline_model", ""))
    segment = str(row.get("segment", ""))
    relative_return = _number(row, "relative_return_pct_delta")
    stability = _number(row, "stability_score_delta")
    drawdown = _number(row, "max_drawdown_pct_delta")
    turnover = _number(row, "turnover_delta")

    if relative_return <= -0.5 and stability < 0.0:
        return (
            "return_stability_lag",
            f"{model} 在 {segment} 同时落后基准 {baseline} 的相对收益和稳定性，优先检查该 regime/window 的信号适配。",
        )
    if drawdown > 2.0:
        return (
            "drawdown_expansion",
            f"{model} 在 {segment} 相比基准 {baseline} 回撤扩大，收益接近时应优先降权或增加风险约束。",
        )
    if turnover > 1.0 and relative_return <= 0.25:
        return (
            "turnover_without_return",
            f"{model} 在 {segment} 增加换手但没有换来足够相对收益，需要检查交易成本和信号噪声。",
        )
    if stability < -0.5 and relative_return <= 0.25:
        return (
            "stability_lag",
            f"{model} 在 {segment} 稳定性明显弱于基准 {baseline}，不宜只按全区间收益晋升。",
        )
    if relative_return < -0.5:
        return (
            "relative_return_lag",
            f"{model} 在 {segment} 相对收益弱于基准 {baseline}，需要确认是否仍有组合分散价值。",
        )
    return "", ""


def _severity(row: pd.Series) -> float:
    relative_return = _number(row, "relative_return_pct_delta")
    stability = _number(row, "stability_score_delta")
    drawdown = _number(row, "max_drawdown_pct_delta")
    turnover = _number(row, "turnover_delta")
    return round(
        max(0.0, -relative_return) * 2.0
        + max(0.0, -stability)
        + max(0.0, drawdown) * 0.5
        + max(0.0, turnover) * 0.1,
        6,
    )


def build_failure_modes(
    model_effect_deltas: pd.DataFrame,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    if model_effect_deltas.empty:
        return pd.DataFrame(columns=FAILURE_MODE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, row in model_effect_deltas.iterrows():
        model = str(row.get("model", ""))
        baseline = str(row.get("baseline_model", ""))
        if baseline and model == baseline:
            continue
        failure_mode, review_note = _mode_and_note(row)
        if not failure_mode:
            continue
        payload = {
            "schema_version": FAILURE_MODES_SCHEMA_VERSION,
            "model": model,
            "baseline_model": baseline,
            "segment_type": str(row.get("segment_type", "")),
            "segment": str(row.get("segment", "")),
            "failure_mode": failure_mode,
            "severity_score": _severity(row),
            "relative_return_pct_delta": _number(row, "relative_return_pct_delta"),
            "stability_score_delta": _number(row, "stability_score_delta"),
            "max_drawdown_pct_delta": _number(row, "max_drawdown_pct_delta"),
            "turnover_delta": _number(row, "turnover_delta"),
            "diagnosis": str(row.get("diagnosis", "")),
            "review_note": review_note,
        }
        rows.append(payload)

    if not rows:
        return pd.DataFrame(columns=FAILURE_MODE_COLUMNS)
    result = pd.DataFrame(rows, columns=FAILURE_MODE_COLUMNS).sort_values(
        ["severity_score", "relative_return_pct_delta", "stability_score_delta"],
        ascending=[False, True, True],
    )
    if top_n is not None:
        result = result.head(top_n)
    return result.reset_index(drop=True)


def render_failure_modes_markdown(
    failure_modes: pd.DataFrame,
    *,
    title: str = "模型失败模式诊断",
    max_rows: int = 10,
) -> str:
    lines = [f"# {title}", "", "## 中文结论", ""]
    if failure_modes.empty:
        lines.extend(
            [
                "- 未识别出明显失败模式；仍需结合全区间收益、regime 表现和回撤确认是否可晋升。",
                "- 该结论只说明当前阈值下没有突出分段问题，不代表模型已经适合生产。",
            ]
        )
        return "\n".join(lines) + "\n"

    top = failure_modes.head(max_rows)
    lines.extend(
        [
            f"- 共识别 {len(failure_modes)} 条失败模式，以下列出 severity 最高的 {len(top)} 条。",
            "- 诊断优先级按相对收益落后、稳定性下降、回撤扩大和换手增加综合排序。",
            "",
            "| model | segment | failure_mode | severity | review_note |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            "| {model} | {segment} | {failure_mode} | {severity:.4f} | {note} |".format(
                model=str(row["model"]),
                segment=str(row["segment"]),
                failure_mode=str(row["failure_mode"]),
                severity=float(row["severity_score"]),
                note=str(row["review_note"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_failure_modes_markdown(
    failure_modes: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "模型失败模式诊断",
    max_rows: int = 10,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_failure_modes_markdown(failure_modes, title=title, max_rows=max_rows),
        encoding="utf-8",
    )
    return path
