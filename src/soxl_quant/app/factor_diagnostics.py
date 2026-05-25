from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from src.soxl_quant.app.oos_diagnostics import ExternalReference


@dataclass(frozen=True)
class FactorGateThresholds:
    min_abs_ic_mean: float = 0.03
    min_ic_positive_ratio: float = 0.55
    min_profit_factor: float = 1.2
    min_signal_accuracy: float = 0.55
    max_core_correlation: float = 0.80


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _best_ic_rows(ic_df: pd.DataFrame) -> pd.DataFrame:
    required = {"feature_name", "horizon", "ic_mean", "ic_t", "ic_positive_ratio"}
    _require_columns(ic_df, required, "ic_df")
    frame = ic_df.copy()
    frame["abs_ic_mean"] = pd.to_numeric(frame["ic_mean"], errors="coerce").abs()
    frame = frame.sort_values(["feature_name", "abs_ic_mean", "ic_positive_ratio"], ascending=[True, False, False])
    return frame.groupby("feature_name", as_index=False).head(1)


def _best_backtest_rows(backtest_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "feature_name",
        "direction",
        "threshold",
        "sharpe",
        "max_drawdown",
        "profit_factor",
        "signal_accuracy",
    }
    _require_columns(backtest_df, required, "backtest_df")
    frame = backtest_df.copy()
    frame["sharpe"] = pd.to_numeric(frame["sharpe"], errors="coerce")
    frame["profit_factor"] = pd.to_numeric(frame["profit_factor"], errors="coerce")
    frame = frame.sort_values(["feature_name", "sharpe", "profit_factor"], ascending=[True, False, False])
    return frame.groupby("feature_name", as_index=False).head(1)


def _primary_failure(row: Mapping[str, object]) -> str:
    if bool(row["full_gate"]):
        return "pass"
    if not bool(row["correlation_gate"]):
        return "correlation_redundant"
    if not bool(row["ic_gate"]) and not bool(row["backtest_gate"]):
        return "weak_direction_and_execution"
    if not bool(row["ic_gate"]):
        return "weak_direction"
    if not bool(row["backtest_gate"]):
        return "weak_execution"
    return "unknown"


def build_factor_gate_summary(
    ic_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    *,
    thresholds: FactorGateThresholds = FactorGateThresholds(),
) -> pd.DataFrame:
    """Summarize factor IC, standalone execution, and correlation gates."""

    _require_columns(
        corr_df,
        {"feature_name", "max_abs_core_correlation", "max_abs_core_correlation_feature"},
        "corr_df",
    )
    best_ic = _best_ic_rows(ic_df)
    best_bt = _best_backtest_rows(backtest_df)
    corr = corr_df.copy()

    rows = []
    for feature_name in sorted(set(best_ic["feature_name"]) | set(best_bt["feature_name"]) | set(corr["feature_name"])):
        ic_row = best_ic[best_ic["feature_name"] == feature_name]
        bt_row = best_bt[best_bt["feature_name"] == feature_name]
        corr_row = corr[corr["feature_name"] == feature_name]
        if ic_row.empty or bt_row.empty or corr_row.empty:
            continue
        ic = ic_row.iloc[0]
        bt = bt_row.iloc[0]
        cr = corr_row.iloc[0]
        abs_ic = abs(float(ic["ic_mean"]))
        ic_positive_ratio = float(ic["ic_positive_ratio"])
        profit_factor = float(bt["profit_factor"])
        signal_accuracy = float(bt["signal_accuracy"])
        max_core_corr = abs(float(cr["max_abs_core_correlation"]))
        summary_row = {
            "feature_name": str(feature_name),
            "best_ic_horizon": int(ic["horizon"]),
            "best_ic_mean": float(ic["ic_mean"]),
            "best_abs_ic_mean": abs_ic,
            "best_ic_t": float(ic["ic_t"]),
            "best_ic_positive_ratio": ic_positive_ratio,
            "best_backtest_direction": str(bt["direction"]),
            "best_threshold": float(bt["threshold"]),
            "best_sharpe": float(bt["sharpe"]),
            "best_profit_factor": profit_factor,
            "best_signal_accuracy": signal_accuracy,
            "best_max_drawdown": float(bt["max_drawdown"]),
            "max_abs_core_correlation": max_core_corr,
            "max_abs_core_correlation_feature": str(cr["max_abs_core_correlation_feature"]),
        }
        summary_row["ic_gate"] = bool(
            abs_ic > thresholds.min_abs_ic_mean
            and ic_positive_ratio > thresholds.min_ic_positive_ratio
        )
        summary_row["backtest_gate"] = bool(
            profit_factor > thresholds.min_profit_factor
            and signal_accuracy > thresholds.min_signal_accuracy
        )
        summary_row["correlation_gate"] = bool(max_core_corr <= thresholds.max_core_correlation)
        summary_row["full_gate"] = bool(
            summary_row["ic_gate"]
            and summary_row["backtest_gate"]
            and summary_row["correlation_gate"]
        )
        summary_row["primary_failure"] = _primary_failure(summary_row)
        rows.append(summary_row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "best_ic_horizon",
                "best_ic_mean",
                "best_abs_ic_mean",
                "best_ic_t",
                "best_ic_positive_ratio",
                "best_backtest_direction",
                "best_threshold",
                "best_sharpe",
                "best_profit_factor",
                "best_signal_accuracy",
                "best_max_drawdown",
                "max_abs_core_correlation",
                "max_abs_core_correlation_feature",
                "ic_gate",
                "backtest_gate",
                "correlation_gate",
                "full_gate",
                "primary_failure",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["full_gate", "best_abs_ic_mean", "best_profit_factor", "feature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _reference_rows(external_references: Sequence[Mapping[str, str]] | Sequence[ExternalReference]) -> list[str]:
    rows = []
    for ref in external_references:
        if isinstance(ref, ExternalReference):
            label, url, note = ref.label, ref.url, ref.note
        else:
            label = str(ref.get("label", "reference"))
            url = str(ref.get("url", ""))
            note = str(ref.get("note", ""))
        rows.append(f"- [{label}]({url}): {note}")
    return rows


def render_factor_gate_summary(
    summary: pd.DataFrame,
    *,
    external_references: Sequence[Mapping[str, str]] | Sequence[ExternalReference] = (),
) -> str:
    lines = ["# Factor Gate Summary", ""]
    if external_references:
        lines.extend(["## External References", "", *_reference_rows(external_references), ""])
    lines.extend(
        [
            "## Gate Table",
            "",
            "| feature | IC | IC+ | PF | Acc | max rho | full | failure |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if summary.empty:
        lines.append("| none | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 | false | none |")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                "| "
                f"{row.feature_name} | {float(row.best_ic_mean):+.4f} | "
                f"{float(row.best_ic_positive_ratio):.3f} | "
                f"{float(row.best_profit_factor):.3f} | "
                f"{float(row.best_signal_accuracy):.3f} | "
                f"{float(row.max_abs_core_correlation):.3f} | "
                f"{bool(row.full_gate)} | {row.primary_failure} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Full-gate factors can move to a profile or gated-composite proposal.",
            "- Correlation failures should not be promoted as core RL features without a distinct non-linear use case.",
            "- Weak-direction failures need IC or regime-slice improvement before any training branch.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "FactorGateThresholds",
    "build_factor_gate_summary",
    "render_factor_gate_summary",
]
