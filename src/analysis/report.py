from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _max_drawdown_pct(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1.0
    return round(float(drawdown.min() * 100.0), 10)


def _window_metrics(report: pd.DataFrame, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    frame = report.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, Any]] = []
    for name, (start, end) in windows.items():
        window = frame.loc[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        if window.empty:
            continue
        nav = window["nav"].astype(float)
        rows.append(
            {
                "window": name,
                "start": pd.Timestamp(start).date().isoformat(),
                "end": pd.Timestamp(end).date().isoformat(),
                "return_pct": round((float(nav.iloc[-1]) / float(nav.iloc[0]) - 1.0) * 100.0, 10),
                "max_drawdown_pct": _max_drawdown_pct(nav),
                "turnover": round(float(window["turnover"].sum()), 10),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AnalysisReport:
    summary: dict[str, Any]
    regime_metrics: pd.DataFrame
    window_metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    failure_modes: list[str]

    def write_sidecars(self, directory: Path) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "summary": directory / "analysis_summary.json",
            "regime_metrics": directory / "regime_metrics.csv",
            "window_metrics": directory / "window_metrics.csv",
            "model_comparison": directory / "model_comparison.csv",
            "failure_modes": directory / "failure_modes.md",
        }
        paths["summary"].write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        self.regime_metrics.to_csv(paths["regime_metrics"], index=False)
        self.window_metrics.to_csv(paths["window_metrics"], index=False)
        self.model_comparison.to_csv(paths["model_comparison"], index=False)
        paths["failure_modes"].write_text("\n".join(f"- {item}" for item in self.failure_modes) + "\n", encoding="utf-8")
        return paths


def build_analysis_report_from_qlib_recorder(
    summary: dict,
    portfolio_report: pd.DataFrame,
    *,
    windows: dict[str, tuple[str, str]],
) -> AnalysisReport:
    recorder_dir = Path(summary["recorder_local_dir"])
    metrics = {path.name: _read_metric(path) for path in (recorder_dir / "metrics").glob("*") if path.is_file()}
    comparison = pd.DataFrame(
        [
            {
                "model": "qlib_signal_record",
                "return_pct": round((portfolio_report["nav"].iloc[-1] / portfolio_report["nav"].iloc[0] - 1.0) * 100.0, 10),
                "max_drawdown_pct": _max_drawdown_pct(portfolio_report["nav"]),
                "turnover": round(float(portfolio_report["turnover"].sum()), 10),
                "ic": metrics.get("IC", 0.0),
                "rank_ic": metrics.get("Rank IC", 0.0),
                "recorder_id": summary["recorder_id"],
            }
        ]
    )
    regime_frame = portfolio_report.copy()
    regime_frame["return_regime"] = regime_frame["forward_return"].map(
        lambda value: "positive_forward_return" if value >= 0 else "negative_forward_return"
    )
    regime_metrics = (
        regime_frame.groupby("return_regime", observed=True)
        .agg(
            rows=("return_regime", "count"),
            mean_strategy_return=("strategy_return", "mean"),
            mean_position=("position", "mean"),
            turnover=("turnover", "sum"),
        )
        .reset_index()
    )
    return AnalysisReport(
        summary={
            "source": "qlib_recorder",
            "experiment": summary["experiment"],
            "recorder_id": summary["recorder_id"],
            "provider_uri": summary["provider_uri"],
            "qlib_docs": summary["qlib_docs"],
            "metrics": metrics,
        },
        regime_metrics=regime_metrics,
        window_metrics=_window_metrics(portfolio_report, windows),
        model_comparison=comparison,
        failure_modes=["short-window-signal-analysis-can-produce-nan-ic"],
    )


def _read_metric(path: Path) -> float:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0.0
    parts = text.split()
    raw = parts[1] if len(parts) >= 2 else parts[0]
    value = float(raw)
    return 0.0 if math.isnan(value) else value
