from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _max_drawdown_pct(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1.0
    return round(float(drawdown.min() * 100.0), 10)


def assign_regimes(frame: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    result = frame.copy()
    result["vix_regime"] = pd.cut(
        result["vix"],
        bins=[float("-inf"), 20.0, 30.0, float("inf")],
        labels=["low_vix", "mid_vix", "high_vix"],
        right=True,
    ).astype(str)
    trend = result[price_col].pct_change().fillna(0.0)
    result["trend_regime"] = trend.map(lambda value: "uptrend" if value >= 0 else "downtrend")
    volume_ratio = result["volume"] / result["volume"].rolling(2, min_periods=1).mean()
    result["volume_regime"] = volume_ratio.map(lambda value: "volume_shock" if value >= 1.5 else "normal_volume")
    return result


def compute_window_metrics(result: pd.DataFrame, *, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = result.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for name, (start, end) in windows.items():
        window = frame.loc[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        if window.empty:
            continue
        rows.append(
            {
                "window": name,
                "start": pd.Timestamp(start).date().isoformat(),
                "end": pd.Timestamp(end).date().isoformat(),
                "return_pct": round((float(window["nav"].iloc[-1]) / float(window["nav"].iloc[0]) - 1.0) * 100.0, 10),
                "max_drawdown_pct": _max_drawdown_pct(window["nav"]),
                "turnover": round(float(window.get("turnover", pd.Series(dtype="float64")).sum()), 10),
            }
        )
    return pd.DataFrame(rows)


def compare_model_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["promotion_score"] = (
        result["return_pct"].astype(float)
        - result["max_drawdown_pct"].abs().astype(float)
        - result["turnover"].astype(float) * 5.0
    )
    return result.sort_values(["promotion_score", "return_pct"], ascending=[False, False]).reset_index(drop=True)


@dataclass(frozen=True)
class AnalysisReport:
    name: str
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
        paths["summary"].write_text(
            json.dumps({"name": self.name, "summary": self.summary}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.regime_metrics.to_csv(paths["regime_metrics"], index=False)
        self.window_metrics.to_csv(paths["window_metrics"], index=False)
        self.model_comparison.to_csv(paths["model_comparison"], index=False)
        paths["failure_modes"].write_text("\n".join(f"- {item}" for item in self.failure_modes) + "\n", encoding="utf-8")
        return paths
