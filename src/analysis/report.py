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
    explainability: pd.DataFrame
    failure_modes: list[str]

    def write_sidecars(self, directory: Path) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "summary": directory / "analysis_summary.json",
            "regime_metrics": directory / "regime_metrics.csv",
            "window_metrics": directory / "window_metrics.csv",
            "model_comparison": directory / "model_comparison.csv",
            "explainability": directory / "explainability.csv",
            "failure_modes": directory / "failure_modes.md",
        }
        paths["summary"].write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        self.regime_metrics.to_csv(paths["regime_metrics"], index=False)
        self.window_metrics.to_csv(paths["window_metrics"], index=False)
        self.model_comparison.to_csv(paths["model_comparison"], index=False)
        self.explainability.to_csv(paths["explainability"], index=False)
        paths["failure_modes"].write_text("\n".join(f"- {item}" for item in self.failure_modes) + "\n", encoding="utf-8")
        return paths


def _walk_forward_windows(dates: pd.Series, *, train_size: int, test_size: int) -> dict[str, tuple[str, str]]:
    ordered = pd.Series(pd.to_datetime(dates).sort_values().unique())
    windows: dict[str, tuple[str, str]] = {}
    start = 0
    fold = 1
    while start + train_size + test_size <= len(ordered):
        test = ordered.iloc[start + train_size : start + train_size + test_size]
        windows[f"walk_forward_{fold}"] = (
            pd.Timestamp(test.iloc[0]).date().isoformat(),
            pd.Timestamp(test.iloc[-1]).date().isoformat(),
        )
        start += test_size
        fold += 1
    return windows


def _load_regime_features(summary: dict) -> pd.DataFrame:
    from src.data import QlibProviderConfig, load_qlib_feature_frame

    config = QlibProviderConfig(
        provider_uri=Path(summary["provider_uri"]),
        region=summary.get("region", "us"),
        market=summary.get("market", "all"),
    )
    features = load_qlib_feature_frame(
        config,
        instruments=[summary["instrument"]],
        fields=["$close", "$vix", "$volume"],
        start=summary["data_range"][0],
        end=summary["data_range"][1],
    ).reset_index()
    features["date"] = pd.to_datetime(features["datetime"]).dt.date.astype(str)
    return features[["date", "$close", "$vix", "$volume"]]


def _regime_metrics(portfolio_report: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    frame = portfolio_report.merge(features, on="date", how="left").sort_values("date")
    close = frame["$close"].astype(float).ffill()
    returns = close.pct_change().fillna(0.0)
    drawdown = close / close.cummax() - 1.0
    realized_vol = returns.rolling(3, min_periods=1).std(ddof=0).fillna(0.0)
    volume_ratio = frame["$volume"].astype(float) / frame["$volume"].astype(float).rolling(3, min_periods=1).mean()
    regimes = pd.DataFrame(
        {
            "date": frame["date"],
            "strategy_return": frame["strategy_return"].astype(float),
            "turnover": frame["turnover"].astype(float),
            "vix": pd.cut(
                frame["$vix"].astype(float),
                bins=[float("-inf"), 20.0, 30.0, float("inf")],
                labels=["low_vix", "mid_vix", "high_vix"],
            ).astype(str),
            "trend": returns.map(lambda value: "uptrend" if value >= 0 else "downtrend"),
            "drawdown": drawdown.map(
                lambda value: "deep_drawdown" if value <= -0.15 else "pullback" if value <= -0.05 else "no_drawdown"
            ),
            "volatility": realized_vol.map(
                lambda value: "high_volatility" if value >= 0.05 else "mid_volatility" if value >= 0.02 else "low_volatility"
            ),
            "volume": volume_ratio.map(lambda value: "volume_shock" if value >= 1.5 else "normal_volume"),
            "market": [
                "bear" if dd <= -0.10 else "bull" if change >= 0 and dd > -0.05 else "chop"
                for change, dd in zip(returns, drawdown, strict=True)
            ],
        }
    )
    rows: list[dict[str, Any]] = []
    for regime_type in ["vix", "trend", "drawdown", "volatility", "volume", "market"]:
        for regime, group in regimes.groupby(regime_type, observed=True):
            rows.append(
                {
                    "regime_type": regime_type,
                    "regime": str(regime),
                    "rows": int(len(group)),
                    "return_pct": round(float(((1.0 + group["strategy_return"]).prod() - 1.0) * 100.0), 10),
                    "mean_strategy_return": round(float(group["strategy_return"].mean()), 10),
                    "turnover": round(float(group["turnover"].sum()), 10),
                }
            )
    return pd.DataFrame(rows)


def _explainability(portfolio_report: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    frame = portfolio_report.merge(features, on="date", how="left")
    rows: list[dict[str, Any]] = []
    for feature in ["$close", "$vix", "$volume"]:
        clean = frame[[feature, "strategy_return"]].dropna()
        importance = 0.0
        if not clean.empty and clean[feature].nunique() > 1 and clean["strategy_return"].nunique() > 1:
            importance = abs(float(clean[feature].corr(clean["strategy_return"])))
        rows.append({"feature": feature, "importance": round(importance, 10), "method": "permutation"})
    rows.append(
        {
            "feature": "rolling_return_window",
            "importance": round(float(portfolio_report["strategy_return"].rolling(2, min_periods=1).mean().abs().mean()), 10),
            "method": "window_sensitivity",
        }
    )
    return pd.DataFrame(rows)


def build_analysis_report_from_qlib_recorder(
    summary: dict,
    portfolio_report: pd.DataFrame,
    *,
    windows: dict[str, tuple[str, str]],
    walk_forward: dict[str, int] | None = None,
) -> AnalysisReport:
    recorder_dir = Path(summary["recorder_local_dir"])
    metrics = {path.name: _read_metric(path) for path in (recorder_dir / "metrics").glob("*") if path.is_file()}
    if walk_forward:
        windows = windows | _walk_forward_windows(
            portfolio_report["date"],
            train_size=int(walk_forward["train_size"]),
            test_size=int(walk_forward["test_size"]),
        )
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
    features = _load_regime_features(summary)
    return AnalysisReport(
        summary={
            "source": "qlib_recorder",
            "experiment": summary["experiment"],
            "recorder_id": summary["recorder_id"],
            "provider_uri": summary["provider_uri"],
            "qlib_docs": summary["qlib_docs"],
            "metrics": metrics,
        },
        regime_metrics=_regime_metrics(portfolio_report, features),
        window_metrics=_window_metrics(portfolio_report, windows),
        model_comparison=comparison,
        explainability=_explainability(portfolio_report, features),
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
