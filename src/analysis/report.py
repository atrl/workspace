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
    drawdown = result[price_col] / result[price_col].cummax() - 1.0
    result["drawdown_regime"] = drawdown.map(
        lambda value: "deep_drawdown" if value <= -0.15 else "pullback" if value <= -0.05 else "no_drawdown"
    )
    realized_vol = trend.rolling(3, min_periods=2).std(ddof=0).fillna(0.0)
    result["volatility_regime"] = realized_vol.map(
        lambda value: "high_volatility" if value >= 0.05 else "mid_volatility" if value >= 0.02 else "low_volatility"
    )
    result["market_regime"] = [
        "bear" if dd <= -0.10 else "bull" if change >= 0 and dd > -0.05 else "chop"
        for change, dd in zip(trend, drawdown, strict=True)
    ]
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


@dataclass(frozen=True)
class QlibRecorderArtifacts:
    recorder_dir: Path
    summary: dict[str, Any]
    portfolio_report: pd.DataFrame
    risk_metrics: dict[str, float]
    model_comparison: pd.DataFrame

    def to_analysis_report(
        self,
        *,
        windows: dict[str, tuple[str, str]],
        failure_modes: list[str],
    ) -> AnalysisReport:
        window_metrics = compute_window_metrics(self.portfolio_report, windows=windows)
        return AnalysisReport(
            name=self.summary["experiment"],
            summary={
                "source": "qlib_recorder_artifacts",
                "recorder_id": self.summary["recorder_id"],
                "risk_metrics": self.risk_metrics,
                "qlib_docs": self.summary["qlib_docs"],
            },
            regime_metrics=pd.DataFrame(),
            window_metrics=window_metrics,
            model_comparison=compare_model_metrics(self.model_comparison),
            failure_modes=failure_modes,
        )


def _read_risk_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if {"metric", "value"}.issubset(frame.columns):
        return {str(row["metric"]): float(row["value"]) for _, row in frame.iterrows()}
    if len(frame) == 1:
        return {str(column): float(frame.iloc[0][column]) for column in frame.columns}
    raise ValueError(f"unsupported risk analysis shape: {path}")


def load_qlib_recorder_artifacts(
    recorder_dir: Path,
    *,
    experiment: str,
    recorder_id: str,
) -> QlibRecorderArtifacts:
    recorder_path = Path(recorder_dir)
    metrics_path = recorder_path / "metrics.json"
    portfolio_path = recorder_path / "portfolio_report.csv"
    risk_path = recorder_path / "risk_analysis.csv"
    model_path = recorder_path / "model_comparison.csv"
    for path in [metrics_path, portfolio_path, risk_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(f"missing qlib recorder artifact: {path}")

    recorder_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    portfolio_report = pd.read_csv(portfolio_path, parse_dates=["date"])
    if "cum return w cost" in portfolio_report.columns and "nav" not in portfolio_report.columns:
        portfolio_report["nav"] = portfolio_report["cum return w cost"].astype(float)
    if "turnover" not in portfolio_report.columns:
        portfolio_report["turnover"] = 0.0

    risk_metrics = _read_risk_metrics(risk_path)
    model_comparison = pd.read_csv(model_path)
    summary = {
        "experiment": experiment,
        "recorder_id": recorder_id,
        "recorder_dir": recorder_path.as_posix(),
        "recorder_metrics": recorder_metrics,
        "qlib_docs": {
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
            "report": "https://qlib.readthedocs.io/en/latest/component/report.html",
        },
    }
    return QlibRecorderArtifacts(
        recorder_dir=recorder_path,
        summary=summary,
        portfolio_report=portfolio_report,
        risk_metrics=risk_metrics,
        model_comparison=model_comparison,
    )
