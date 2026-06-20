from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core import ArtifactManifest


@dataclass(frozen=True)
class PortfolioReport:
    source: str
    report: pd.DataFrame
    manifest: ArtifactManifest
    input_artifacts: list[str]


@dataclass(frozen=True)
class BacktestExtensionReport:
    source: str
    window_metrics: pd.DataFrame
    walk_forward_metrics: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    manifest: ArtifactManifest

    def write_sidecars(self, directory: Path) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "backtest_window_metrics": directory / "backtest_window_metrics.csv",
            "backtest_walk_forward_metrics": directory / "backtest_walk_forward_metrics.csv",
            "backtest_benchmark_comparison": directory / "backtest_benchmark_comparison.csv",
        }
        self.window_metrics.to_csv(paths["backtest_window_metrics"], index=False)
        self.walk_forward_metrics.to_csv(paths["backtest_walk_forward_metrics"], index=False)
        self.benchmark_comparison.to_csv(paths["backtest_benchmark_comparison"], index=False)
        return paths


def _as_score_frame(pred: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(pred, pd.Series):
        frame = pred.rename("score").to_frame()
    else:
        frame = pred.copy()
        if "score" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "score"})
    return frame.sort_index()


def _as_label_frame(label: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(label, pd.Series):
        frame = label.rename("forward_return").to_frame()
    else:
        frame = label.copy()
        if "forward_return" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "forward_return"})
    return frame.sort_index()


def build_portfolio_report_from_qlib_recorder(
    summary: dict,
    *,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> PortfolioReport:
    recorder_dir = Path(summary["recorder_local_dir"])
    qlib_report_path = recorder_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl"
    qlib_risk_path = recorder_dir / "artifacts" / "portfolio_analysis" / "port_analysis_1day.pkl"
    if qlib_report_path.exists() and qlib_risk_path.exists():
        qlib_report = pd.read_pickle(qlib_report_path).reset_index()
        if "datetime" in qlib_report.columns:
            qlib_report["date"] = pd.to_datetime(qlib_report["datetime"]).dt.date.astype(str)
        else:
            qlib_report["date"] = pd.to_datetime(qlib_report.iloc[:, 0]).dt.date.astype(str)
        qlib_report["strategy_return"] = qlib_report.get("return", 0.0).astype(float) - qlib_report.get(
            "cost", 0.0
        ).astype(float)
        qlib_report["benchmark_return"] = qlib_report.get("bench", 0.0).astype(float)
        qlib_report["turnover"] = qlib_report.get("turnover", pd.Series(0.0, index=qlib_report.index)).astype(float)
        qlib_report["cost"] = qlib_report.get("cost", pd.Series(0.0, index=qlib_report.index)).astype(float)
        qlib_report["nav"] = (1.0 + qlib_report["strategy_return"]).cumprod()
        qlib_report["score"] = 0.0
        qlib_report["forward_return"] = qlib_report["strategy_return"]
        qlib_report["position"] = 1.0
        report = qlib_report[
            [
                "date",
                "score",
                "forward_return",
                "position",
                "turnover",
                "cost",
                "strategy_return",
                "benchmark_return",
                "nav",
            ]
        ]
        manifest = ArtifactManifest(
            kind="portfolio_report",
            path="qlib_recorder/artifacts/portfolio_analysis/report_normal_1day.pkl",
            source="qlib_recorder",
            provider_uri=summary["provider_uri"],
            recorder_id=summary["recorder_id"],
            date_range=tuple(summary["data_range"]),
            purpose="backtest",
            metadata={
                "experiment": summary["experiment"],
                "record_classes": summary.get("qlib_record_classes_available", []),
                "cost_bps": cost_bps,
                "slippage_bps": slippage_bps,
                "risk_analysis": "qlib_recorder/artifacts/portfolio_analysis/port_analysis_1day.pkl",
            },
        )
        return PortfolioReport(
            source="qlib_recorder",
            report=report,
            manifest=manifest,
            input_artifacts=["portfolio_analysis/report_normal_1day.pkl", "portfolio_analysis/port_analysis_1day.pkl"],
        )

    pred_path = recorder_dir / "artifacts" / "pred.pkl"
    label_path = recorder_dir / "artifacts" / "label.pkl"
    if not pred_path.exists() or not label_path.exists():
        raise FileNotFoundError("qlib recorder must include SignalRecord pred.pkl and label.pkl")

    frame = _as_score_frame(pd.read_pickle(pred_path)).join(_as_label_frame(pd.read_pickle(label_path)), how="left")
    frame = frame.reset_index()
    if "datetime" in frame.columns:
        frame["date"] = pd.to_datetime(frame["datetime"]).dt.date.astype(str)
    else:
        frame["date"] = pd.RangeIndex(len(frame)).astype(str)
    frame["position"] = frame["score"].astype(float).map(lambda value: 1.0 if value > 0 else 0.0)
    frame["forward_return"] = frame["forward_return"].fillna(0.0).astype(float)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["cost"] = frame["turnover"] * ((cost_bps + slippage_bps) / 10000.0)
    frame["strategy_return"] = frame["position"] * frame["forward_return"] - frame["cost"]
    frame["nav"] = (1.0 + frame["strategy_return"]).cumprod()
    report = frame[["date", "score", "forward_return", "position", "turnover", "cost", "strategy_return", "nav"]]
    manifest = ArtifactManifest(
        kind="portfolio_report",
        path="qlib_recorder/artifacts/pred.pkl",
        source="qlib_recorder",
        provider_uri=summary["provider_uri"],
        recorder_id=summary["recorder_id"],
        date_range=tuple(summary["data_range"]),
        purpose="backtest",
        metadata={
            "experiment": summary["experiment"],
            "record_classes": summary.get("qlib_record_classes_available", []),
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
        },
    )
    return PortfolioReport(
        source="qlib_recorder",
        report=report,
        manifest=manifest,
        input_artifacts=["pred.pkl", "label.pkl"],
    )


def _max_drawdown_pct(nav: pd.Series) -> float:
    drawdown = nav.astype(float) / nav.astype(float).cummax() - 1.0
    return round(float(drawdown.min() * 100.0), 10)


def _portfolio_window_metrics(
    report: pd.DataFrame,
    windows: dict[str, tuple[str, str]],
    *,
    window_type: str,
    slippage_bps: float,
) -> pd.DataFrame:
    frame = report.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, object]] = []
    for name, (start, end) in windows.items():
        window = frame.loc[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        if window.empty:
            continue
        strategy_return = (1.0 + window["strategy_return"].astype(float)).prod() - 1.0
        if "benchmark_return" in window.columns:
            benchmark_return = (1.0 + window["benchmark_return"].astype(float)).prod() - 1.0
        else:
            benchmark_return = 0.0
        rows.append(
            {
                "window": name,
                "window_type": window_type,
                "start": pd.Timestamp(start).date().isoformat(),
                "end": pd.Timestamp(end).date().isoformat(),
                "rows": int(len(window)),
                "strategy_return_pct": round(float(strategy_return * 100.0), 10),
                "benchmark_return_pct": round(float(benchmark_return * 100.0), 10),
                "excess_return_pct": round(float((strategy_return - benchmark_return) * 100.0), 10),
                "max_drawdown_pct": _max_drawdown_pct(window["nav"]),
                "turnover": round(float(window["turnover"].astype(float).sum()), 10),
                "cost": round(float(window["cost"].astype(float).sum()), 10),
                "slippage_cost": round(
                    float(window["turnover"].astype(float).sum() * (slippage_bps / 10000.0)),
                    10,
                ),
            }
        )
    return pd.DataFrame(rows)


def _walk_forward_windows(report: pd.DataFrame, *, train_size: int, test_size: int) -> dict[str, tuple[str, str]]:
    dates = pd.Series(pd.to_datetime(report["date"]).sort_values().unique())
    windows: dict[str, tuple[str, str]] = {}
    fold = 1
    start = 0
    while start + train_size + test_size <= len(dates):
        test = dates.iloc[start + train_size : start + train_size + test_size]
        windows[f"walk_forward_{fold}"] = (
            pd.Timestamp(test.iloc[0]).date().isoformat(),
            pd.Timestamp(test.iloc[-1]).date().isoformat(),
        )
        fold += 1
        start += test_size
    return windows


def _benchmark_comparison(
    summary: dict,
    report: pd.DataFrame,
    *,
    cost_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    strategy_return = float((1.0 + report["strategy_return"].astype(float)).prod() - 1.0)
    benchmark_return = (
        float((1.0 + report["benchmark_return"].astype(float)).prod() - 1.0)
        if "benchmark_return" in report.columns
        else 0.0
    )
    excess = report["strategy_return"].astype(float) - report.get(
        "benchmark_return",
        pd.Series(0.0, index=report.index),
    ).astype(float)
    return pd.DataFrame(
        [
            {
                "benchmark_source": "qlib_portfolio_analysis",
                "recorder_id": summary["recorder_id"],
                "strategy_return_pct": round(strategy_return * 100.0, 10),
                "benchmark_return_pct": round(benchmark_return * 100.0, 10),
                "excess_return_pct": round((strategy_return - benchmark_return) * 100.0, 10),
                "tracking_error": round(float(excess.std(ddof=0)), 10),
                "cost_bps": cost_bps,
                "slippage_bps": slippage_bps,
            }
        ]
    )


def build_backtest_extension_report_from_qlib_recorder(
    summary: dict,
    portfolio_report: pd.DataFrame,
    *,
    windows: dict[str, tuple[str, str]],
    stress_windows: dict[str, tuple[str, str]] | None = None,
    walk_forward: dict[str, int] | None = None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestExtensionReport:
    recorder_dir = Path(summary["recorder_local_dir"])
    required = [
        recorder_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl",
        recorder_dir / "artifacts" / "portfolio_analysis" / "port_analysis_1day.pkl",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing qlib portfolio_analysis artifacts: {missing}")

    fixed_metrics = _portfolio_window_metrics(
        portfolio_report,
        windows,
        window_type="fixed",
        slippage_bps=slippage_bps,
    )
    stress_metrics = _portfolio_window_metrics(
        portfolio_report,
        stress_windows or {},
        window_type="stress",
        slippage_bps=slippage_bps,
    )
    window_metrics = pd.concat([fixed_metrics, stress_metrics], ignore_index=True)
    walk_windows = (
        _walk_forward_windows(
            portfolio_report,
            train_size=int(walk_forward["train_size"]),
            test_size=int(walk_forward["test_size"]),
        )
        if walk_forward
        else {}
    )
    walk_forward_metrics = _portfolio_window_metrics(
        portfolio_report,
        walk_windows,
        window_type="walk_forward",
        slippage_bps=slippage_bps,
    )
    manifest = ArtifactManifest(
        kind="backtest_extension_report",
        path="qlib_recorder/artifacts/portfolio_analysis/report_normal_1day.pkl",
        source="qlib_recorder",
        provider_uri=summary["provider_uri"],
        recorder_id=summary["recorder_id"],
        date_range=tuple(summary["data_range"]),
        purpose="backtest",
        metadata={
            "experiment": summary["experiment"],
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
            "input_artifacts": [
                "portfolio_analysis/report_normal_1day.pkl",
                "portfolio_analysis/port_analysis_1day.pkl",
            ],
        },
    )
    return BacktestExtensionReport(
        source="qlib_portfolio_analysis",
        window_metrics=window_metrics,
        walk_forward_metrics=walk_forward_metrics,
        benchmark_comparison=_benchmark_comparison(
            summary,
            portfolio_report,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        ),
        manifest=manifest,
    )
