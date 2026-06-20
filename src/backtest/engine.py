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
