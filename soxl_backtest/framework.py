from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    direction: str
    start: str
    end: str
    changed_layer: str
    role: str
    baseline: str
    comparison_key: str
    instrument: str = "SOXL"
    qlib_region: str = "us"
    qlib_freq: str = "day"


@dataclass(frozen=True)
class BacktestResult:
    spec: ExperimentSpec
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def repo_root() -> Path:
    return ROOT


def qlib_data_dir(root: Path | None = None) -> Path:
    return Path(root or ROOT) / "data" / "qlib" / "soxl"


def load_qlib_calendar(root: Path | None = None) -> pd.DatetimeIndex:
    calendar_path = qlib_data_dir(root) / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"missing qlib calendar: {calendar_path}")
    values = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DatetimeIndex(pd.to_datetime(values), name="date")


def load_qlib_frame(
    *,
    start: str,
    end: str,
    instrument: str = "SOXL",
    root: Path | None = None,
) -> pd.DataFrame:
    repo = Path(root or ROOT)
    calendar_path = qlib_data_dir(repo) / "calendars" / "day.txt"
    calendar = load_qlib_calendar(repo)
    panel_path = repo / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"missing canonical SOXL panel: {panel_path}")

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    valid_dates = calendar[(calendar >= start_ts) & (calendar <= end_ts)]
    frame = panel.loc[panel["date"].isin(valid_dates)].copy()
    if frame.empty:
        raise ValueError(f"no rows for {instrument} between {start} and {end}")

    instrument_key = instrument.lower()
    close_col = f"{instrument_key}_close"
    open_col = f"{instrument_key}_open"
    high_col = f"{instrument_key}_high"
    low_col = f"{instrument_key}_low"
    volume_col = f"{instrument_key}_volume"
    required = ["date", close_col, open_col, high_col, low_col, volume_col, "vix"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"canonical panel missing columns for {instrument}: {missing}")

    result = pd.DataFrame(
        {
            "date": frame["date"],
            "instrument": instrument,
            "open": frame[open_col],
            "high": frame[high_col],
            "low": frame[low_col],
            "close": frame[close_col],
            "price": frame[close_col],
            "volume": frame[volume_col],
            "vix": frame["vix"].ffill().bfill(),
        }
    ).reset_index(drop=True)
    result.attrs["data_source"] = "qlib_calendar_plus_canonical_panel"
    result.attrs["qlib_calendar_path"] = str(calendar_path)
    result.attrs["canonical_panel_path"] = str(panel_path)
    return result


def write_card_artifacts(card_dir: Path, result: BacktestResult) -> dict[str, Any]:
    card_dir.mkdir(parents=True, exist_ok=True)
    daily_path = card_dir / "daily.csv"
    trades_path = card_dir / "trades.csv"
    metrics_path = card_dir / "metrics.json"
    spec_path = card_dir / "experiment_spec.json"
    summary_path = card_dir / "summary.json"

    result.daily.to_csv(daily_path, index=False)
    result.trades.to_csv(trades_path, index=False)
    metrics_path.write_text(json.dumps(_json_safe(result.metrics), indent=2), encoding="utf-8")
    spec_path.write_text(json.dumps(_json_safe(asdict(result.spec)), indent=2), encoding="utf-8")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": result.spec.experiment,
        "direction": result.spec.direction,
        "changed_layer": result.spec.changed_layer,
        "role": result.spec.role,
        "baseline": result.spec.baseline,
        "comparison_key": result.spec.comparison_key,
        "metrics": _json_safe(result.metrics),
        "artifacts": {
            "daily": daily_path.name,
            "trades": trades_path.name,
            "metrics": metrics_path.name,
            "experiment_spec": spec_path.name,
            "summary": summary_path.name,
        },
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


class QlibBacktestRunner:
    def __init__(self, spec: ExperimentSpec):
        self.spec = spec

    def run_reference(self, card_dir: Path | None = None) -> BacktestResult:
        from .reference import replay_soxlbacktest_reference

        replay = replay_soxlbacktest_reference(start=self.spec.start, end=self.spec.end)
        result = BacktestResult(
            spec=self.spec,
            daily=replay.daily,
            trades=replay.trades,
            metrics=replay.metrics,
        )
        if card_dir is not None:
            write_card_artifacts(Path(card_dir), result)
        return result
