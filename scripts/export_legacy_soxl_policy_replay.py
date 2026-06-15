#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.policy.legacy_soxl import replay_legacy_soxl_policy


ORACLE_CARD = ROOT / "runtime" / "production" / "card" / "soxl_backtest_jan_june_windows_v1"
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "production" / "card" / "soxl_backtest_policy_reimplementation_v1"

WINDOWS = {
    "ytd_2026": {
        "legacy_window": "legacy_ytd_2026_fresh_start",
        "start": "2026-01-01",
        "comparison_end": "2026-06-09",
    },
    "june_2026": {
        "legacy_window": "legacy_june_2026_fresh_start",
        "start": "2026-06-01",
        "comparison_end": "2026-06-09",
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_oracle_metrics() -> pd.DataFrame:
    path = ORACLE_CARD / "legacy_soxl_backtest_window_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing legacy oracle metrics: {path}")
    return pd.read_csv(path)


def _load_oracle_trades() -> pd.DataFrame:
    path = ORACLE_CARD / "legacy_soxl_backtest_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing legacy oracle trades: {path}")
    return pd.read_csv(path)


def _calendar_safe_backtest_end(root: Path = ROOT) -> str:
    path = root / "data" / "qlib" / "soxl" / "calendars" / "day.txt"
    if not path.exists():
        return ""
    dates = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(dates) >= 2:
        return dates[-2]
    return dates[-1] if dates else ""


def _resolve_export_end(end_date: str | None = None, root: Path = ROOT) -> str:
    return end_date or _calendar_safe_backtest_end(root) or "2026-06-09"


def _metric_row(
    window: str,
    result_metrics: dict[str, object],
    legacy_row: pd.Series,
    *,
    replay_metrics: dict[str, object],
) -> dict[str, object]:
    legacy_return = float(legacy_row["rulebase_return_pct"])
    new_return = float(result_metrics["rulebase_return_pct"])
    legacy_end_position = float(legacy_row["end_position_pct"])
    new_end_position = float(result_metrics["end_position_pct"])
    legacy_trade_count = int(legacy_row["trade_count"])
    new_trade_count = int(result_metrics["trade_count"])
    return {
        "window": window,
        "legacy_window": str(legacy_row["window"]),
        "start": str(result_metrics["start"]),
        "end": str(result_metrics["end"]),
        "comparison_end": str(result_metrics["end"]),
        "replay_end": str(replay_metrics.get("end") or ""),
        "new_return_pct": new_return,
        "legacy_return_pct": legacy_return,
        "return_delta_pct": new_return - legacy_return,
        "new_end_position_pct": new_end_position,
        "legacy_end_position_pct": legacy_end_position,
        "end_position_delta_pct": new_end_position - legacy_end_position,
        "new_trade_count": new_trade_count,
        "legacy_trade_count": legacy_trade_count,
        "trade_count_delta": new_trade_count - legacy_trade_count,
        "new_max_drawdown_pct": float(result_metrics["rulebase_max_drawdown_pct"]),
        "legacy_max_drawdown_pct": float(legacy_row["rulebase_max_drawdown_pct"]),
    }


def _timing_rows(window: str, new_trades: pd.DataFrame, legacy_trades: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_len = max(len(new_trades), len(legacy_trades))
    for idx in range(max_len):
        new_row = new_trades.iloc[idx] if idx < len(new_trades) else None
        legacy_row = legacy_trades.iloc[idx] if idx < len(legacy_trades) else None
        new_date = str(new_row["date"]) if new_row is not None else ""
        legacy_date = str(legacy_row["date"]) if legacy_row is not None else ""
        date_delta = (
            (pd.Timestamp(new_date) - pd.Timestamp(legacy_date)).days
            if new_date and legacy_date
            else None
        )
        rows.append(
            {
                "window": window,
                "sequence": idx + 1,
                "new_date": new_date,
                "legacy_date": legacy_date,
                "date_delta_days": date_delta,
                "new_type": str(new_row["type"]) if new_row is not None else "",
                "legacy_type": str(legacy_row["type"]) if legacy_row is not None else "",
                "new_position_pct": float(new_row["position"]) * 100.0 if new_row is not None else None,
                "legacy_position_pct": float(legacy_row["position"]) * 100.0 if legacy_row is not None else None,
            }
        )
    return rows


def write_legacy_soxl_policy_replay_outputs(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    end_date: str | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_metrics = _load_oracle_metrics()
    oracle_trades = _load_oracle_trades()
    export_end = _resolve_export_end(end_date)

    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    summary_windows: dict[str, object] = {}

    for window, spec in WINDOWS.items():
        result = replay_legacy_soxl_policy(start=str(spec["start"]), end=export_end)
        legacy_window = str(spec["legacy_window"])
        legacy_metric = oracle_metrics.loc[oracle_metrics["window"] == legacy_window].iloc[0]
        legacy_trade_slice = oracle_trades.loc[oracle_trades["window"] == legacy_window].reset_index(drop=True)
        comparison_end = str(spec.get("comparison_end") or legacy_metric.get("end") or result.metrics["end"])
        comparison_result = (
            result
            if str(result.metrics.get("end")) == comparison_end
            else replay_legacy_soxl_policy(start=str(spec["start"]), end=comparison_end)
        )

        daily = result.daily.copy()
        daily.insert(0, "window", window)
        trades = result.trades.copy()
        trades.insert(0, "window", window)
        daily_frames.append(daily)
        trade_frames.append(trades)

        metric = _metric_row(
            window,
            comparison_result.metrics,
            legacy_metric,
            replay_metrics=result.metrics,
        )
        timing = _timing_rows(window, comparison_result.trades.reset_index(drop=True), legacy_trade_slice)
        metric_rows.append(metric)
        timing_rows.extend(timing)
        summary_windows[window] = {
            "new_policy": _json_safe(result.metrics),
            "legacy_comparison_new_policy": _json_safe(comparison_result.metrics),
            "legacy": _json_safe(legacy_metric.to_dict()),
            "comparison": _json_safe(metric),
            "trade_timing": _json_safe(timing),
        }

    daily_path = output_dir / "new_policy_daily.csv"
    trades_path = output_dir / "new_policy_trades.csv"
    metrics_path = output_dir / "legacy_comparison_metrics.csv"
    timing_path = output_dir / "trade_timing_diagnostics.csv"
    summary_path = output_dir / "summary.json"

    pd.concat(daily_frames, ignore_index=True).to_csv(daily_path, index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(trades_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "soxl_backtest_policy_reimplementation_v1",
        "direction": "production",
        "oracle_card": str(ORACLE_CARD.relative_to(ROOT)),
        "export_end": export_end,
        "windows": summary_windows,
        "artifacts": {
            "new_policy_daily": str(daily_path.relative_to(ROOT)) if daily_path.is_relative_to(ROOT) else str(daily_path),
            "new_policy_trades": str(trades_path.relative_to(ROOT)) if trades_path.is_relative_to(ROOT) else str(trades_path),
            "legacy_comparison_metrics": str(metrics_path.relative_to(ROOT)) if metrics_path.is_relative_to(ROOT) else str(metrics_path),
            "trade_timing_diagnostics": str(timing_path.relative_to(ROOT)) if timing_path.is_relative_to(ROOT) else str(timing_path),
            "summary": str(summary_path.relative_to(ROOT)) if summary_path.is_relative_to(ROOT) else str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--end-date",
        help="Override the dashboard-safe export end date; defaults to the penultimate qlib calendar date.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_legacy_soxl_policy_replay_outputs(args.output_dir, end_date=args.end_date)
    print(summary["artifacts"]["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
