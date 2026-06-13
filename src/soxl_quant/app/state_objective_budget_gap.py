from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


YTD_WINDOW = "ytd_2026"
WF_WINDOWS = ("wf_2020", "wf_2022", "wf_2023_2024")


@dataclass(frozen=True)
class BudgetUsageGapDiagnostic:
    summary: dict[str, Any]
    window_summary: pd.DataFrame
    flat_positive_runs: pd.DataFrame


def _date_column(frame: pd.DataFrame) -> str:
    for column in ("datetime", "date"):
        if column in frame.columns:
            return column
    raise ValueError("daily frame must contain datetime or date")


def _max_drawdown_pct(nav: pd.Series) -> float:
    values = nav.astype(float)
    peak = values.cummax()
    drawdown = values / peak - 1.0
    return abs(float(drawdown.min())) * 100.0


def _return_pct(values: pd.Series) -> float:
    series = values.astype(float)
    if series.empty or float(series.iloc[0]) == 0.0:
        return 0.0
    return (float(series.iloc[-1]) / float(series.iloc[0]) - 1.0) * 100.0


def find_flat_positive_runs(
    daily: pd.DataFrame,
    *,
    flat_position_threshold: float = 0.05,
    min_run_return_pct: float = 20.0,
) -> list[dict[str, Any]]:
    if daily.empty:
        return []
    frame = daily.copy()
    date_col = _date_column(frame)
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).reset_index(drop=True)
    if "price" not in frame.columns or "position" not in frame.columns:
        raise ValueError("daily frame must contain price and position")

    runs: list[dict[str, Any]] = []
    start_idx: int | None = None
    previous_idx: int | None = None
    for idx, row in frame.iterrows():
        position = float(row["position"])
        price = float(row["price"])
        is_flat = position <= flat_position_threshold and pd.notna(price)
        if is_flat:
            if start_idx is None:
                start_idx = idx
            previous_idx = idx
            continue
        if start_idx is not None and previous_idx is not None:
            _append_flat_run(frame, date_col, start_idx, previous_idx, min_run_return_pct, runs)
        start_idx = None
        previous_idx = None
    if start_idx is not None and previous_idx is not None:
        _append_flat_run(frame, date_col, start_idx, previous_idx, min_run_return_pct, runs)
    return sorted(runs, key=lambda row: float(row["return_pct"]), reverse=True)


def _append_flat_run(
    frame: pd.DataFrame,
    date_col: str,
    start_idx: int,
    end_idx: int,
    min_run_return_pct: float,
    runs: list[dict[str, Any]],
) -> None:
    start_price = float(frame.loc[start_idx, "price"])
    end_price = float(frame.loc[end_idx, "price"])
    if start_price <= 0:
        return
    run_return_pct = (end_price / start_price - 1.0) * 100.0
    if run_return_pct < min_run_return_pct:
        return
    runs.append(
        {
            "start": pd.Timestamp(frame.loc[start_idx, date_col]).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(frame.loc[end_idx, date_col]).strftime("%Y-%m-%d"),
            "days": int(end_idx - start_idx + 1),
            "return_pct": run_return_pct,
            "start_price": start_price,
            "end_price": end_price,
        }
    )


def _window_row(window: str, daily: pd.DataFrame, metric_row: dict[str, Any] | None = None) -> dict[str, Any]:
    metric_row = metric_row or {}
    date_col = _date_column(daily)
    frame = daily.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col)
    return_pct = _coalesce_number(metric_row.get("return_pct"), _return_pct(frame["strategy_nav"]))
    buyhold_return_pct = _coalesce_number(metric_row.get("buyhold_return_pct"), _return_pct(frame["buyhold_nav"]))
    max_dd_pct = _coalesce_number(metric_row.get("max_dd_pct"), _max_drawdown_pct(frame["strategy_nav"]))
    avg_position = _coalesce_number(metric_row.get("avg_position"), float(frame["position"].astype(float).mean()))
    flat_runs = find_flat_positive_runs(frame)
    largest_flat = flat_runs[0] if flat_runs else {}
    return {
        "window": window,
        "start": pd.Timestamp(frame[date_col].iloc[0]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(frame[date_col].iloc[-1]).strftime("%Y-%m-%d"),
        "rows": int(len(frame)),
        "return_pct": return_pct,
        "buyhold_return_pct": buyhold_return_pct,
        "max_dd_pct": max_dd_pct,
        "avg_position": avg_position,
        "flat_positive_run_count": len(flat_runs),
        "largest_flat_positive_run_return_pct": largest_flat.get("return_pct"),
        "largest_flat_positive_run_start": largest_flat.get("start", ""),
        "largest_flat_positive_run_end": largest_flat.get("end", ""),
        "largest_flat_positive_run_days": largest_flat.get("days"),
    }


def _coalesce_number(primary: Any, fallback: float) -> float:
    try:
        if primary is not None and not pd.isna(primary):
            return float(primary)
    except TypeError:
        pass
    return float(fallback)


def build_budget_usage_gap_diagnostic(
    *,
    metrics: pd.DataFrame,
    daily_by_window: dict[str, pd.DataFrame],
    required_soxlbacktest_return_pct: float,
    required_best_return_pct: float,
    max_walkforward_drawdown_pct: float,
    ytd_structural_gap_threshold_pp: float = 50.0,
) -> BudgetUsageGapDiagnostic:
    metrics_by_window = {
        str(row["window"]): row.to_dict()
        for _, row in metrics.iterrows()
        if "window" in metrics.columns
    }
    window_rows = [
        _window_row(window, daily, metrics_by_window.get(window))
        for window, daily in daily_by_window.items()
    ]
    window_summary = pd.DataFrame(window_rows)
    ytd_row = _select_window(window_summary, YTD_WINDOW)
    ytd_return = float(ytd_row.get("return_pct", 0.0))
    ytd_gap_to_soxlbacktest = required_soxlbacktest_return_pct - ytd_return
    ytd_gap_to_best = required_best_return_pct - ytd_return
    largest_ytd_flat_run = _number_or_zero(ytd_row.get("largest_flat_positive_run_return_pct"))
    structural_ytd_gap = (
        ytd_gap_to_soxlbacktest > ytd_structural_gap_threshold_pp
        and largest_ytd_flat_run > ytd_structural_gap_threshold_pp
    )

    wf_rows = window_summary[window_summary["window"].isin(WF_WINDOWS)]
    worst_wf_max_dd = float(wf_rows["max_dd_pct"].max()) if not wf_rows.empty else 0.0
    walkforward_drawdown_gap = worst_wf_max_dd > max_walkforward_drawdown_pct
    ppo_allowed = not structural_ytd_gap and not walkforward_drawdown_gap
    if structural_ytd_gap:
        decision = "revise_fold_local_reentry_participation_targets_before_ppo"
        next_mechanism = "build_fold_local_reentry_participation_target_then_bc_or_ppo"
    elif walkforward_drawdown_gap:
        decision = "revise_fold_local_drawdown_target_before_ppo"
        next_mechanism = "tighten_fold_local_drawdown_target_then_replay"
    else:
        decision = "ppo_candidate_gap_not_structural"
        next_mechanism = "run_fold_local_ppo_replay_gate"

    flat_runs = []
    for window, daily in daily_by_window.items():
        for run in find_flat_positive_runs(daily):
            flat_runs.append({"window": window, **run})

    summary = {
        "experiment": "state_objective_budget_gap_diagnostic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "ppo_allowed_from_current_target": bool(ppo_allowed),
        "training_allowed_from_this_card": bool(ppo_allowed),
        "source_experiment": "state_objective_budget_usage_bc_official_replay_v1",
        "reference_baseline": "soxlbacktest_qlib_buysell_reference",
        "required_soxlbacktest_return_pct": float(required_soxlbacktest_return_pct),
        "required_best_return_pct": float(required_best_return_pct),
        "max_walkforward_drawdown_pct": float(max_walkforward_drawdown_pct),
        "ytd_return_pct": ytd_return,
        "ytd_gap_to_soxlbacktest_pp": float(ytd_gap_to_soxlbacktest),
        "ytd_gap_to_best_pp": float(ytd_gap_to_best),
        "largest_ytd_flat_positive_run_return_pct": float(largest_ytd_flat_run),
        "structural_ytd_participation_gap": bool(structural_ytd_gap),
        "worst_wf_max_drawdown_pct": worst_wf_max_dd,
        "walkforward_drawdown_gap": bool(walkforward_drawdown_gap),
        "next_mechanism": next_mechanism,
    }
    return BudgetUsageGapDiagnostic(
        summary=summary,
        window_summary=window_summary,
        flat_positive_runs=pd.DataFrame(flat_runs),
    )


def _select_window(window_summary: pd.DataFrame, window: str) -> dict[str, Any]:
    selected = window_summary[window_summary["window"].eq(window)]
    if selected.empty:
        raise ValueError(f"missing required window: {window}")
    return selected.iloc[0].to_dict()


def _number_or_zero(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_official_replay_inputs(card_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics_path = card_dir / "official_replay_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    daily_by_window: dict[str, pd.DataFrame] = {}
    for window in [*WF_WINDOWS, YTD_WINDOW]:
        path = card_dir / f"{window}_daily.csv"
        if path.exists():
            daily_by_window[window] = pd.read_csv(path, parse_dates=[0])
    if not daily_by_window:
        raise ValueError(f"no daily CSVs found in {card_dir}")
    return metrics, daily_by_window


def write_budget_usage_gap_diagnostic(
    *,
    source_card_dir: Path,
    output_dir: Path,
    required_soxlbacktest_return_pct: float,
    required_best_return_pct: float,
    max_walkforward_drawdown_pct: float,
) -> BudgetUsageGapDiagnostic:
    metrics, daily_by_window = load_official_replay_inputs(source_card_dir)
    diagnostic = build_budget_usage_gap_diagnostic(
        metrics=metrics,
        daily_by_window=daily_by_window,
        required_soxlbacktest_return_pct=required_soxlbacktest_return_pct,
        required_best_return_pct=required_best_return_pct,
        max_walkforward_drawdown_pct=max_walkforward_drawdown_pct,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(diagnostic.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostic.window_summary.to_csv(output_dir / "window_gap_summary.csv", index=False)
    diagnostic.flat_positive_runs.to_csv(output_dir / "flat_positive_runs.csv", index=False)
    return diagnostic
