from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FACTOR_LABELS = {
    "soxl_ret20": "SOXL 20日动量",
    "soxx_ret20": "SOXX 20日动量",
    "relative_momentum20": "杠杆相对动量",
    "soxx_trend60": "SOXX 60日趋势",
    "trend_efficiency_20": "趋势效率",
    "levered_decay_20": "杠杆衰减",
    "soxx_vol20": "SOXX 20日波动",
    "soxx_drawdown60": "SOXX 60日回撤",
    "vix": "VIX",
    "vix_z_60": "VIX 60日Z分",
    "rv20_expansion": "实现波动扩张",
    "chop_breakdown_risk": "震荡破位风险",
    "high_chop_pressure": "高震荡压力",
    "cross_dispersion_20": "跨资产离散度",
}


@dataclass(frozen=True)
class Stage1DashboardSource:
    root: Path
    summary: dict[str, Any]
    report_path: Path

    @property
    def recorder_id(self) -> str:
        return str(self.summary["recorder_id"])


def _resolve_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def resolve_stage1_dashboard_source(
    root: Path | str | None = None,
) -> Stage1DashboardSource:
    repo_root = _resolve_root(root)
    card = repo_root / "production" / "stage1_futu_sim_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    if summary.get("artifact_policy", {}).get("policy") != "slim_card_recorder_first":
        raise ValueError("blocked: Stage1 dashboard requires recorder-first artifacts")
    recorder_dir = Path(summary["recorder_local_dir"]).resolve()
    report_path = recorder_dir / "artifacts" / "stage1_futu_sim_v1" / "portfolio_report.csv"
    if not report_path.is_file():
        raise ValueError("blocked: Stage1 recorder portfolio report is missing")
    if recorder_dir not in report_path.parents:
        raise ValueError("blocked: Stage1 report resolved outside the pinned recorder")
    return Stage1DashboardSource(repo_root, summary, report_path)


def _risk_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    wealth = (1.0 + values).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / peak - 1.0
    years = max(len(values) / 252.0, 1.0 / 252.0)
    annualized = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(values.std(ddof=0) * np.sqrt(252.0))
    return {
        "return_pct": float((wealth.iloc[-1] - 1.0) * 100.0),
        "annualized_return_pct": annualized * 100.0,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "volatility_pct": volatility * 100.0,
        "sharpe": annualized / volatility if volatility > 0 else 0.0,
    }


def _normalise_factor_changes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    changes = frame.loc[:, columns].diff()
    scale = changes.rolling(60, min_periods=20).std().replace(0.0, np.nan)
    return (changes / scale).replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _runtime_file(runtime_dir: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError("blocked: Stage1 runtime filename is invalid")
    path = (runtime_dir / filename).resolve()
    if runtime_dir.resolve() not in path.parents:
        raise ValueError("blocked: Stage1 runtime file resolved outside its directory")
    return path


def _run_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    ledger = payload.get("ledger") if isinstance(payload.get("ledger"), dict) else {}
    execution = (
        payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    )
    return {
        "status": payload.get("status"),
        "slot": payload.get("slot"),
        "reason": payload.get("reason"),
        "recorded_at": payload.get("recorded_at"),
        "order_count": len(execution.get("orders", []))
        if isinstance(execution.get("orders"), list)
        else 0,
        "equity": ledger.get("equity"),
        "drawdown": ledger.get("drawdown"),
        "positions": ledger.get("positions"),
    }


def build_stage1_paper_simulation_status(
    *, root: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    """Build a read-only operational view from the registered paper-cycle runtime."""
    execution = summary.get("execution", {})
    manifest_text = str(execution.get("cycle_manifest", ""))
    expected_runtime = (root / "runtime" / "soxl" / "stage1_futu_sim").resolve()
    manifest_path = (root / manifest_text).resolve() if manifest_text else None
    if (
        manifest_path is None
        or manifest_path.parent != expected_runtime
        or manifest_path.name != "paper_cycle.json"
    ):
        return {
            "status": "needs_data",
            "reason": "missing_registered_paper_cycle_manifest",
            "source": "registered_stage1_runtime",
        }
    cycle = _read_json(manifest_path)
    if cycle is None:
        return {
            "status": "needs_data",
            "reason": "paper_cycle_manifest_unavailable",
            "source": "registered_stage1_runtime",
        }

    ledger = _read_json(
        _runtime_file(expected_runtime, str(cycle.get("ledger_file", "")))
    )
    execution_state = _read_json(
        _runtime_file(expected_runtime, str(cycle.get("execution_state_file", "")))
    )
    if ledger is None or execution_state is None:
        return {
            "status": "needs_data",
            "reason": "active_cycle_ledger_or_execution_state_unavailable",
            "source": "registered_stage1_runtime",
            "cycle": {
                "cycle_id": cycle.get("cycle_id"),
                "status": cycle.get("status"),
            },
        }

    latest_target = summary.get("latest_target", {})
    target_is_executable = (
        latest_target.get("status") == "ready"
        and latest_target.get("controller_state_source") == "paper_ledger_cycle"
    )
    weights = latest_target.get("weights", {}) if target_is_executable else {}
    positions = {
        symbol: int(dict(ledger.get("positions", {})).get(symbol, 0))
        for symbol in ("SOXL", "SOXX")
    }
    prices = {
        symbol: float(dict(ledger.get("prices", {})).get(symbol, 0.0))
        for symbol in ("SOXL", "SOXX")
    }
    equity = float(ledger.get("equity", 0.0))
    approximate_quantities = {
        symbol: int(equity * float(weights.get(symbol, 0.0)) / prices[symbol])
        if target_is_executable and equity > 0.0 and prices[symbol] > 0.0
        else None
        for symbol in ("SOXL", "SOXX")
    }
    quantity_deltas = {
        symbol: approximate_quantities[symbol] - positions[symbol]
        if approximate_quantities[symbol] is not None
        else None
        for symbol in ("SOXL", "SOXX")
    }

    completed_slots = execution_state.get("completed_slots", {})
    pending_slots = execution_state.get("pending_slots", {})
    if not isinstance(completed_slots, dict) or not isinstance(pending_slots, dict):
        raise ValueError("blocked: invalid Stage1 simulated execution state")
    completed_rows = [value for value in completed_slots.values() if isinstance(value, dict)]
    latest_completed = (
        max(completed_rows, key=lambda row: str(row.get("recorded_at", "")))
        if completed_rows
        else None
    )
    repair = cycle.get("execution_repair", {})
    contaminated = repair.get("prior_recovery_slots_contaminated", [])
    if not isinstance(contaminated, list):
        contaminated = []

    return {
        "status": "repair_armed"
        if repair.get("status") == "armed"
        else "ready",
        "source": "registered_stage1_runtime",
        "cycle": {
            "cycle_id": cycle.get("cycle_id"),
            "status": cycle.get("status"),
            "inception_date": cycle.get("inception_date"),
            "repair_status": repair.get("status"),
            "repair_effective_session": repair.get("effective_session_date"),
            "repair_commit": repair.get("applied_commit"),
            "history_policy": repair.get("history_policy"),
        },
        "ledger": {
            "cash": float(ledger.get("cash", 0.0)),
            "equity": equity,
            "peak_equity": float(ledger.get("peak_equity", 0.0)),
            "drawdown": float(ledger.get("drawdown", 0.0)),
            "total_inferred_fees": float(ledger.get("total_inferred_fees", 0.0)),
            "positions": positions,
            "prices": prices,
            "updated_at": ledger.get("updated_at"),
        },
        "target": {
            "status": latest_target.get("status")
            if target_is_executable
            else "pending_live_open_recompute",
            "signal_date": latest_target.get("signal_date"),
            "weights": weights if target_is_executable else None,
            "controller_state_source": latest_target.get("controller_state_source"),
            "executable": target_is_executable,
            "recovery_action": bool(latest_target.get("recovery_action")),
            "approximate_quantities_at_last_mark": approximate_quantities,
            "quantity_deltas_at_last_mark": quantity_deltas,
            "sizing_note": (
                "Open quantities are shown only for a paper_ledger_cycle target; "
                "otherwise the paper target is recomputed from live prices and "
                "reconciled sleeve equity at the open"
            ),
        },
        "execution_sync": {
            "completed_slot_count": len(completed_slots),
            "pending_slot_count": len(pending_slots),
            "recovery_slot_count": sum(
                str(key).endswith(":recovery") for key in completed_slots
            ),
            "contaminated_recovery_slot_count": len(contaminated),
            "latest_completed": {
                "slot": latest_completed.get("slot"),
                "signal_date": latest_completed.get("signal_date"),
                "status": latest_completed.get("status"),
                "recorded_at": latest_completed.get("recorded_at"),
                "target_quantities": latest_completed.get("target_quantities"),
            }
            if latest_completed
            else None,
        },
        "scheduler": {
            "latest_run": _run_summary(_read_json(expected_runtime / "latest_run.json")),
            "latest_actionable_run": _run_summary(
                _read_json(expected_runtime / "latest_actionable_run.json")
            ),
        },
        "policy": {
            "open": "rebalance_to_fresh_target",
            "recovery": "rebalance_only_when_recovery_action_true_otherwise_no_action",
            "manual_start": execution.get("manual_start", {}),
            "history": "observed_fills_are_preserved_and_not_rewritten",
            "real_broker_order": False,
        },
    }


@lru_cache(maxsize=2)
def _available_stage1_trade_range(
    root_text: str, recorder_id: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    source = resolve_stage1_dashboard_source(Path(root_text))
    if source.recorder_id != recorder_id:
        raise ValueError("blocked: Stage1 recorder changed while loading trade range")
    dates = pd.read_csv(source.report_path, usecols=["trade_date"])["trade_date"]
    dates = pd.to_datetime(dates, errors="coerce").dropna()
    if dates.empty:
        raise ValueError("blocked: Stage1 recorder contains no completed trade dates")
    return dates.min().normalize(), dates.max().normalize()


def build_stage1_dashboard_payload(
    *,
    root: Path | str | None = None,
    start: str | None = None,
) -> dict[str, Any]:
    from src.workflow.stage1_futu_sim import replay_stage1_simulation_from_start

    source = resolve_stage1_dashboard_source(root)
    available_start, available_end = _available_stage1_trade_range(
        source.root.as_posix(), source.recorder_id
    )
    requested_start = pd.Timestamp(start).normalize() if start else available_start
    start_ts = max(requested_start, available_start)
    if start_ts > available_end:
        raise ValueError("simulation start date is after the latest completed trade session")
    replay, inference, simulation_audit = replay_stage1_simulation_from_start(
        root=source.root,
        start=start_ts.date().isoformat(),
        stage1_recorder_id=source.recorder_id,
    )
    selected = replay.set_index("signal_date").sort_index()
    selected.index = pd.to_datetime(selected.index)
    inference.index = pd.to_datetime(inference.index)
    factors = inference.reindex(selected.index)
    if selected.empty:
        raise ValueError("simulation start date contains no Stage1 observations")

    strategy = pd.to_numeric(selected["strategy_return"], errors="coerce").fillna(0.0)
    soxl = pd.to_numeric(selected["soxl_return"], errors="coerce").fillna(0.0)
    soxx = pd.to_numeric(selected["soxx_return"], errors="coerce").fillna(0.0)
    nav = pd.DataFrame(
        {
            "Stage1": (1.0 + strategy).cumprod(),
            "SOXL": (1.0 + soxl).cumprod(),
            "SOXX": (1.0 + soxx).cumprod(),
        },
        index=selected.index,
    )
    drawdowns = nav.div(nav.cummax().clip(lower=1.0)).sub(1.0)
    weight_columns = ["soxl_weight", "soxx_weight", "cash_weight"]
    weights = selected.loc[:, weight_columns].apply(pd.to_numeric, errors="coerce")
    trade_delta = selected.loc[
        :, ["soxl_trade_delta", "soxx_trade_delta", "cash_trade_delta"]
    ].apply(pd.to_numeric, errors="coerce")
    trade_delta.columns = weight_columns
    turnover = pd.to_numeric(selected["turnover"], errors="coerce").fillna(0.0)

    factor_columns = [column for column in FACTOR_LABELS if column in factors.columns]
    factor_values = factors.loc[:, factor_columns].apply(pd.to_numeric, errors="coerce")
    full_factor_values = inference.loc[:, factor_columns].apply(pd.to_numeric, errors="coerce")
    factor_z = _normalise_factor_changes(full_factor_values, factor_columns).reindex(
        selected.index
    )
    events: list[dict[str, Any]] = []
    initial_trade = selected["initial_trade"].fillna(False).astype(bool)
    event_mask = (
        initial_trade
        | (turnover > 1e-10)
        | selected["recovery_action"].fillna(False).astype(bool)
    )
    for signal_date in selected.index[event_mask]:
        delta = trade_delta.loc[signal_date]
        is_initial = bool(initial_trade.loc[signal_date])
        invested_delta = float(delta["soxl_weight"] + delta["soxx_weight"])
        direction = (
            "初始建仓"
            if is_initial
            else "买入/加仓"
            if invested_delta > 1e-8
            else "卖出/减仓"
            if invested_delta < -1e-8
            else "再平衡"
        )
        ranked = factor_z.loc[signal_date].dropna().sort_values(
            key=lambda values: values.abs(), ascending=False
        )
        drivers = [
            {
                "factor": column,
                "label": FACTOR_LABELS[column],
                "change_z": float(ranked[column]),
                "value": (
                    float(factor_values.loc[signal_date, column])
                    if pd.notna(factor_values.loc[signal_date, column])
                    else None
                ),
            }
            for column in ranked.index[:4]
        ]
        events.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "trade_date": pd.Timestamp(selected.loc[signal_date, "trade_date"]).date().isoformat(),
                "direction": direction,
                "turnover": float(turnover.loc[signal_date]),
                "soxl_delta": float(delta["soxl_weight"]),
                "soxx_delta": float(delta["soxx_weight"]),
                "cash_delta": float(delta["cash_weight"]),
                "initial_trade": is_initial,
                "recovery_action": bool(selected.loc[signal_date, "recovery_action"]),
                "factors": drivers,
            }
        )

    rows = []
    for signal_date in selected.index:
        row = selected.loc[signal_date]
        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "trade_date": pd.Timestamp(row["trade_date"]).date().isoformat(),
                "exit_date": pd.Timestamp(row["exit_date"]).date().isoformat(),
                "stage1_nav": float(nav.loc[signal_date, "Stage1"]),
                "soxl_nav": float(nav.loc[signal_date, "SOXL"]),
                "soxx_nav": float(nav.loc[signal_date, "SOXX"]),
                "stage1_drawdown": float(drawdowns.loc[signal_date, "Stage1"]),
                "soxl_drawdown": float(drawdowns.loc[signal_date, "SOXL"]),
                "soxx_drawdown": float(drawdowns.loc[signal_date, "SOXX"]),
                "strategy_return": float(strategy.loc[signal_date]),
                "soxl_return": float(soxl.loc[signal_date]),
                "soxx_return": float(soxx.loc[signal_date]),
                "soxl_weight": float(row["soxl_weight"]),
                "soxx_weight": float(row["soxx_weight"]),
                "cash_weight": float(row["cash_weight"]),
                "turnover": float(turnover.loc[signal_date]),
                "initial_trade": bool(row["initial_trade"]),
                "recovery_action": bool(row["recovery_action"]),
            }
        )

    first_completed = selected.iloc[0]
    last_completed = selected.iloc[-1]
    start_trade_date = pd.Timestamp(first_completed["trade_date"])
    end_trade_date = pd.Timestamp(last_completed["trade_date"])
    portfolio_drawdown = float(-drawdowns.iloc[-1]["Stage1"])
    initial_open_weights = {
        "SOXL": float(first_completed["open_soxl_weight"]),
        "SOXX": float(first_completed["open_soxx_weight"]),
        "cash": float(first_completed["open_cash_weight"]),
    }
    latest_weights = {
        "SOXL": float(last_completed["soxl_weight"]),
        "SOXX": float(last_completed["soxx_weight"]),
        "cash": float(last_completed["cash_weight"]),
    }
    total_turnover = float(turnover.sum())
    event_turnover = turnover.loc[event_mask]
    return {
        "status": "ready",
        "source": "qlib_recorder",
        "recorder_id": source.recorder_id,
        "artifact_policy": source.summary["artifact_policy"]["policy"],
        "artifact": "artifacts/stage1_futu_sim_v1/portfolio_report.csv",
        "available_range": {
            "start": available_start.date().isoformat(),
            "end": available_end.date().isoformat(),
        },
        "simulation_range": {
            "requested_start": requested_start.date().isoformat(),
            "start": start_trade_date.date().isoformat(),
            "end": end_trade_date.date().isoformat(),
            "start_signal_date": selected.index[0].date().isoformat(),
            "end_signal_date": selected.index[-1].date().isoformat(),
            "rows": len(selected),
        },
        "window_summary": {
            "positive_day_rate_pct": float(strategy.gt(0.0).mean() * 100.0),
            "total_turnover": total_turnover,
            "average_event_turnover": float(event_turnover.mean()) if len(event_turnover) else 0.0,
            "trade_event_count": len(events),
            "initial_trade_count": sum(event["initial_trade"] for event in events),
            "buy_event_count": sum(event["direction"] == "买入/加仓" for event in events),
            "sell_event_count": sum(event["direction"] == "卖出/减仓" for event in events),
            "rebalance_event_count": sum(event["direction"] == "再平衡" for event in events),
            "recovery_event_count": sum(event["recovery_action"] for event in events),
            "average_soxl_weight_pct": float(weights["soxl_weight"].mean() * 100.0),
            "average_soxx_weight_pct": float(weights["soxx_weight"].mean() * 100.0),
            "average_cash_weight_pct": float(weights["cash_weight"].mean() * 100.0),
        },
        "metrics": {
            "Stage1": _risk_metrics(strategy),
            "SOXL": _risk_metrics(soxl),
            "SOXX": _risk_metrics(soxx),
        },
        "last_completed": {
            "signal_date": selected.index[-1].date().isoformat(),
            "trade_date": pd.Timestamp(last_completed["trade_date"]).date().isoformat(),
            "weights": {
                "SOXL": float(last_completed["soxl_weight"]),
                "SOXX": float(last_completed["soxx_weight"]),
                "cash": float(last_completed["cash_weight"]),
            },
            "turnover": float(turnover.iloc[-1]),
            "recovery_action": bool(last_completed["recovery_action"]),
        },
        "simulation": {
            "status": "fresh_cash_replay_complete",
            "mode": simulation_audit["simulation_mode"],
            "requested_start": simulation_audit["requested_start"],
            "start_signal_date": selected.index[0].date().isoformat(),
            "start_trade_date": start_trade_date.date().isoformat(),
            "end_signal_date": selected.index[-1].date().isoformat(),
            "end_trade_date": end_trade_date.date().isoformat(),
            "initial_state": simulation_audit["initial_state"],
            "initial_trade": {
                "recorded": bool(first_completed["initial_trade"]),
                "weights": initial_open_weights,
                "turnover": float(first_completed["turnover"]),
                "transaction_cost_bps": float(first_completed["cost_bps"]),
            },
            "latest_state": {
                "weights": latest_weights,
                "portfolio_drawdown": portfolio_drawdown,
                "current_beta_cap": float(last_completed["current_beta_cap"]),
                "recovery_probability": float(last_completed["recovery_probability"])
                if pd.notna(last_completed["recovery_probability"])
                else None,
                "recovery_action": bool(last_completed["recovery_action"]),
            },
            "evidence": {
                "source": "qlib_recorder_pinned_frozen_model_replay",
                "source_model_frozen": bool(
                    simulation_audit.get("source_model_frozen", False)
                ),
                "source_training_end": simulation_audit.get("source_training_end"),
                "source_recorder_id": simulation_audit.get("source_recorder_id"),
                "frozen_stage1_recorder_id": simulation_audit.get(
                    "stage1_recorder_id"
                ),
                "production_recorder_id": simulation_audit.get(
                    "production_recorder_id"
                ),
                "qlib_source_api": simulation_audit.get("market_audit", {}).get(
                    "source_api"
                ),
            },
        },
        "paper_simulation": build_stage1_paper_simulation_status(
            root=source.root,
            summary=source.summary,
        ),
        "rows": rows,
        "trade_events": events,
        "factor_labels": FACTOR_LABELS,
        "factor_disclaimer": "因子变化是交易点附近的标准化诊断，属于非因果归因，也不是精确 SHAP。",
    }
