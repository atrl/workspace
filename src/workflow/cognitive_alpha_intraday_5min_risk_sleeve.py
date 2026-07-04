from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame
from src.workflow.cognitive_alpha_intraday_local_fallback_overlay import (
    EXPERIMENT as LOCAL_FALLBACK_EXPERIMENT,
    INTRADAY_SOURCE_CARD,
    _fit_local_intraday_walk_forward,
    _load_intraday_label_frame,
)
from src.workflow.cognitive_alpha_intraday_overlay import (
    CRASH_THRESHOLD,
    DAILY_EXECUTION_TIMING,
    DAILY_SOURCE_CARD,
    OPEN_BUY_DUMP_THRESHOLD,
    PREDICTION_AVAILABLE_GATE_COL,
    QLIB_DOCS,
    QLIB_NATIVE_API,
    REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS,
    REQUIRED_WALK_FORWARD_START,
    _date_or_none,
    _daily_walk_forward_metadata,
    _json_ready_dict,
    _json_ready_records,
    _load_daily_open_prices,
    _load_daily_selected_positions,
    _normalize_daily_positions,
    _normalize_daily_prices,
    _read_json,
    _read_optional_json,
    _recorder_files,
    _remove_transient_code_snapshots,
    _resolve_root,
    _segment_return_metrics,
    _timestamp_or_none,
    summarize_execution_risk_diagnostics,
    summarize_overlay_metrics,
)
from src.workflow.intraday_bar_model import QLIB_FIELDS, _wide_feature_frame
from src.workflow.intraday_execution_label_model import (
    LABEL_POLICY,
    MODEL_NAMES,
    REQUESTED_EXTERNAL_CONTEXT_SYMBOLS,
    build_external_context_feature_frame,
)


EXPERIMENT = "cognitive_alpha_intraday_5min_risk_sleeve_v1"
REQUIRED_COST_BPS = (20.0, 50.0)
RISK_SLEEVE_VARIANTS = (
    "risk_exit_full_5m",
    "risk_exit_full_5m_reenter_close",
    "risk_reduce_half_5m",
    "risk_reduce_quarter_5m",
    "risk_reduce_quarter_5m_reenter_close",
    "risk_reduce_eighth_5m",
)
TRIGGER_MODES = (
    "drawdown_only",
    "risk_score_only",
    "risk_score_or_drawdown",
    "risk_score_and_drawdown",
    "context_stress_or_drawdown",
)
RISK_SLEEVE_EXECUTION_DELAY_BARS = 1
RISK_SLEEVE_EXECUTION_PRICE = "open"
RISK_SLEEVE_EXECUTION_TIMING = "daily_next_open_entry_then_5min_close_trigger_next_5min_open_reduce_or_exit"
LOCAL_CONTEXT_THRESHOLD_COLUMNS = (
    "soxl_vwap_dist",
    "soxx_cum_ret_from_open",
    "soxl_rv_6",
    "soxl_minus_3x_soxx_ret",
    "open_gap_reclaim_quality",
    "soxl_trend_quality_60m",
    "soxl_volume_price_absorption",
    "leveraged_replication_absorption",
)
EXTERNAL_CONTEXT_THRESHOLD_COLUMNS = (
    "qqq_smh_soxx_lead_lag_3",
    "semis_breadth_mom3_pos",
    "semis_vwap_breadth",
)
CONTEXT_THRESHOLD_COLUMNS = (*LOCAL_CONTEXT_THRESHOLD_COLUMNS, *EXTERNAL_CONTEXT_THRESHOLD_COLUMNS)
CONTEXT_STRESS_UPPER_COLUMNS = (
    "soxl_rv_6",
    "soxl_volume_price_absorption",
    "leveraged_replication_absorption",
)


@dataclass(frozen=True)
class CognitiveAlphaIntraday5MinRiskSleeveSpec:
    root: Path
    experiment: str = EXPERIMENT
    market: str = "all"
    region: str = "us"
    as_of: str = "2026-07-02"
    selection_cost_bps: float = 20.0
    required_cost_bps: tuple[float, ...] = REQUIRED_COST_BPS
    source_daily_card: str = DAILY_SOURCE_CARD
    source_intraday_card: str = INTRADAY_SOURCE_CARD
    source_local_fallback_card: str = f"research/{LOCAL_FALLBACK_EXPERIMENT}"

    @property
    def output_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def recorder_uri(self) -> Path:
        return self.output_dir / "qlib_recorder"

    @property
    def daily_card_dir(self) -> Path:
        return self.root / self.source_daily_card

    @property
    def intraday_card_dir(self) -> Path:
        return self.root / self.source_intraday_card

    @property
    def local_fallback_card_dir(self) -> Path:
        return self.root / self.source_local_fallback_card

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"

    @property
    def intraday_5min_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_5min"

    @property
    def external_context_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_external_intraday_15min"

    @property
    def external_context_card_dir(self) -> Path:
        return self.root / "research" / "intraday_external_context_data_v1"


def simulate_5min_risk_sleeve(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    bars_5min: pd.DataFrame,
    intraday_scores: pd.DataFrame | None = None,
    drawdown_threshold: float = -0.02,
    risk_score_threshold: float | None = None,
    trigger_mode: str = "drawdown_only",
    cost_bps: float = 0.0,
    risk_score_col: str = "risk_score",
    gate_col: str = PREDICTION_AVAILABLE_GATE_COL,
    execution_delay_bars: int = 0,
    execution_price: str = "close",
    context_thresholds: dict[str, float] | None = None,
    context_trigger_source: str = "manual",
    variants: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if trigger_mode not in {
        "drawdown_only",
        "risk_score_only",
        "risk_score_or_drawdown",
        "risk_score_and_drawdown",
        "context_stress_or_drawdown",
    }:
        raise ValueError(f"unsupported 5min risk trigger mode: {trigger_mode}")
    execution_delay_bars = int(execution_delay_bars)
    if execution_delay_bars < 0:
        raise ValueError("execution_delay_bars must be non-negative")
    if execution_price not in {"open", "close"}:
        raise ValueError(f"unsupported 5min execution price: {execution_price}")
    context_thresholds = _normalize_context_thresholds(context_thresholds)
    sleeve_variants = _resolve_requested_risk_sleeve_variants(variants)
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    bars = _normalize_5min_bars(bars_5min, intraday_scores=intraday_scores, risk_score_col=risk_score_col, gate_col=gate_col)
    bars_by_date = {
        date: group.sort_values("datetime").reset_index(drop=True)
        for date, group in bars.groupby("date", sort=True)
    }
    price_index = pd.Index(prices["datetime"])
    rows: list[dict[str, Any]] = []
    for daily_row in daily_frame.to_dict("records"):
        decision_date = pd.Timestamp(daily_row["datetime"]).normalize()
        if decision_date not in set(price_index):
            rows.extend(
                _missing_5min_rows(
                    daily_row,
                    cost_bps=float(cost_bps),
                    reason="missing_decision_open",
                    execution_delay_bars=execution_delay_bars,
                    execution_price=execution_price,
                    variants=sleeve_variants,
                )
            )
            continue
        location = int(price_index.get_loc(decision_date))
        if isinstance(location, slice):
            location = int(location.start)
        entry_date = prices.iloc[location + 1]["datetime"] if location + 1 < len(prices) else pd.NaT
        exit_date = prices.iloc[location + 2]["datetime"] if location + 2 < len(prices) else pd.NaT
        target = float(daily_row.get("target_position", 0.0) or 0.0)
        source_return = float(daily_row.get("strategy_return", 0.0) or 0.0)
        execution_cost = abs(target) * float(cost_bps) / 10000.0
        common = {
            "datetime": decision_date.date().isoformat(),
            "segment": str(daily_row.get("segment", "unknown")),
            **_daily_walk_forward_metadata(daily_row),
            "target_position": target,
            "cost_bps": float(cost_bps),
            "entry_date": _date_or_none(entry_date),
            "exit_date": _date_or_none(exit_date),
            "source_daily_strategy_return": source_return,
            "drawdown_threshold": float(drawdown_threshold),
            "risk_score_threshold": np.nan if risk_score_threshold is None else float(risk_score_threshold),
            "trigger_mode": trigger_mode,
            "context_trigger_source": context_trigger_source if trigger_mode == "context_stress_or_drawdown" else "disabled",
            **_context_threshold_row(context_thresholds),
            "risk_sleeve_execution_delay_bars": execution_delay_bars,
            "risk_sleeve_execution_price": execution_price,
        }
        if abs(target) <= 1e-12:
            rows.extend(_zero_5min_rows(common, variants=sleeve_variants))
            continue
        if pd.isna(entry_date) or pd.isna(exit_date):
            rows.extend(
                _missing_5min_rows(
                    daily_row,
                    cost_bps=float(cost_bps),
                    reason="missing_entry_or_exit_open",
                    execution_delay_bars=execution_delay_bars,
                    execution_price=execution_price,
                    variants=sleeve_variants,
                )
            )
            continue
        entry_open = float(prices.iloc[location + 1]["open"])
        exit_open = float(prices.iloc[location + 2]["open"])
        baseline_return = source_return - execution_cost
        entry_key = pd.Timestamp(entry_date).date().isoformat()
        entry_bars = bars_by_date.get(entry_key, pd.DataFrame())
        entry_close = float(entry_bars["close"].dropna().iloc[-1]) if not entry_bars.empty and entry_bars["close"].notna().any() else np.nan
        entry_close_time = entry_bars["datetime"].dropna().iloc[-1] if not entry_bars.empty and entry_bars["datetime"].notna().any() else pd.NaT
        baseline_entry_day = target * (entry_close / entry_open - 1.0) - execution_cost if not pd.isna(entry_close) else np.nan
        rows.append(
            {
                **common,
                "variant": "baseline_next_open",
                "entry_time": pd.Timestamp(entry_date).date().isoformat(),
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": entry_open,
                "exit_price": exit_open,
                "trigger_time": None,
                "trigger_price": np.nan,
                "trigger_drawdown_from_entry": np.nan,
                "trigger_risk_score": np.nan,
                **_empty_context_trigger_row(),
                "risk_execution_time": None,
                "risk_execution_price": np.nan,
                "risk_triggered": False,
                "gross_return": source_return,
                "strategy_return": baseline_return,
                "entry_day_close_time": _timestamp_or_none(entry_close_time),
                "entry_day_close_price": entry_close,
                "entry_day_strategy_return": baseline_entry_day,
                "skipped_reason": None,
            }
        )
        if entry_bars.empty:
            rows.extend(
                _no_trigger_variant_rows(
                    common,
                    entry_open=entry_open,
                    exit_open=exit_open,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    baseline_return=baseline_return,
                    entry_day_strategy_return=baseline_entry_day,
                    entry_day_close_time=entry_close_time,
                    entry_day_close_price=entry_close,
                    reason="missing_5min_entry_bars",
                    variants=sleeve_variants,
                )
            )
            continue
        trigger = _first_5min_trigger(
            entry_bars,
            target=target,
            entry_open=entry_open,
            drawdown_threshold=float(drawdown_threshold),
            risk_score_threshold=risk_score_threshold,
            trigger_mode=trigger_mode,
            risk_score_col=risk_score_col,
            gate_col=gate_col,
            context_thresholds=context_thresholds,
        )
        if trigger is None:
            rows.extend(
                _no_trigger_variant_rows(
                    common,
                    entry_open=entry_open,
                    exit_open=exit_open,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    baseline_return=baseline_return,
                    entry_day_strategy_return=baseline_entry_day,
                    entry_day_close_time=entry_close_time,
                    entry_day_close_price=entry_close,
                    reason=None,
                    variants=sleeve_variants,
                )
            )
            continue
        trigger_time = pd.Timestamp(trigger["datetime"])
        trigger_price = float(trigger["close"])
        trigger_pnl = target * (trigger_price / entry_open - 1.0)
        trigger_score = float(trigger[risk_score_col]) if pd.notna(trigger.get(risk_score_col)) else np.nan
        context_trigger_row = _context_trigger_row(trigger, context_thresholds=context_thresholds)
        execution_bar = _resolve_5min_execution_bar(
            entry_bars,
            trigger=trigger,
            execution_delay_bars=execution_delay_bars,
            execution_price=execution_price,
        )
        if execution_bar is None:
            rows.extend(
                _no_trigger_variant_rows(
                    common,
                    entry_open=entry_open,
                    exit_open=exit_open,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    baseline_return=baseline_return,
                    entry_day_strategy_return=baseline_entry_day,
                    entry_day_close_time=entry_close_time,
                    entry_day_close_price=entry_close,
                    reason="missing_delayed_execution_bar",
                    variants=sleeve_variants,
                )
            )
            continue
        risk_execution_time = pd.Timestamp(execution_bar["datetime"])
        risk_execution_price = float(execution_bar[execution_price])
        executed_pnl = target * (risk_execution_price / entry_open - 1.0)
        full_return = executed_pnl - execution_cost
        half_gross = target * (0.5 * (risk_execution_price / entry_open - 1.0) + 0.5 * (exit_open / entry_open - 1.0))
        half_return = half_gross - execution_cost
        quarter_gross = target * (0.25 * (risk_execution_price / entry_open - 1.0) + 0.75 * (exit_open / entry_open - 1.0))
        quarter_return = quarter_gross - execution_cost
        eighth_gross = target * (0.125 * (risk_execution_price / entry_open - 1.0) + 0.875 * (exit_open / entry_open - 1.0))
        eighth_return = eighth_gross - execution_cost
        full_reenter_gross = _reenter_close_gross_return(
            target=target,
            entry_open=entry_open,
            risk_execution_price=risk_execution_price,
            entry_close=entry_close,
            exit_open=exit_open,
            closed_fraction=1.0,
        )
        full_reenter_return = full_reenter_gross - execution_cost if not pd.isna(full_reenter_gross) else np.nan
        quarter_reenter_gross = _reenter_close_gross_return(
            target=target,
            entry_open=entry_open,
            risk_execution_price=risk_execution_price,
            entry_close=entry_close,
            exit_open=exit_open,
            closed_fraction=0.25,
        )
        quarter_reenter_return = quarter_reenter_gross - execution_cost if not pd.isna(quarter_reenter_gross) else np.nan
        half_entry_day = (
            target * (0.5 * (risk_execution_price / entry_open - 1.0) + 0.5 * (entry_close / entry_open - 1.0)) - execution_cost
            if not pd.isna(entry_close)
            else np.nan
        )
        quarter_entry_day = (
            target * (0.25 * (risk_execution_price / entry_open - 1.0) + 0.75 * (entry_close / entry_open - 1.0)) - execution_cost
            if not pd.isna(entry_close)
            else np.nan
        )
        eighth_entry_day = (
            target * (0.125 * (risk_execution_price / entry_open - 1.0) + 0.875 * (entry_close / entry_open - 1.0)) - execution_cost
            if not pd.isna(entry_close)
            else np.nan
        )
        triggered_rows = {
            "risk_exit_full_5m": {
                    **common,
                    "variant": "risk_exit_full_5m",
                    "entry_time": _timestamp_or_none(entry_bars.iloc[0]["datetime"]),
                    "exit_time": _timestamp_or_none(risk_execution_time),
                    "entry_price": entry_open,
                    "exit_price": risk_execution_price,
                    "trigger_time": _timestamp_or_none(trigger_time),
                    "trigger_price": trigger_price,
                    "trigger_drawdown_from_entry": trigger_pnl,
                    "trigger_risk_score": trigger_score,
                    **context_trigger_row,
                    "risk_execution_time": _timestamp_or_none(risk_execution_time),
                    "risk_execution_price": risk_execution_price,
                    "risk_triggered": True,
                    "gross_return": executed_pnl,
                    "strategy_return": full_return,
                    "entry_day_close_time": _timestamp_or_none(risk_execution_time),
                    "entry_day_close_price": risk_execution_price,
                    "entry_day_strategy_return": full_return,
                    "skipped_reason": None,
                },
            "risk_exit_full_5m_reenter_close": {
                    **common,
                    "variant": "risk_exit_full_5m_reenter_close",
                    "entry_time": _timestamp_or_none(entry_bars.iloc[0]["datetime"]),
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "trigger_time": _timestamp_or_none(trigger_time),
                    "trigger_price": trigger_price,
                    "trigger_drawdown_from_entry": trigger_pnl,
                    "trigger_risk_score": trigger_score,
                    **context_trigger_row,
                    "risk_execution_time": _timestamp_or_none(risk_execution_time),
                    "risk_execution_price": risk_execution_price,
                    "risk_triggered": True,
                    "gross_return": full_reenter_gross,
                    "strategy_return": full_reenter_return,
                    "entry_day_close_time": _timestamp_or_none(entry_close_time),
                    "entry_day_close_price": entry_close,
                    "entry_day_strategy_return": full_return,
                    "skipped_reason": None,
                },
            "risk_reduce_half_5m": {
                    **common,
                    "variant": "risk_reduce_half_5m",
                    "entry_time": _timestamp_or_none(entry_bars.iloc[0]["datetime"]),
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "trigger_time": _timestamp_or_none(trigger_time),
                    "trigger_price": trigger_price,
                    "trigger_drawdown_from_entry": trigger_pnl,
                    "trigger_risk_score": trigger_score,
                    **context_trigger_row,
                    "risk_execution_time": _timestamp_or_none(risk_execution_time),
                    "risk_execution_price": risk_execution_price,
                    "risk_triggered": True,
                    "gross_return": half_gross,
                    "strategy_return": half_return,
                    "entry_day_close_time": _timestamp_or_none(entry_close_time),
                    "entry_day_close_price": entry_close,
                    "entry_day_strategy_return": half_entry_day,
                    "skipped_reason": None,
                },
            "risk_reduce_quarter_5m": {
                    **common,
                    "variant": "risk_reduce_quarter_5m",
                    "entry_time": _timestamp_or_none(entry_bars.iloc[0]["datetime"]),
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "trigger_time": _timestamp_or_none(trigger_time),
                    "trigger_price": trigger_price,
                    "trigger_drawdown_from_entry": trigger_pnl,
                    "trigger_risk_score": trigger_score,
                    **context_trigger_row,
                    "risk_execution_time": _timestamp_or_none(risk_execution_time),
                    "risk_execution_price": risk_execution_price,
                    "risk_triggered": True,
                    "gross_return": quarter_gross,
                    "strategy_return": quarter_return,
                    "entry_day_close_time": _timestamp_or_none(entry_close_time),
                    "entry_day_close_price": entry_close,
                    "entry_day_strategy_return": quarter_entry_day,
                    "skipped_reason": None,
                },
            "risk_reduce_quarter_5m_reenter_close": {
                    **common,
                    "variant": "risk_reduce_quarter_5m_reenter_close",
                    "entry_time": _timestamp_or_none(entry_bars.iloc[0]["datetime"]),
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "trigger_time": _timestamp_or_none(trigger_time),
                    "trigger_price": trigger_price,
                    "trigger_drawdown_from_entry": trigger_pnl,
                    "trigger_risk_score": trigger_score,
                    **context_trigger_row,
                    "risk_execution_time": _timestamp_or_none(risk_execution_time),
                    "risk_execution_price": risk_execution_price,
                    "risk_triggered": True,
                    "gross_return": quarter_reenter_gross,
                    "strategy_return": quarter_reenter_return,
                    "entry_day_close_time": _timestamp_or_none(entry_close_time),
                    "entry_day_close_price": entry_close,
                    "entry_day_strategy_return": quarter_entry_day,
                    "skipped_reason": None,
                },
            "risk_reduce_eighth_5m": {
                    **common,
                    "variant": "risk_reduce_eighth_5m",
                    "entry_time": _timestamp_or_none(entry_bars.iloc[0]["datetime"]),
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "trigger_time": _timestamp_or_none(trigger_time),
                    "trigger_price": trigger_price,
                    "trigger_drawdown_from_entry": trigger_pnl,
                    "trigger_risk_score": trigger_score,
                    **context_trigger_row,
                    "risk_execution_time": _timestamp_or_none(risk_execution_time),
                    "risk_execution_price": risk_execution_price,
                    "risk_triggered": True,
                    "gross_return": eighth_gross,
                    "strategy_return": eighth_return,
                    "entry_day_close_time": _timestamp_or_none(entry_close_time),
                    "entry_day_close_price": entry_close,
                    "entry_day_strategy_return": eighth_entry_day,
                    "skipped_reason": None,
                },
        }
        rows.extend(triggered_rows[variant] for variant in sleeve_variants)
    return pd.DataFrame(rows)


def _resolve_requested_risk_sleeve_variants(variants: tuple[str, ...] | None) -> tuple[str, ...]:
    if variants is None:
        return RISK_SLEEVE_VARIANTS
    requested = tuple(str(variant) for variant in variants)
    unsupported = sorted(set(requested) - set(RISK_SLEEVE_VARIANTS))
    if unsupported:
        raise ValueError(f"unsupported 5min risk sleeve variants: {unsupported}")
    return requested


def _reenter_close_gross_return(
    *,
    target: float,
    entry_open: float,
    risk_execution_price: float,
    entry_close: float,
    exit_open: float,
    closed_fraction: float,
) -> float:
    if pd.isna(entry_close) or float(entry_close) <= 0.0:
        return np.nan
    closed_fraction = float(closed_fraction)
    held_fraction = 1.0 - closed_fraction
    held_leg = float(exit_open) / float(entry_open) - 1.0
    reentered_leg = (float(risk_execution_price) / float(entry_open)) * (float(exit_open) / float(entry_close)) - 1.0
    return float(target) * (held_fraction * held_leg + closed_fraction * reentered_leg)


def evaluate_5min_risk_sleeve_candidates(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    bars_5min: pd.DataFrame,
    intraday_scores: pd.DataFrame,
    cost_bps_values: tuple[float, ...] = REQUIRED_COST_BPS,
    risk_score_col: str = "risk_score",
    gate_col: str = PREDICTION_AVAILABLE_GATE_COL,
    execution_delay_bars: int = 0,
    execution_price: str = "close",
) -> pd.DataFrame:
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    bars = _normalize_5min_bars(bars_5min, intraday_scores=intraday_scores, risk_score_col=risk_score_col, gate_col=gate_col)
    policy_grid = _validation_policy_grid(daily_frame, daily_prices=prices, bars_5min=bars, risk_score_col=risk_score_col, gate_col=gate_col)
    rows: list[dict[str, Any]] = []
    for policy_group in _group_risk_sleeve_policies_by_trigger(policy_grid):
        replay_policy = policy_group[0]
        replay_variants = tuple(str(policy["variant"]) for policy in policy_group)
        trade_frames = [
            simulate_5min_risk_sleeve(
                daily_frame,
                daily_prices=prices,
                bars_5min=bars,
                intraday_scores=None,
                drawdown_threshold=float(replay_policy["drawdown_threshold"]),
                risk_score_threshold=replay_policy.get("risk_score_threshold"),
                trigger_mode=str(replay_policy["trigger_mode"]),
                cost_bps=float(cost_bps),
                risk_score_col=risk_score_col,
                gate_col=gate_col,
                execution_delay_bars=execution_delay_bars,
                execution_price=execution_price,
                context_thresholds=_context_thresholds_from_policy(replay_policy),
                context_trigger_source=str(replay_policy.get("context_trigger_source", "disabled")),
                variants=replay_variants,
            ).assign(cost_bps=float(cost_bps))
            for cost_bps in cost_bps_values
        ]
        trades = pd.concat(trade_frames, ignore_index=True)
        metrics = summarize_overlay_metrics(trades)
        risk = summarize_execution_risk_diagnostics(trades)
        for policy in policy_group:
            rows.extend(_policy_metric_rows(policy, metrics=metrics, risk=risk))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        [
            "passes_validation_target",
            "validation_max_drawdown_pct",
            "validation_return_retention_score",
            "validation_open_buy_dump_period_reduction",
            "validation_crash_period_reduction",
            "selection_score",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def _group_risk_sleeve_policies_by_trigger(policies: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for policy in policies:
        key = _risk_sleeve_trigger_key(policy)
        grouped.setdefault(key, []).append(policy)
    return list(grouped.values())


def _risk_sleeve_trigger_key(policy: dict[str, Any]) -> tuple[Any, ...]:
    def clean(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, float):
            return round(float(value), 12)
        return value

    return (
        clean(policy.get("trigger_mode")),
        clean(policy.get("drawdown_threshold")),
        clean(policy.get("risk_score_threshold")),
        clean(policy.get("context_trigger_source")),
        *tuple(clean(policy.get(f"context_{column}_threshold")) for column in CONTEXT_THRESHOLD_COLUMNS),
    )


def select_validation_robust_5min_risk_policy(
    candidates: pd.DataFrame,
    *,
    required_cost_bps: tuple[float, ...] = REQUIRED_COST_BPS,
    selection_cost_bps: float = 20.0,
) -> dict[str, Any]:
    priority = [
        "all_required_costs_pass_validation_target",
        "validation_max_drawdown_pct_desc",
        "validation_return_retention_score_desc",
        "validation_open_buy_dump_period_reduction_desc",
        "validation_crash_period_reduction_desc",
        "selection_score_desc",
    ]
    if candidates.empty:
        return {
            "policy_id": None,
            "selection_role": "5min_intraday_risk_sleeve",
            "selection_uses_test_segment": False,
            "selection_priority": priority,
            "validation_passes_target": False,
        }
    frame = candidates.copy()
    frame["cost_bps"] = pd.to_numeric(frame["cost_bps"], errors="coerce")
    required = {float(value) for value in required_cost_bps}
    robust_rows: list[pd.Series] = []
    required_rows_by_key: dict[Any, pd.DataFrame] = {}
    for policy_id, group in frame.groupby("policy_id", sort=False, dropna=False):
        costs = {float(value) for value in group["cost_bps"].dropna().unique()}
        if not required.issubset(costs):
            continue
        required_rows = group.loc[group["cost_bps"].isin(required)].copy()
        if not bool(required_rows["passes_validation_target"].astype(bool).all()):
            continue
        selected_cost = required_rows.loc[required_rows["cost_bps"].eq(float(selection_cost_bps))]
        if selected_cost.empty:
            continue
        selected = selected_cost.iloc[0].copy()
        selected["required_cost_min_validation_retention_pct"] = float(
            pd.to_numeric(required_rows["validation_return_retention_pct"], errors="coerce").min()
        )
        selected["required_cost_min_validation_drawdown_pct"] = float(
            pd.to_numeric(required_rows["validation_max_drawdown_pct"], errors="coerce").min()
        )
        robust_rows.append(selected)
        required_rows_by_key[policy_id] = required_rows.sort_values("cost_bps")
    if robust_rows:
        source = pd.DataFrame(robust_rows)
        validation_cost_robust = True
        sort_columns = [
            "validation_max_drawdown_pct",
            "validation_return_retention_score",
            "validation_open_buy_dump_period_reduction",
            "validation_crash_period_reduction",
            "validation_worst_entry_day_return_improvement_pct",
            "selection_score",
        ]
        ascending = [False, False, False, False, False, False]
    else:
        validation_pass = frame.loc[frame["passes_validation_target"].astype(bool)].copy()
        source = validation_pass if not validation_pass.empty else frame.copy()
        validation_cost_robust = False
        retention = pd.to_numeric(source["validation_return_retention_pct"], errors="coerce")
        source["validation_retention_gap_to_target_band"] = np.where(
            retention.between(85.0, 95.0),
            0.0,
            np.minimum((retention - 85.0).abs(), (retention - 95.0).abs()),
        )
        source["validation_risk_reduction_score"] = (
            pd.to_numeric(source["validation_open_buy_dump_period_reduction"], errors="coerce").fillna(0.0)
            + pd.to_numeric(source["validation_crash_period_reduction"], errors="coerce").fillna(0.0)
            + pd.to_numeric(source["validation_worst_entry_day_return_improvement_pct"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        sort_columns = [
            "validation_retention_gap_to_target_band",
            "validation_risk_reduction_pass",
            "validation_risk_reduction_score",
            "validation_max_drawdown_pct",
            "selection_score",
        ]
        ascending = [True, False, False, False, False]
    selected = source.sort_values(sort_columns, ascending=ascending).iloc[0]
    result = _json_ready_dict(selected.to_dict())
    result["selection_role"] = "5min_intraday_risk_sleeve"
    result["selection_segment"] = "validation"
    result["report_segment"] = "test"
    result["selection_uses_test_segment"] = False
    result["selection_priority"] = priority
    result["required_cost_bps"] = sorted(required)
    result["selection_cost_bps"] = float(selection_cost_bps)
    result["validation_cost_robust"] = bool(validation_cost_robust)
    result["validation_passes_target"] = bool(result.get("passes_validation_target", False))
    result["test_passes_target"] = bool(result.get("passes_test_target", False))
    required_rows = required_rows_by_key.get(result.get("policy_id"), pd.DataFrame())
    if not required_rows.empty:
        result["required_cost_validation_target_count"] = int(required_rows["passes_validation_target"].astype(bool).sum())
        result["required_cost_test_target_count"] = int(required_rows["passes_test_target"].astype(bool).sum())
        result["required_cost_validation_and_test_target_count"] = int(
            required_rows["passes_validation_and_test_target"].astype(bool).sum()
        )
        result["required_cost_results"] = _json_ready_records(required_rows)
    else:
        result["required_cost_validation_target_count"] = 0
        result["required_cost_test_target_count"] = 0
        result["required_cost_validation_and_test_target_count"] = 0
        result["required_cost_results"] = []
    return result


def run_cognitive_alpha_intraday_5min_risk_sleeve_card(
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    from qlib.workflow import R

    spec = CognitiveAlphaIntraday5MinRiskSleeveSpec(root=_resolve_root(root))
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)

    daily_summary = _read_json(spec.daily_card_dir / "summary.json")
    intraday_summary = _read_json(spec.intraday_card_dir / "summary.json")
    workflow_yaml = _write_workflow_yaml(spec, daily_summary=daily_summary, intraday_summary=intraday_summary)
    daily_positions = _load_daily_selected_positions(daily_summary, prefer_walk_forward=True)
    as_of = str(daily_summary.get("as_of") or spec.as_of)
    daily_prices = _load_daily_open_prices(
        spec,  # type: ignore[arg-type]
        start=daily_positions["datetime"].min().date().isoformat(),
        end=as_of,
    )
    (
        provider_meta_15m,
        fold_predictions,
        model_selection,
        feature_importance,
        feature_cols,
        label_audit,
        score_source,
    ) = _load_or_fit_lgb_xgb_fold_predictions(
        spec,
        as_of=as_of,
    )
    provider_meta_5m, bars_5min = _load_5min_bar_frame(spec, as_of=as_of)
    external_provider_meta, external_context_15m, external_context_audit = _load_external_15min_context_frame(spec, as_of=as_of)
    bars_5min, external_merge_audit = _attach_external_context_to_5min_bars(bars_5min, external_context_15m)
    external_context_audit["merge_audit"] = external_merge_audit
    annual_walk_forward, selected_costs, selected_trades, all_policy_selection, risk_analysis = _evaluate_annual_5min_risk_sleeve(
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        bars_5min=bars_5min,
        fold_predictions=fold_predictions,
        spec=spec,
        as_of=as_of,
    )
    selected_model = _selected_model_payload(annual_walk_forward)
    walk_forward_summary = _walk_forward_summary(annual_walk_forward)
    context_candidate_diagnostics = _risk_sleeve_context_candidate_diagnostics(
        all_policy_selection=all_policy_selection,
        annual_walk_forward=annual_walk_forward,
    )
    analysis_summary = _analysis_summary(
        spec,
        daily_summary=daily_summary,
        intraday_summary=intraday_summary,
        provider_meta_15m=provider_meta_15m,
        provider_meta_5m=provider_meta_5m,
        external_provider_meta=external_provider_meta,
        external_context_audit=external_context_audit,
        daily_positions=daily_positions,
        bars_5min=bars_5min,
        feature_cols=feature_cols,
        label_audit=label_audit,
        model_selection=model_selection,
        annual_walk_forward=annual_walk_forward,
        selected_policy_cost_sensitivity=selected_costs,
        risk_analysis=risk_analysis,
        selected_model=selected_model,
        walk_forward_summary=walk_forward_summary,
        context_candidate_diagnostics=context_candidate_diagnostics,
    )
    sidecars = _write_sidecars(
        spec,
        annual_walk_forward=annual_walk_forward,
        model_selection=model_selection,
        selected_policy_cost_sensitivity=selected_costs,
        risk_analysis=risk_analysis,
        feature_importance=feature_importance,
        selected_model=selected_model,
        analysis_summary=analysis_summary,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        sleeve_dir = upload_dir / "five_min_risk_sleeve"
        sleeve_dir.mkdir(parents=True, exist_ok=True)
        fold_predictions.to_csv(sleeve_dir / "intraday_fold_predictions.csv", index=False)
        all_policy_selection.to_csv(sleeve_dir / "all_policy_selection.csv.gz", index=False, compression="gzip")
        selected_trades.to_csv(sleeve_dir / "selected_trades.csv", index=False)
        daily_positions.to_csv(sleeve_dir / "daily_walk_forward_positions.csv", index=False)
        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                daily_source=spec.source_daily_card,
                intraday_source=spec.source_intraday_card,
                label_policy=LABEL_POLICY,
                risk_sleeve_frequency="5min",
                selection_segment="validation",
                report_segment="test",
                selection_uses_test_segment=False,
                required_cost_bps="20.0,50.0",
                risk_sleeve_execution_delay_bars=RISK_SLEEVE_EXECUTION_DELAY_BARS,
                risk_sleeve_execution_price=RISK_SLEEVE_EXECUTION_PRICE,
                risk_sleeve_context_trigger_features=",".join(CONTEXT_THRESHOLD_COLUMNS),
                risk_sleeve_external_context_features=",".join(EXTERNAL_CONTEXT_THRESHOLD_COLUMNS),
                walk_forward_policy=walk_forward_summary["policy"],
            )
            R.log_metrics(
                feasible_fold_count=float(len(walk_forward_summary["feasible_test_years"])),
                required_policy_satisfied=float(1.0 if walk_forward_summary["required_policy_satisfied"] else 0.0),
                target_pass_fold_count=float(len(walk_forward_summary["target_pass_test_years"])),
                risk_reduction_fold_count=float(len(walk_forward_summary["risk_reduction_test_years"])),
            )
            for file_name in [
                "intraday_fold_predictions.csv",
                "all_policy_selection.csv.gz",
                "selected_trades.csv",
                "daily_walk_forward_positions.csv",
            ]:
                recorder.log_artifact((sleeve_dir / file_name).as_posix(), artifact_path="five_min_risk_sleeve")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    summary = {
        "experiment": spec.experiment,
        "card_type": "intraday_5min_risk_sleeve_replay",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": _promotion_decision(walk_forward_summary),
        "as_of": as_of,
        "market": spec.market,
        "region": spec.region,
        "source_daily_model": {
            "card": spec.source_daily_card,
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing", DAILY_EXECUTION_TIMING),
            "return_col": daily_summary.get("position_model", {}).get("return_col", "open_to_open_return"),
        },
        "source_intraday_model": {
            "card": spec.source_intraday_card,
            "recorder_id": intraday_summary.get("recorder_id"),
            "label_policy_name": LABEL_POLICY,
            "score_source": score_source,
            "score_semantics": "probability_of_positive_1h_2h_execution_improvement",
            "risk_trigger_direction": "trigger_when_execution_improvement_probability_is_less_or_equal_validation_quantile",
            "models": MODEL_NAMES,
            "feature_set": {
                "source": "local_soxl_soxx_15m_lgb_xgb_triple_barrier",
                "feature_count": int(len(feature_cols)),
                "features": feature_cols,
            },
        },
        "provider_uris": {
            "daily": spec.daily_provider_uri.resolve().as_posix(),
            "15min": spec.intraday_provider_uri.resolve().as_posix(),
            "5min": spec.intraday_5min_provider_uri.resolve().as_posix(),
            "external_context_15min": spec.external_context_provider_uri.resolve().as_posix(),
        },
        "external_context": {
            "source_card": spec.external_context_card_dir.relative_to(spec.root).as_posix(),
            "provider_uri": spec.external_context_provider_uri.resolve().as_posix(),
            "requested_symbols": list(REQUESTED_EXTERNAL_CONTEXT_SYMBOLS),
            "feature_columns": list(EXTERNAL_CONTEXT_THRESHOLD_COLUMNS),
            "status": external_context_audit.get("status"),
            "merge_status": external_merge_audit.get("status"),
            "full_walk_forward_history_available": bool(
                external_context_audit.get("coverage_gap_analysis", {}).get("full_walk_forward_history_available", False)
            ),
            "missing_required_history_years": external_context_audit.get("coverage_gap_analysis", {}).get("missing_required_history_years", []),
            "external_context_available_5min_rows": external_merge_audit.get("external_context_available_5min_rows", 0),
        },
        "execution_overlay": {
            "frequency": "5min",
            "decision_model_timing": "daily_close_t_signal",
            "daily_execution_timing": DAILY_EXECUTION_TIMING,
            "risk_sleeve_timing": RISK_SLEEVE_EXECUTION_TIMING,
            "risk_sleeve_execution_delay_bars": RISK_SLEEVE_EXECUTION_DELAY_BARS,
            "risk_sleeve_execution_price": f"next_5min_{RISK_SLEEVE_EXECUTION_PRICE}",
            "open_to_open_baseline": True,
            "daily_entry_is_not_skipped": True,
            "variants": ["baseline_next_open", *RISK_SLEEVE_VARIANTS],
            "trigger_modes": list(TRIGGER_MODES),
            "context_trigger_features": list(CONTEXT_THRESHOLD_COLUMNS),
        },
        "model_selection": {
            "models": MODEL_NAMES,
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "required_cost_bps": [float(value) for value in spec.required_cost_bps],
            "return_retention_target_pct": [85.0, 95.0],
            "max_drawdown_floor_pct": -18.0,
            "uses_5min_context_trigger_candidates": True,
            "context_policy_candidate_rows": context_candidate_diagnostics["context_policy_candidate_rows"],
            "selected_context_trigger_years": context_candidate_diagnostics["selected_context_trigger_years"],
            "policy_candidate_rows": int(len(all_policy_selection)),
            "selected_policy_cost_rows": int(len(selected_costs)),
        },
        "walk_forward": walk_forward_summary,
        "selected_model": selected_model,
        "analysis_summary": analysis_summary,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API + ["LGBMClassifier", "XGBClassifier", LABEL_POLICY],
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_conclusion(spec, summary=summary, annual_walk_forward=annual_walk_forward)
    return summary


def _normalize_5min_bars(
    bars_5min: pd.DataFrame,
    *,
    intraday_scores: pd.DataFrame | None,
    risk_score_col: str,
    gate_col: str,
) -> pd.DataFrame:
    bars = bars_5min.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    if "date" not in bars.columns:
        bars["date"] = bars["datetime"].dt.date.astype(str)
    else:
        bars["date"] = pd.to_datetime(bars["date"]).dt.date.astype(str)
    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        risk_score_col,
        *CONTEXT_THRESHOLD_COLUMNS,
    ]:
        if column in bars.columns:
            bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if intraday_scores is not None and not intraday_scores.empty:
        scores = intraday_scores.copy()
        scores["datetime"] = pd.to_datetime(scores["datetime"])
        if "date" not in scores.columns:
            scores["date"] = scores["datetime"].dt.date.astype(str)
        else:
            scores["date"] = pd.to_datetime(scores["date"]).dt.date.astype(str)
        score_cols = ["datetime", "date", risk_score_col]
        if gate_col in scores.columns:
            score_cols.append(gate_col)
        scores = scores[score_cols].sort_values(["date", "datetime"])
        bars = pd.merge_asof(
            bars.sort_values(["date", "datetime"]),
            scores,
            on="datetime",
            by="date",
            direction="backward",
            suffixes=("", "_score"),
        )
        if f"{risk_score_col}_score" in bars.columns:
            bars[risk_score_col] = bars[f"{risk_score_col}_score"].combine_first(bars.get(risk_score_col))
            bars = bars.drop(columns=[f"{risk_score_col}_score"])
        if f"{gate_col}_score" in bars.columns:
            bars[gate_col] = bars[f"{gate_col}_score"].combine_first(bars.get(gate_col))
            bars = bars.drop(columns=[f"{gate_col}_score"])
    if risk_score_col not in bars.columns:
        bars[risk_score_col] = np.nan
    if gate_col not in bars.columns:
        bars[gate_col] = False
    bars[gate_col] = bars[gate_col].fillna(False).astype(bool)
    return bars.dropna(subset=["datetime", "open", "close"]).sort_values(["date", "datetime"]).reset_index(drop=True)


def _first_5min_trigger(
    bars: pd.DataFrame,
    *,
    target: float,
    entry_open: float,
    drawdown_threshold: float,
    risk_score_threshold: float | None,
    trigger_mode: str,
    risk_score_col: str,
    gate_col: str,
    context_thresholds: dict[str, float] | None,
) -> pd.Series | None:
    frame = bars.copy()
    frame["pnl_from_entry"] = float(target) * (pd.to_numeric(frame["close"], errors="coerce") / float(entry_open) - 1.0)
    drawdown_hit = frame["pnl_from_entry"].le(float(drawdown_threshold))
    score = pd.to_numeric(frame.get(risk_score_col), errors="coerce")
    score_hit = frame.get(gate_col, pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    score_hit &= score.le(float(risk_score_threshold)) if risk_score_threshold is not None else False
    context_hit = _context_stress_mask(frame, context_thresholds=context_thresholds)
    if trigger_mode == "drawdown_only":
        trigger_mask = drawdown_hit
    elif trigger_mode == "risk_score_only":
        trigger_mask = score_hit
    elif trigger_mode == "risk_score_or_drawdown":
        trigger_mask = score_hit | drawdown_hit
    elif trigger_mode == "context_stress_or_drawdown":
        trigger_mask = context_hit | drawdown_hit
    else:
        trigger_mask = score_hit & drawdown_hit
    trigger_rows = frame.loc[trigger_mask & frame["close"].notna()]
    if trigger_rows.empty:
        return None
    return trigger_rows.iloc[0]


def _normalize_context_thresholds(context_thresholds: dict[str, float] | None) -> dict[str, float]:
    if not context_thresholds:
        return {}
    normalized: dict[str, float] = {}
    for column in CONTEXT_THRESHOLD_COLUMNS:
        value = context_thresholds.get(column)
        if value is not None and pd.notna(value):
            normalized[column] = float(value)
    return normalized


def _context_stress_mask(
    frame: pd.DataFrame,
    *,
    context_thresholds: dict[str, float] | None,
) -> pd.Series:
    thresholds = _normalize_context_thresholds(context_thresholds)
    if "soxl_vwap_dist" not in thresholds or "soxl_vwap_dist" not in frame.columns:
        return pd.Series(False, index=frame.index)
    below_vwap = pd.to_numeric(frame["soxl_vwap_dist"], errors="coerce").le(thresholds["soxl_vwap_dist"])
    stress_masks: list[pd.Series] = []
    external_available = frame.get("external_context_available", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    for column in CONTEXT_THRESHOLD_COLUMNS:
        if column == "soxl_vwap_dist" or column not in thresholds or column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if column in CONTEXT_STRESS_UPPER_COLUMNS:
            mask = values.ge(thresholds[column])
        else:
            mask = values.le(thresholds[column])
        if column in EXTERNAL_CONTEXT_THRESHOLD_COLUMNS:
            mask = mask & external_available
        stress_masks.append(mask)
    if not stress_masks:
        return pd.Series(False, index=frame.index)
    stress = stress_masks[0].fillna(False)
    for mask in stress_masks[1:]:
        stress = stress | mask.fillna(False)
    return below_vwap.fillna(False) & stress


def _context_threshold_row(context_thresholds: dict[str, float]) -> dict[str, Any]:
    return {f"context_{column}_threshold": context_thresholds.get(column, np.nan) for column in CONTEXT_THRESHOLD_COLUMNS}


def _empty_context_trigger_row() -> dict[str, Any]:
    row = {f"context_{column}": np.nan for column in CONTEXT_THRESHOLD_COLUMNS}
    row["context_stress_hit"] = False
    return row


def _context_trigger_row(trigger: pd.Series, *, context_thresholds: dict[str, float]) -> dict[str, Any]:
    row = {f"context_{column}": float(trigger[column]) if column in trigger and pd.notna(trigger[column]) else np.nan for column in CONTEXT_THRESHOLD_COLUMNS}
    frame = pd.DataFrame([trigger.to_dict()])
    row["context_stress_hit"] = bool(_context_stress_mask(frame, context_thresholds=context_thresholds).iloc[0])
    return row


def _context_thresholds_from_policy(policy: dict[str, Any] | pd.Series) -> dict[str, float]:
    data = policy.to_dict() if isinstance(policy, pd.Series) else dict(policy)
    thresholds: dict[str, float] = {}
    for column in CONTEXT_THRESHOLD_COLUMNS:
        value = data.get(f"context_{column}_threshold")
        if value is not None and pd.notna(value):
            thresholds[column] = float(value)
    return thresholds


def _resolve_5min_execution_bar(
    bars: pd.DataFrame,
    *,
    trigger: pd.Series,
    execution_delay_bars: int,
    execution_price: str,
) -> pd.Series | None:
    trigger_time = pd.Timestamp(trigger["datetime"])
    trigger_positions = np.flatnonzero(pd.to_datetime(bars["datetime"]).eq(trigger_time).to_numpy())
    if len(trigger_positions) == 0:
        return None
    execution_position = int(trigger_positions[0]) + int(execution_delay_bars)
    if execution_position >= len(bars):
        return None
    execution_bar = bars.iloc[execution_position]
    if pd.isna(execution_bar.get(execution_price)):
        return None
    return execution_bar


def _zero_5min_rows(
    common: dict[str, Any],
    *,
    variants: tuple[str, ...] = RISK_SLEEVE_VARIANTS,
) -> list[dict[str, Any]]:
    rows = []
    for variant in ["baseline_next_open", *variants]:
        rows.append(
            {
                **common,
                "variant": variant,
                "entry_time": None,
                "exit_time": None,
                "entry_price": np.nan,
                "exit_price": np.nan,
                "trigger_time": None,
                "trigger_price": np.nan,
                "trigger_drawdown_from_entry": np.nan,
                "trigger_risk_score": np.nan,
                **_empty_context_trigger_row(),
                "risk_execution_time": None,
                "risk_execution_price": np.nan,
                "risk_triggered": False,
                "gross_return": 0.0,
                "strategy_return": 0.0,
                "entry_day_close_time": None,
                "entry_day_close_price": np.nan,
                "entry_day_strategy_return": np.nan,
                "skipped_reason": None,
            }
        )
    return rows


def _missing_5min_rows(
    daily_row: dict[str, Any],
    *,
    cost_bps: float,
    reason: str,
    execution_delay_bars: int = 0,
    execution_price: str = "close",
    variants: tuple[str, ...] = RISK_SLEEVE_VARIANTS,
) -> list[dict[str, Any]]:
    common = {
        "datetime": pd.Timestamp(daily_row["datetime"]).date().isoformat(),
        "segment": str(daily_row.get("segment", "unknown")),
        **_daily_walk_forward_metadata(daily_row),
        "target_position": float(daily_row.get("target_position", 0.0) or 0.0),
        "cost_bps": float(cost_bps),
        "entry_date": None,
        "exit_date": None,
        "source_daily_strategy_return": float(daily_row.get("strategy_return", 0.0) or 0.0),
        "drawdown_threshold": np.nan,
        "risk_score_threshold": np.nan,
        "trigger_mode": None,
        "risk_sleeve_execution_delay_bars": int(execution_delay_bars),
        "risk_sleeve_execution_price": execution_price,
    }
    rows = _zero_5min_rows(common, variants=variants)
    for row in rows:
        row["skipped_reason"] = reason
    return rows


def _no_trigger_variant_rows(
    common: dict[str, Any],
    *,
    entry_open: float,
    exit_open: float,
    entry_date: Any,
    exit_date: Any,
    baseline_return: float,
    entry_day_strategy_return: float,
    entry_day_close_time: Any,
    entry_day_close_price: float,
    reason: str | None,
    variants: tuple[str, ...] = RISK_SLEEVE_VARIANTS,
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        rows.append(
            {
                **common,
                "variant": variant,
                "entry_time": pd.Timestamp(entry_date).date().isoformat() if pd.notna(entry_date) else None,
                "exit_time": pd.Timestamp(exit_date).date().isoformat() if pd.notna(exit_date) else None,
                "entry_price": entry_open,
                "exit_price": exit_open,
                "trigger_time": None,
                "trigger_price": np.nan,
                "trigger_drawdown_from_entry": np.nan,
                "trigger_risk_score": np.nan,
                **_empty_context_trigger_row(),
                "risk_execution_time": None,
                "risk_execution_price": np.nan,
                "risk_triggered": False,
                "gross_return": baseline_return,
                "strategy_return": baseline_return,
                "entry_day_close_time": _timestamp_or_none(entry_day_close_time),
                "entry_day_close_price": entry_day_close_price,
                "entry_day_strategy_return": entry_day_strategy_return,
                "skipped_reason": reason,
            }
        )
    return rows


def _validation_policy_grid(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    bars_5min: pd.DataFrame,
    risk_score_col: str,
    gate_col: str,
) -> list[dict[str, Any]]:
    validation = daily.loc[daily["segment"].eq("validation")].copy()
    if validation.empty:
        validation = daily.copy()
    drawdowns: list[float] = []
    scores: list[float] = []
    context_values: dict[str, list[float]] = {column: [] for column in CONTEXT_THRESHOLD_COLUMNS}
    prices = _normalize_daily_prices(daily_prices)
    price_index = pd.Index(prices["datetime"])
    bars_by_date = {
        date: group.sort_values("datetime").reset_index(drop=True)
        for date, group in bars_5min.groupby("date", sort=True)
    }
    for row in validation.to_dict("records"):
        if abs(float(row.get("target_position", 0.0) or 0.0)) <= 1e-12:
            continue
        decision = pd.Timestamp(row["datetime"]).normalize()
        if decision not in set(price_index):
            continue
        location = int(price_index.get_loc(decision))
        if location + 1 >= len(prices):
            continue
        entry_date = prices.iloc[location + 1]["datetime"]
        entry_open = float(prices.iloc[location + 1]["open"])
        entry_bars = bars_by_date.get(pd.Timestamp(entry_date).date().isoformat(), pd.DataFrame())
        if entry_bars.empty:
            continue
        pnl = float(row.get("target_position", 0.0) or 0.0) * (
            pd.to_numeric(entry_bars["close"], errors="coerce") / entry_open - 1.0
        )
        if pnl.notna().any():
            drawdowns.append(float(pnl.min()))
        valid_scores = pd.to_numeric(entry_bars.loc[entry_bars[gate_col].fillna(False).astype(bool), risk_score_col], errors="coerce").dropna()
        scores.extend(float(value) for value in valid_scores.tolist())
        for column in CONTEXT_THRESHOLD_COLUMNS:
            if column in entry_bars.columns:
                context_values[column].extend(float(value) for value in pd.to_numeric(entry_bars[column], errors="coerce").dropna().tolist())
    drawdown_series = pd.Series(drawdowns, dtype=float).dropna()
    score_series = pd.Series(scores, dtype=float).dropna()
    drawdown_thresholds = sorted(
        {
            float(min(value, -0.005))
            for value in drawdown_series.quantile([0.03, 0.10, 0.30]).dropna().tolist()
        }
    )
    if not drawdown_thresholds:
        drawdown_thresholds = [-0.015, -0.020, -0.030]
    risk_thresholds: list[float | None] = [None]
    if not score_series.empty:
        risk_thresholds.extend(
            sorted({float(value) for value in score_series.quantile([0.10, 0.35]).dropna().tolist()})
        )
    context_candidates = _validation_context_threshold_candidates(context_values)
    policies: list[dict[str, Any]] = []
    score_thresholds = [threshold for threshold in risk_thresholds if threshold is not None]
    for variant in RISK_SLEEVE_VARIANTS:
        for drawdown_threshold in drawdown_thresholds:
            policy_id = f"{variant}_drawdown_only_dd_{drawdown_threshold:.5f}_risk_none"
            policies.append(
                {
                    "policy_id": policy_id,
                    "variant": variant,
                    "trigger_mode": "drawdown_only",
                    "drawdown_threshold": float(drawdown_threshold),
                    "drawdown_threshold_source": "validation_quantile",
                    "risk_score_threshold": None,
                    "risk_score_threshold_source": "disabled",
                    "context_trigger_source": "disabled",
                    **_context_threshold_row({}),
                    "selection_segment": "validation",
                    "report_segment": "test",
                    "selection_uses_test_segment": False,
                }
            )
        for risk_threshold in score_thresholds:
            policy_id = f"{variant}_risk_score_only_dd_disabled_risk_{risk_threshold:.5f}"
            policies.append(
                {
                    "policy_id": policy_id,
                    "variant": variant,
                    "trigger_mode": "risk_score_only",
                    "drawdown_threshold": np.nan,
                    "drawdown_threshold_source": "disabled_for_risk_score_only",
                    "risk_score_threshold": float(risk_threshold),
                    "risk_score_threshold_source": "validation_quantile",
                    "context_trigger_source": "disabled",
                    **_context_threshold_row({}),
                    "selection_segment": "validation",
                    "report_segment": "test",
                    "selection_uses_test_segment": False,
                }
            )
        for trigger_mode in ("risk_score_or_drawdown", "risk_score_and_drawdown"):
            for drawdown_threshold in drawdown_thresholds:
                for risk_threshold in score_thresholds:
                    policy_id = f"{variant}_{trigger_mode}_dd_{drawdown_threshold:.5f}_risk_{risk_threshold:.5f}"
                    policies.append(
                        {
                            "policy_id": policy_id,
                            "variant": variant,
                            "trigger_mode": trigger_mode,
                            "drawdown_threshold": float(drawdown_threshold),
                            "drawdown_threshold_source": "validation_quantile",
                            "risk_score_threshold": float(risk_threshold),
                            "risk_score_threshold_source": "validation_quantile",
                            "context_trigger_source": "disabled",
                            **_context_threshold_row({}),
                            "selection_segment": "validation",
                            "report_segment": "test",
                            "selection_uses_test_segment": False,
                        }
                    )
        for drawdown_threshold in drawdown_thresholds:
            for context_candidate in context_candidates:
                policy_id = (
                    f"{variant}_context_stress_or_drawdown_dd_{drawdown_threshold:.5f}_"
                    f"ctx_{context_candidate['context_trigger_source']}"
                )
                policies.append(
                    {
                        "policy_id": policy_id,
                        "variant": variant,
                        "trigger_mode": "context_stress_or_drawdown",
                        "drawdown_threshold": float(drawdown_threshold),
                        "drawdown_threshold_source": "validation_quantile",
                        "risk_score_threshold": None,
                        "risk_score_threshold_source": "disabled",
                        "selection_segment": "validation",
                        "report_segment": "test",
                        "selection_uses_test_segment": False,
                        **context_candidate,
                    }
                )
    return policies


def _validation_context_threshold_candidates(context_values: dict[str, list[float]]) -> list[dict[str, Any]]:
    series = {column: pd.Series(values, dtype=float).dropna() for column, values in context_values.items()}

    def lower_threshold(column: str, quantile: float, fallback: float, cap: float) -> float:
        values = series.get(column, pd.Series(dtype=float))
        if values.empty:
            return float(fallback)
        return float(min(values.quantile(quantile), cap))

    def upper_threshold(column: str, quantile: float, fallback: float, floor: float) -> float:
        values = series.get(column, pd.Series(dtype=float))
        if values.empty:
            return float(fallback)
        return float(max(values.quantile(quantile), floor))

    vwap = lower_threshold("soxl_vwap_dist", 0.25, -0.005, -0.001)
    soxx_weak = lower_threshold("soxx_cum_ret_from_open", 0.25, -0.004, -0.001)
    dislocation = lower_threshold("soxl_minus_3x_soxx_ret", 0.25, -0.002, -0.0005)
    rv_expansion = upper_threshold("soxl_rv_6", 0.75, 0.006, 0.001)
    gap_reclaim_failure = lower_threshold("open_gap_reclaim_quality", 0.25, -0.20, -0.05)
    weak_trend = lower_threshold("soxl_trend_quality_60m", 0.25, 0.20, 0.30)
    absorption = upper_threshold("soxl_volume_price_absorption", 0.75, 1.0, 0.5)
    leveraged_absorption = upper_threshold("leveraged_replication_absorption", 0.75, 0.75, 0.25)
    candidates = [
        {
            "context_trigger_source": "vwap_soxx_weak",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "soxx_cum_ret_from_open": soxx_weak,
                }
            ),
        },
        {
            "context_trigger_source": "vwap_vol_expansion",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "soxl_rv_6": rv_expansion,
                }
            ),
        },
        {
            "context_trigger_source": "vwap_leverage_dislocation",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "soxl_minus_3x_soxx_ret": dislocation,
                }
            ),
        },
        {
            "context_trigger_source": "vwap_multi_stress",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "soxx_cum_ret_from_open": soxx_weak,
                    "soxl_rv_6": rv_expansion,
                    "soxl_minus_3x_soxx_ret": dislocation,
                }
            ),
        },
        {
            "context_trigger_source": "vwap_gap_reclaim_failure",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "open_gap_reclaim_quality": gap_reclaim_failure,
                }
            ),
        },
        {
            "context_trigger_source": "vwap_weak_trend",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "soxl_trend_quality_60m": weak_trend,
                }
            ),
        },
        {
            "context_trigger_source": "vwap_volume_absorption",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "soxl_volume_price_absorption": absorption,
                    "leveraged_replication_absorption": leveraged_absorption,
                }
            ),
        },
        {
            "context_trigger_source": "vwap_local_microstructure_stress",
            **_context_threshold_row(
                {
                    "soxl_vwap_dist": vwap,
                    "open_gap_reclaim_quality": gap_reclaim_failure,
                    "soxl_trend_quality_60m": weak_trend,
                    "soxl_volume_price_absorption": absorption,
                    "leveraged_replication_absorption": leveraged_absorption,
                }
            ),
        },
    ]
    if not series.get("qqq_smh_soxx_lead_lag_3", pd.Series(dtype=float)).empty:
        lead_lag_weak = lower_threshold("qqq_smh_soxx_lead_lag_3", 0.25, -0.004, -0.0005)
        candidates.append(
            {
                "context_trigger_source": "external_lead_lag_weak",
                **_context_threshold_row(
                    {
                        "soxl_vwap_dist": vwap,
                        "qqq_smh_soxx_lead_lag_3": lead_lag_weak,
                    }
                ),
            }
        )
    if not series.get("semis_breadth_mom3_pos", pd.Series(dtype=float)).empty:
        semis_breadth_weak = lower_threshold("semis_breadth_mom3_pos", 0.25, 0.40, 0.40)
        candidates.append(
            {
                "context_trigger_source": "external_semis_breadth_weak",
                **_context_threshold_row(
                    {
                        "soxl_vwap_dist": vwap,
                        "semis_breadth_mom3_pos": semis_breadth_weak,
                    }
                ),
            }
        )
    if (
        not series.get("qqq_smh_soxx_lead_lag_3", pd.Series(dtype=float)).empty
        and not series.get("semis_breadth_mom3_pos", pd.Series(dtype=float)).empty
        and not series.get("semis_vwap_breadth", pd.Series(dtype=float)).empty
    ):
        lead_lag_weak = lower_threshold("qqq_smh_soxx_lead_lag_3", 0.25, -0.004, -0.0005)
        semis_breadth_weak = lower_threshold("semis_breadth_mom3_pos", 0.25, 0.40, 0.40)
        semis_vwap_weak = lower_threshold("semis_vwap_breadth", 0.25, 0.40, 0.40)
        candidates.append(
            {
                "context_trigger_source": "external_multi_breadth_weak",
                **_context_threshold_row(
                    {
                        "soxl_vwap_dist": vwap,
                        "qqq_smh_soxx_lead_lag_3": lead_lag_weak,
                        "semis_breadth_mom3_pos": semis_breadth_weak,
                        "semis_vwap_breadth": semis_vwap_weak,
                    }
                ),
            }
        )
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique[str(candidate["context_trigger_source"])] = candidate
    return list(unique.values())


def _policy_metric_rows(policy: dict[str, Any], *, metrics: pd.DataFrame, risk: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cost_bps in sorted(float(value) for value in metrics["cost_bps"].dropna().unique()):
        for variant in [str(policy["variant"])]:
            val = _single_metric(metrics, cost_bps=cost_bps, segment="validation", variant=variant)
            test = _single_metric(metrics, cost_bps=cost_bps, segment="test", variant=variant)
            val_risk = _single_metric(risk, cost_bps=cost_bps, segment="validation", variant=variant)
            test_risk = _single_metric(risk, cost_bps=cost_bps, segment="test", variant=variant)
            if not val or not test:
                continue
            row = {
                **policy,
                "variant": variant,
                "cost_bps": float(cost_bps),
                "validation_periods": val.get("periods", 0),
                "validation_return_pct": val.get("total_return_pct", 0.0),
                "validation_return_retention_pct": val.get("return_retention_pct", np.nan),
                "validation_max_drawdown_pct": val.get("max_drawdown_pct", 0.0),
                "validation_drawdown_improvement_pct": val.get("drawdown_improvement_pct", 0.0),
                "validation_daily_sharpe": val.get("daily_sharpe", 0.0),
                "validation_skipped_periods": val.get("skipped_periods", 0),
                "test_periods": test.get("periods", 0),
                "test_return_pct": test.get("total_return_pct", 0.0),
                "test_return_retention_pct": test.get("return_retention_pct", np.nan),
                "test_max_drawdown_pct": test.get("max_drawdown_pct", 0.0),
                "test_drawdown_improvement_pct": test.get("drawdown_improvement_pct", 0.0),
                "test_daily_sharpe": test.get("daily_sharpe", 0.0),
                "test_skipped_periods": test.get("skipped_periods", 0),
                "validation_open_buy_dump_period_reduction": val_risk.get("open_buy_dump_period_reduction", 0.0),
                "validation_crash_period_reduction": val_risk.get("crash_period_reduction", 0.0),
                "validation_worst_entry_day_return_improvement_pct": val_risk.get("worst_entry_day_return_improvement_pct", 0.0),
                "validation_worst_strategy_return_improvement_pct": val_risk.get("worst_strategy_return_improvement_pct", 0.0),
                "test_open_buy_dump_period_reduction": test_risk.get("open_buy_dump_period_reduction", 0.0),
                "test_crash_period_reduction": test_risk.get("crash_period_reduction", 0.0),
                "test_worst_entry_day_return_improvement_pct": test_risk.get("worst_entry_day_return_improvement_pct", 0.0),
                "test_worst_strategy_return_improvement_pct": test_risk.get("worst_strategy_return_improvement_pct", 0.0),
            }
            row["validation_return_retention_score"] = float(min(float(row["validation_return_retention_pct"]), 95.0))
            row["test_return_retention_score"] = float(min(float(row["test_return_retention_pct"]), 95.0))
            row["validation_risk_reduction_pass"] = bool(
                float(row["validation_open_buy_dump_period_reduction"]) > 0.0
                or float(row["validation_crash_period_reduction"]) > 0.0
                or float(row["validation_worst_entry_day_return_improvement_pct"]) > 0.0
            )
            row["test_risk_reduction_pass"] = bool(
                float(row["test_open_buy_dump_period_reduction"]) > 0.0
                or float(row["test_crash_period_reduction"]) > 0.0
                or float(row["test_worst_entry_day_return_improvement_pct"]) > 0.0
            )
            row["passes_validation_target"] = bool(
                85.0 <= float(row["validation_return_retention_pct"]) <= 95.0
                and float(row["validation_max_drawdown_pct"]) > -18.0
                and int(row["validation_skipped_periods"]) <= int(row["validation_periods"]) * 0.15
                and row["validation_risk_reduction_pass"]
            )
            row["passes_test_target"] = bool(
                85.0 <= float(row["test_return_retention_pct"]) <= 95.0
                and float(row["test_max_drawdown_pct"]) > -18.0
                and int(row["test_skipped_periods"]) <= int(row["test_periods"]) * 0.15
                and row["test_risk_reduction_pass"]
            )
            row["passes_validation_and_test_target"] = bool(row["passes_validation_target"] and row["passes_test_target"])
            row["selection_score"] = (
                row["validation_return_retention_score"]
                + 2.0 * float(row["validation_drawdown_improvement_pct"])
                + 0.25 * float(row["validation_daily_sharpe"])
                + 2.0 * float(row["validation_open_buy_dump_period_reduction"])
                + 2.0 * float(row["validation_crash_period_reduction"])
                - 0.10 * float(row["validation_skipped_periods"])
            )
            rows.append(row)
    return rows


def _single_metric(frame: pd.DataFrame, *, cost_bps: float, segment: str, variant: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    matched = frame.loc[
        pd.to_numeric(frame["cost_bps"], errors="coerce").eq(float(cost_bps))
        & frame["segment"].astype(str).eq(segment)
        & frame["variant"].astype(str).eq(variant)
    ]
    return matched.iloc[0].to_dict() if not matched.empty else {}


def _load_or_fit_lgb_xgb_fold_predictions(
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    *,
    as_of: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], dict[str, Any]]:
    config = QlibProviderConfig(
        provider_uri=spec.intraday_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="15min",
        kernels=1,
    )
    provider_meta = init_qlib_provider(config)
    fallback_summary_path = spec.local_fallback_card_dir / "summary.json"
    if fallback_summary_path.exists():
        fallback_summary = _read_json(fallback_summary_path)
        recorder_dir = Path(str(fallback_summary.get("recorder_local_dir", "")))
        prediction_path = recorder_dir / "artifacts" / "local_fallback_overlay" / "intraday_fold_predictions.csv"
        model_selection_path = spec.local_fallback_card_dir / "model_selection.csv"
        feature_importance_path = spec.local_fallback_card_dir / "feature_importance.csv"
        analysis_path = spec.local_fallback_card_dir / "analysis_summary.json"
        if prediction_path.exists() and model_selection_path.exists() and analysis_path.exists():
            analysis = _read_json(analysis_path)
            feature_set = analysis.get("feature_set", {})
            return (
                provider_meta,
                pd.read_csv(prediction_path),
                pd.read_csv(model_selection_path),
                pd.read_csv(feature_importance_path) if feature_importance_path.exists() else pd.DataFrame(),
                [str(value) for value in feature_set.get("features", [])],
                analysis.get("label_policy", {}),
                {
                    "card": spec.source_local_fallback_card,
                    "recorder_id": fallback_summary.get("recorder_id"),
                    "artifact": prediction_path.relative_to(recorder_dir).as_posix(),
                    "mode": "loaded_from_qlib_recorder",
                },
            )
    provider_meta, intraday_frame, feature_cols, label_audit = _load_intraday_label_frame(
        spec,  # type: ignore[arg-type]
        as_of=as_of,
    )
    fold_predictions, model_selection, feature_importance = _fit_local_intraday_walk_forward(
        intraday_frame,
        feature_cols=feature_cols,
        spec=spec,  # type: ignore[arg-type]
        as_of=as_of,
    )
    return (
        provider_meta,
        fold_predictions,
        model_selection,
        feature_importance,
        feature_cols,
        label_audit,
        {
            "card": spec.source_intraday_card,
            "mode": "refit_from_15min_qlib_provider",
        },
    )


def _load_5min_bar_frame(
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    *,
    as_of: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config = QlibProviderConfig(
        provider_uri=spec.intraday_5min_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="5min",
        kernels=1,
    )
    provider_meta = init_qlib_provider(config)
    raw = load_qlib_feature_frame(
        config,
        instruments=["SOXL", "SOXX"],
        fields=QLIB_FIELDS,
        start=f"{REQUIRED_WALK_FORWARD_START} 09:35:00",
        end=f"{as_of} 16:00:00",
    )
    wide = _wide_feature_frame(raw).sort_values("datetime").reset_index(drop=True)
    wide["date"] = pd.to_datetime(wide["date"]).dt.date.astype(str)
    for prefix in ["soxl", "soxx"]:
        wide[f"{prefix}_ret_cc1"] = wide.groupby("date")[f"{prefix}_close"].pct_change()
        day_open = wide.groupby("date")[f"{prefix}_open"].transform("first")
        day_high = wide.groupby("date")[f"{prefix}_high"].cummax()
        day_low = wide.groupby("date")[f"{prefix}_low"].cummin()
        prior_close_by_day = wide.groupby("date")[f"{prefix}_close"].last().shift(1)
        wide[f"{prefix}_prior_day_close"] = wide["date"].map(prior_close_by_day)
        wide[f"{prefix}_open_gap_from_prior_close"] = day_open / (wide[f"{prefix}_prior_day_close"] + 1e-12) - 1.0
        wide[f"{prefix}_cum_ret_from_open"] = wide[f"{prefix}_close"] / day_open - 1.0
        wide[f"{prefix}_range_so_far"] = day_high / (day_low + 1e-12) - 1.0
        typical = (wide[f"{prefix}_high"] + wide[f"{prefix}_low"] + wide[f"{prefix}_close"]) / 3.0
        pv = typical * wide[f"{prefix}_volume"]
        wide[f"{prefix}_cum_vwap"] = pv.groupby(wide["date"]).cumsum() / (
            wide.groupby("date")[f"{prefix}_volume"].cumsum() + 1e-12
        )
        wide[f"{prefix}_vwap_dist"] = wide[f"{prefix}_close"] / (wide[f"{prefix}_cum_vwap"] + 1e-12) - 1.0
        wide[f"{prefix}_rv_6"] = (
            wide.groupby("date")[f"{prefix}_ret_cc1"].rolling(6, min_periods=3).std().reset_index(level=0, drop=True)
        )
        wide[f"{prefix}_trend_quality_30m"] = (
            wide[f"{prefix}_cum_ret_from_open"].abs()
            / (wide[f"{prefix}_range_so_far"] + 1e-12)
        ).clip(0.0, 5.0)
        wide[f"{prefix}_trend_quality_60m"] = (
            wide.groupby("date")[f"{prefix}_cum_ret_from_open"]
            .rolling(12, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
            .abs()
            / (
                wide.groupby("date")[f"{prefix}_range_so_far"]
                .rolling(12, min_periods=3)
                .max()
                .reset_index(level=0, drop=True)
                + 1e-12
            )
        ).clip(0.0, 5.0)
        rolling_volume = wide.groupby("date")[f"{prefix}_volume"].rolling(6, min_periods=3).mean().reset_index(level=0, drop=True)
        rolling_abs_ret = (
            wide.groupby("date")[f"{prefix}_ret_cc1"].rolling(6, min_periods=3).apply(lambda values: np.nanmean(np.abs(values)), raw=True).reset_index(level=0, drop=True)
        )
        volume_pressure = wide[f"{prefix}_volume"] / (rolling_volume + 1e-12) - 1.0
        ret_response = wide[f"{prefix}_ret_cc1"].abs() / (rolling_abs_ret + 1e-12)
        wide[f"{prefix}_volume_price_absorption"] = (volume_pressure - 0.25 * ret_response).replace([np.inf, -np.inf], np.nan)
    wide["soxl_minus_3x_soxx_ret"] = wide["soxl_ret_cc1"] - 3.0 * wide["soxx_ret_cc1"]
    wide["leveraged_replication_absorption"] = wide["soxl_volume_price_absorption"] - wide["soxx_volume_price_absorption"]
    wide["open_gap_reclaim_quality"] = np.where(
        wide["soxl_open_gap_from_prior_close"].abs() > 1e-12,
        wide["soxl_cum_ret_from_open"] / wide["soxl_open_gap_from_prior_close"].abs(),
        0.0,
    )
    return provider_meta, wide.rename(
        columns={
            "soxl_open": "open",
            "soxl_high": "high",
            "soxl_low": "low",
            "soxl_close": "close",
            "soxl_volume": "volume",
        }
    )


def _load_external_15min_context_frame(
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    *,
    as_of: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    summary = _read_optional_json(spec.external_context_card_dir / "summary.json")
    if not spec.external_context_provider_uri.exists():
        return (
            {},
            pd.DataFrame(),
            {
                "status": "blocked_missing_external_context_provider",
                "provider_uri": spec.external_context_provider_uri.resolve().as_posix(),
                "summary": summary,
            },
        )
    config = QlibProviderConfig(
        provider_uri=spec.external_context_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="15min",
        kernels=1,
    )
    provider_meta = init_qlib_provider(config)
    raw = load_qlib_feature_frame(
        config,
        instruments=list(REQUESTED_EXTERNAL_CONTEXT_SYMBOLS),
        fields=QLIB_FIELDS,
        start=f"{REQUIRED_WALK_FORWARD_START} 09:45:00",
        end=f"{as_of} 16:00:00",
    )
    external_context, feature_cols, feature_audit = build_external_context_feature_frame(raw)
    coverage_gap = summary.get("coverage_gap_analysis", {}) if summary else {}
    return (
        provider_meta,
        external_context,
        {
            "status": "ok_partial" if not bool(coverage_gap.get("full_walk_forward_history_available", False)) else "ok",
            "provider_uri": spec.external_context_provider_uri.resolve().as_posix(),
            "provider_meta": provider_meta,
            "feature_cols": feature_cols,
            "feature_audit": feature_audit,
            "coverage_gap_analysis": coverage_gap,
            "source_card": spec.external_context_card_dir.relative_to(spec.root).as_posix(),
            "summary_recorder_id": summary.get("recorder_id") if summary else None,
        },
    )


def _attach_external_context_to_5min_bars(
    bars_5min: pd.DataFrame,
    external_context_15m: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bars = bars_5min.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars["date"] = pd.to_datetime(bars.get("date", bars["datetime"]).astype(str)).dt.date.astype(str)
    for column in ["external_context_available", *EXTERNAL_CONTEXT_THRESHOLD_COLUMNS]:
        if column in bars.columns:
            bars = bars.drop(columns=[column])
    if external_context_15m.empty:
        bars["external_context_available"] = False
        for column in EXTERNAL_CONTEXT_THRESHOLD_COLUMNS:
            bars[column] = np.nan
        return bars.sort_values(["date", "datetime"]).reset_index(drop=True), {
            "status": "blocked_no_external_context_rows",
            "external_15m_rows": 0,
            "external_context_available_5min_rows": 0,
            "attached_feature_cols": list(EXTERNAL_CONTEXT_THRESHOLD_COLUMNS),
            "lookback_only": True,
        }
    external = external_context_15m.copy()
    external["datetime"] = pd.to_datetime(external["datetime"])
    external["date"] = external["datetime"].dt.date.astype(str)
    if "external_context_available" not in external.columns:
        external["external_context_available"] = False
    attach_cols = ["datetime", "date", "external_context_available"]
    attach_cols.extend([column for column in EXTERNAL_CONTEXT_THRESHOLD_COLUMNS if column in external.columns])
    external = external[attach_cols].sort_values(["date", "datetime"]).reset_index(drop=True)
    merged = pd.merge_asof(
        bars.sort_values(["date", "datetime"]),
        external,
        on="datetime",
        by="date",
        direction="backward",
        suffixes=("", "_external"),
    )
    merged["external_context_available"] = merged["external_context_available"].map(lambda value: bool(value) if pd.notna(value) else False)
    for column in EXTERNAL_CONTEXT_THRESHOLD_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged.loc[~merged["external_context_available"], column] = np.nan
    available = int(merged["external_context_available"].sum())
    return merged.sort_values(["date", "datetime"]).reset_index(drop=True), {
        "status": "ok_partial" if available else "blocked_no_5min_external_context_overlap",
        "external_15m_rows": int(len(external)),
        "external_context_available_5min_rows": available,
        "external_15m_start": _timestamp_or_none(external["datetime"].min()) if not external.empty else None,
        "external_15m_end": _timestamp_or_none(external["datetime"].max()) if not external.empty else None,
        "attached_feature_cols": [column for column in EXTERNAL_CONTEXT_THRESHOLD_COLUMNS if column in merged.columns],
        "lookback_only": True,
    }


def _evaluate_annual_5min_risk_sleeve(
    *,
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    bars_5min: pd.DataFrame,
    fold_predictions: pd.DataFrame,
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    as_of: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_rows: list[dict[str, Any]] = []
    selected_cost_frames: list[pd.DataFrame] = []
    selected_trade_frames: list[pd.DataFrame] = []
    all_policy_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    for test_year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS:
        fold = f"wf_{int(test_year)}"
        train_start = REQUIRED_WALK_FORWARD_START
        train_end = f"{int(test_year) - 2}-12-31"
        valid_start = f"{int(test_year) - 1}-01-01"
        valid_end = f"{int(test_year) - 1}-12-31"
        test_start = f"{int(test_year)}-01-01"
        test_end = as_of if int(test_year) == int(as_of[:4]) else f"{int(test_year)}-12-31"
        fold_daily = daily_positions.loc[daily_positions["walk_forward_fold"].astype(str).eq(fold)].copy()
        fold_scores = _risk_scores_for_fold(fold_predictions, fold=fold)
        if fold_daily.empty or fold_scores.empty:
            annual_rows.append(
                _infeasible_fold_row(
                    fold=fold,
                    test_year=int(test_year),
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    reason="missing_daily_or_intraday_fold_data",
                )
            )
            continue
        candidates = evaluate_5min_risk_sleeve_candidates(
            fold_daily,
            daily_prices=daily_prices,
            bars_5min=bars_5min,
            intraday_scores=fold_scores,
            cost_bps_values=tuple(float(value) for value in spec.required_cost_bps),
            execution_delay_bars=RISK_SLEEVE_EXECUTION_DELAY_BARS,
            execution_price=RISK_SLEEVE_EXECUTION_PRICE,
        )
        if not candidates.empty:
            candidates["fold"] = fold
            candidates["test_year"] = int(test_year)
            all_policy_frames.append(candidates)
        selected = select_validation_robust_5min_risk_policy(
            candidates,
            required_cost_bps=tuple(float(value) for value in spec.required_cost_bps),
            selection_cost_bps=float(spec.selection_cost_bps),
        )
        selected_rows = candidates.loc[candidates["policy_id"].eq(selected.get("policy_id"))].copy() if not candidates.empty else pd.DataFrame()
        if not selected_rows.empty:
            selected_rows["fold"] = fold
            selected_rows["test_year"] = int(test_year)
            selected_rows["selection_uses_test_segment"] = False
            selected_cost_frames.append(selected_rows)
        selected_trades = _simulate_selected_trades(
            selected,
            fold_daily=fold_daily,
            daily_prices=daily_prices,
            bars_5min=bars_5min,
            fold_scores=fold_scores,
            required_cost_bps=tuple(float(value) for value in spec.required_cost_bps),
        )
        if not selected_trades.empty:
            selected_trades["walk_forward_fold"] = fold
            selected_trades["test_year"] = int(test_year)
            selected_trade_frames.append(selected_trades)
            fold_risk = summarize_execution_risk_diagnostics(selected_trades)
            if not fold_risk.empty:
                fold_risk["fold"] = fold
                fold_risk["test_year"] = int(test_year)
                risk_frames.append(fold_risk)
        else:
            fold_risk = pd.DataFrame()
        annual_rows.append(
            _annual_fold_row(
                selected,
                selected_policy_rows=selected_rows,
                risk_analysis=fold_risk,
                fold=fold,
                test_year=int(test_year),
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
                intraday_prediction_rows=int(fold_scores["risk_score"].notna().sum()),
                policy_candidate_rows=int(len(candidates)),
            )
        )
    return (
        pd.DataFrame(annual_rows),
        pd.concat(selected_cost_frames, ignore_index=True) if selected_cost_frames else pd.DataFrame(),
        pd.concat(selected_trade_frames, ignore_index=True) if selected_trade_frames else pd.DataFrame(),
        pd.concat(all_policy_frames, ignore_index=True) if all_policy_frames else pd.DataFrame(),
        pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame(),
    )


def _risk_scores_for_fold(fold_predictions: pd.DataFrame, *, fold: str) -> pd.DataFrame:
    frame = fold_predictions.loc[fold_predictions["walk_forward_fold"].astype(str).eq(fold)].copy()
    if frame.empty:
        return frame
    selected_model = str(frame["selected_intraday_model"].dropna().iloc[0])
    prob_col = f"prob_{selected_model}"
    frame["risk_score"] = pd.to_numeric(frame[prob_col], errors="coerce")
    if PREDICTION_AVAILABLE_GATE_COL not in frame.columns:
        frame[PREDICTION_AVAILABLE_GATE_COL] = frame["risk_score"].notna()
    return frame[["datetime", "date", "risk_score", PREDICTION_AVAILABLE_GATE_COL]].copy()


def _simulate_selected_trades(
    selected: dict[str, Any],
    *,
    fold_daily: pd.DataFrame,
    daily_prices: pd.DataFrame,
    bars_5min: pd.DataFrame,
    fold_scores: pd.DataFrame,
    required_cost_bps: tuple[float, ...],
) -> pd.DataFrame:
    if not selected.get("policy_id"):
        return pd.DataFrame()
    frames = []
    selected_drawdown_threshold = selected.get("drawdown_threshold")
    drawdown_threshold = (
        np.nan
        if selected_drawdown_threshold is None or pd.isna(selected_drawdown_threshold)
        else float(selected_drawdown_threshold)
    )
    for cost_bps in required_cost_bps:
        trades = simulate_5min_risk_sleeve(
            fold_daily,
            daily_prices=daily_prices,
            bars_5min=bars_5min,
            intraday_scores=fold_scores,
            drawdown_threshold=drawdown_threshold,
            risk_score_threshold=selected.get("risk_score_threshold"),
            trigger_mode=str(selected.get("trigger_mode")),
            cost_bps=float(cost_bps),
            execution_delay_bars=RISK_SLEEVE_EXECUTION_DELAY_BARS,
            execution_price=RISK_SLEEVE_EXECUTION_PRICE,
            context_thresholds=_context_thresholds_from_policy(selected),
            context_trigger_source=str(selected.get("context_trigger_source", "disabled")),
        ).assign(cost_bps=float(cost_bps))
        selected_variant = str(selected.get("variant"))
        frames.append(trades.loc[trades["variant"].isin(["baseline_next_open", selected_variant])].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _annual_fold_row(
    selected: dict[str, Any],
    *,
    selected_policy_rows: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    fold: str,
    test_year: int,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    intraday_prediction_rows: int,
    policy_candidate_rows: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold,
        "test_year": int(test_year),
        "status": "ok" if selected.get("policy_id") else "infeasible",
        "blocking_reason": "ok" if selected.get("policy_id") else "missing_policy",
        "required_cost_bps": ",".join(str(float(value)) for value in REQUIRED_COST_BPS),
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "selection_uses_test_segment": False,
        "selected_policy_id": selected.get("policy_id"),
        "selected_variant": selected.get("variant"),
        "trigger_mode": selected.get("trigger_mode"),
        "drawdown_threshold": selected.get("drawdown_threshold"),
        "risk_score_threshold": selected.get("risk_score_threshold"),
        "context_trigger_source": selected.get("context_trigger_source", "disabled"),
        **_context_threshold_row(_context_thresholds_from_policy(selected)),
        "validation_passes_target": bool(selected.get("validation_passes_target", False)),
        "test_passes_target": bool(selected.get("test_passes_target", False)),
        "intraday_prediction_rows": int(intraday_prediction_rows),
        "policy_candidate_rows": int(policy_candidate_rows),
    }
    if selected_policy_rows.empty:
        row["risk_reduction_passes_target"] = False
        row["test_passes_required_cost_target"] = False
        row["validation_passes_required_cost_target"] = False
        row["validation_and_test_passes_required_cost_target"] = False
        return row
    required = selected_policy_rows.loc[selected_policy_rows["cost_bps"].isin(list(REQUIRED_COST_BPS))].copy()
    row["required_cost_validation_target_count"] = int(required["passes_validation_target"].astype(bool).sum())
    row["required_cost_test_target_count"] = int(required["passes_test_target"].astype(bool).sum())
    row["required_cost_validation_and_test_target_count"] = int(
        required["passes_validation_and_test_target"].astype(bool).sum()
    )
    for _, cost_row in required.sort_values("cost_bps").iterrows():
        cost_label = f"{float(cost_row['cost_bps']):.0f}bp"
        for column in [
            "validation_return_pct",
            "validation_return_retention_pct",
            "validation_max_drawdown_pct",
            "validation_open_buy_dump_period_reduction",
            "validation_crash_period_reduction",
            "test_return_pct",
            "test_return_retention_pct",
            "test_max_drawdown_pct",
            "test_open_buy_dump_period_reduction",
            "test_crash_period_reduction",
            "passes_validation_target",
            "passes_test_target",
            "passes_validation_and_test_target",
        ]:
            row[f"{cost_label}_{column}"] = cost_row.get(column)
    if not risk_analysis.empty and selected.get("variant"):
        for _, risk_row in risk_analysis.loc[
            risk_analysis["variant"].eq(str(selected.get("variant")))
            & risk_analysis["cost_bps"].isin(list(REQUIRED_COST_BPS))
            & risk_analysis["segment"].eq("test")
        ].iterrows():
            cost_label = f"{float(risk_row['cost_bps']):.0f}bp"
            for column in [
                "open_buy_dump_period_reduction",
                "crash_period_reduction",
                "worst_entry_day_return_improvement_pct",
                "worst_strategy_return_improvement_pct",
            ]:
                row[f"{cost_label}_test_{column}"] = risk_row[column]
    row["risk_reduction_passes_target"] = bool(
        max(
            float(row.get("20bp_test_open_buy_dump_period_reduction") or 0.0),
            float(row.get("50bp_test_open_buy_dump_period_reduction") or 0.0),
            float(row.get("20bp_test_crash_period_reduction") or 0.0),
            float(row.get("50bp_test_crash_period_reduction") or 0.0),
        )
        > 0.0
    )
    row["test_passes_required_cost_target"] = bool(row.get("required_cost_test_target_count", 0) >= len(REQUIRED_COST_BPS))
    row["validation_passes_required_cost_target"] = bool(
        row.get("required_cost_validation_target_count", 0) >= len(REQUIRED_COST_BPS)
    )
    row["validation_and_test_passes_required_cost_target"] = bool(
        row.get("required_cost_validation_and_test_target_count", 0) >= len(REQUIRED_COST_BPS)
    )
    return row


def _infeasible_fold_row(
    *,
    fold: str,
    test_year: int,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "fold": fold,
        "test_year": int(test_year),
        "status": "infeasible",
        "blocking_reason": reason,
        "required_cost_bps": ",".join(str(float(value)) for value in REQUIRED_COST_BPS),
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "selection_uses_test_segment": False,
        "validation_passes_target": False,
        "test_passes_target": False,
        "risk_reduction_passes_target": False,
    }


def _selected_model_payload(annual_walk_forward: pd.DataFrame) -> dict[str, Any]:
    if annual_walk_forward.empty:
        return {
            "policy_id": None,
            "selection_uses_test_segment": False,
            "validation_passes_target": False,
            "passes_test_target": False,
        }
    latest = annual_walk_forward.sort_values("test_year").iloc[-1].to_dict()
    payload = _json_ready_dict(latest)
    payload["selection_segment"] = "validation"
    payload["report_segment"] = "test"
    payload["selection_uses_test_segment"] = False
    payload["selected_by"] = "latest_fold_5min_risk_sleeve_with_annual_walk_forward_summary"
    payload["validation_passes_target"] = bool(payload.get("validation_passes_required_cost_target", False))
    payload["passes_test_target"] = bool(payload.get("test_passes_required_cost_target", False))
    payload["variant"] = payload.get("selected_variant")
    payload["test_return_retention_pct"] = payload.get("20bp_test_return_retention_pct")
    payload["test_max_drawdown_pct"] = payload.get("20bp_test_max_drawdown_pct")
    return payload


def _walk_forward_summary(annual_walk_forward: pd.DataFrame) -> dict[str, Any]:
    required_years = [int(year) for year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS]
    feasible_years = (
        sorted(int(value) for value in annual_walk_forward.loc[annual_walk_forward["status"].eq("ok"), "test_year"].tolist())
        if not annual_walk_forward.empty
        else []
    )
    target_years = (
        sorted(
            int(value)
            for value in annual_walk_forward.loc[
                annual_walk_forward.get(
                    "validation_and_test_passes_required_cost_target",
                    pd.Series(False, index=annual_walk_forward.index),
                ).fillna(False).astype(bool),
                "test_year",
            ].tolist()
        )
        if not annual_walk_forward.empty
        else []
    )
    risk_years = (
        sorted(
            int(value)
            for value in annual_walk_forward.loc[
                annual_walk_forward.get("risk_reduction_passes_target", pd.Series(False, index=annual_walk_forward.index))
                .fillna(False)
                .astype(bool),
                "test_year",
            ].tolist()
        )
        if not annual_walk_forward.empty
        else []
    )
    selection_uses_test = bool(
        annual_walk_forward.get("selection_uses_test_segment", pd.Series(False, index=annual_walk_forward.index))
        .fillna(False)
        .astype(bool)
        .any()
    )
    blocking_reasons = (
        sorted(
            {
                str(value)
                for value in annual_walk_forward.get("blocking_reason", pd.Series(dtype=object)).dropna().tolist()
                if str(value) != "ok"
            }
        )
        if not annual_walk_forward.empty
        else ["missing_walk_forward"]
    )
    if not set(required_years).issubset(target_years):
        blocking_reasons.append("annual_5min_risk_sleeve_target_gate_failed")
    if not set(required_years).issubset(risk_years):
        blocking_reasons.append("annual_5min_risk_reduction_gate_failed")
    return {
        "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
        "required_start": REQUIRED_WALK_FORWARD_START,
        "required_test_years": required_years,
        "feasible_test_years": feasible_years,
        "target_pass_test_years": target_years,
        "risk_reduction_test_years": risk_years,
        "missing_required_test_years": [year for year in required_years if year not in set(feasible_years)],
        "rows": int(len(annual_walk_forward)),
        "selection_uses_test_segment": selection_uses_test,
        "required_policy_satisfied": bool(
            not selection_uses_test
            and set(required_years).issubset(feasible_years)
            and set(required_years).issubset(target_years)
            and set(required_years).issubset(risk_years)
        ),
        "blocking_reasons": sorted(set(blocking_reasons)),
    }


def _analysis_summary(
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
    provider_meta_15m: dict[str, Any],
    provider_meta_5m: dict[str, Any],
    external_provider_meta: dict[str, Any],
    external_context_audit: dict[str, Any],
    daily_positions: pd.DataFrame,
    bars_5min: pd.DataFrame,
    feature_cols: list[str],
    label_audit: dict[str, Any],
    model_selection: pd.DataFrame,
    annual_walk_forward: pd.DataFrame,
    selected_policy_cost_sensitivity: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    selected_model: dict[str, Any],
    walk_forward_summary: dict[str, Any],
    context_candidate_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "daily_open2open_model_plus_local_15m_lgb_xgb_scores_plus_5min_replay",
        "as_of": spec.as_of,
        "selection_uses_test_segment": False,
        "optimization_answer": [
            "do_not_replace_daily_model_with_intraday_entry_gate",
            "preserve_daily_next_open_entry_and_use_5min_only_as_risk_sleeve",
            "trigger_on_5min_close_then_execute_risk_action_on_next_5min_open",
            "add_validation_selected_5min_context_stress_candidates",
            "select_drawdown_and_risk_score_thresholds_from_validation_year_only",
            "search_risk_score_only_and_risk_score_confirmed_drawdown_modes",
            "search_reenter_close_intraday_risk_reduction_variants",
            "report_test_year_without_threshold_refit",
        ],
        "why_intraday_did_not_beat_daily": [
            "daily_model_captures_open_to_open_and_overnight_trend_alpha",
            "intraday_label_captures_1h_2h_execution_improvement_not_multi_day_trend",
            "skipping_or_delaying_entry_removes_high_return_trend_days",
            "5min_replay_is_better_suited_to_reduce_open_buy_dump_than_to_create_primary_alpha",
        ],
        "data_audit": {
            "daily_walk_forward_position_rows": int(len(daily_positions)),
            "daily_walk_forward_start": daily_positions["datetime"].min().date().isoformat(),
            "daily_walk_forward_end": daily_positions["datetime"].max().date().isoformat(),
            "intraday_15m_provider_uri": provider_meta_15m.get("provider_uri"),
            "intraday_15m_calendar_start": provider_meta_15m.get("calendar_start"),
            "intraday_15m_calendar_end": provider_meta_15m.get("calendar_end"),
            "intraday_5m_provider_uri": provider_meta_5m.get("provider_uri"),
            "intraday_5m_calendar_start": provider_meta_5m.get("calendar_start"),
            "intraday_5m_calendar_end": provider_meta_5m.get("calendar_end"),
            "intraday_5m_rows": int(len(bars_5min)),
            "external_context_15m_provider_uri": external_provider_meta.get("provider_uri")
            or external_context_audit.get("provider_uri"),
            "external_context_15m_calendar_start": external_provider_meta.get("calendar_start"),
            "external_context_15m_calendar_end": external_provider_meta.get("calendar_end"),
            "external_context_available_5m_rows": external_context_audit.get("merge_audit", {}).get(
                "external_context_available_5min_rows",
                0,
            ),
            "external_context_full_walk_forward_history_available": bool(
                external_context_audit.get("coverage_gap_analysis", {}).get("full_walk_forward_history_available", False)
            ),
            "external_context_missing_required_history_years": external_context_audit.get("coverage_gap_analysis", {}).get(
                "missing_required_history_years",
                [],
            ),
        },
        "external_context": external_context_audit,
        "execution_overlay": {
            "risk_sleeve_timing": RISK_SLEEVE_EXECUTION_TIMING,
            "risk_sleeve_execution_delay_bars": RISK_SLEEVE_EXECUTION_DELAY_BARS,
            "risk_sleeve_execution_price": f"next_5min_{RISK_SLEEVE_EXECUTION_PRICE}",
            "trigger_price_role": "signal_only_5min_close",
            "execution_price_role": "risk_reduce_or_exit_fill_price",
            "risk_score_semantics": "probability_of_positive_1h_2h_execution_improvement",
            "risk_score_trigger_direction": "low_probability_triggers_risk_action",
            "trigger_modes": list(TRIGGER_MODES),
            "reenter_close_variants": ["risk_exit_full_5m_reenter_close", "risk_reduce_quarter_5m_reenter_close"],
            "context_trigger_features": list(CONTEXT_THRESHOLD_COLUMNS),
        },
        "label_policy": label_audit,
        "feature_set": {
            "lgb_xgb_15m_feature_count": int(len(feature_cols)),
            "lgb_xgb_15m_features": feature_cols,
            "five_min_replay_context": [
                "SOXL close drawdown from entry open",
                "SOXL VWAP distance",
                "SOXX same-session confirmation",
                "SOXL minus 3x SOXX 5min dislocation",
                "5min realized volatility/trend quality replay",
                "QQQ/SMH/SOXX 15min lead-lag replay aligned backward to 5min bars",
                "NVDA/AVGO/AMD/TSM/ASML 15min breadth replay aligned backward to 5min bars",
            ],
            "five_min_context_trigger_features": list(CONTEXT_THRESHOLD_COLUMNS),
            "local_context_trigger_features": list(LOCAL_CONTEXT_THRESHOLD_COLUMNS),
            "external_context_trigger_features": list(EXTERNAL_CONTEXT_THRESHOLD_COLUMNS),
            "external_context_required": False,
            "external_context_history_gate": "diagnostic_only_until_full_walk_forward_history_available",
        },
        "source_daily_model": {
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing"),
        },
        "source_intraday_model": {
            "recorder_id": intraday_summary.get("recorder_id"),
            "label_policy": intraday_summary.get("label_policy"),
            "score_semantics": "probability_of_positive_1h_2h_execution_improvement",
            "risk_trigger_direction": "trigger_when_execution_improvement_probability_is_less_or_equal_validation_quantile",
            "models": MODEL_NAMES,
        },
        "model_selection_rows": int(len(model_selection)),
        "annual_walk_forward_rows": int(len(annual_walk_forward)),
        "selected_policy_cost_rows": int(len(selected_policy_cost_sensitivity)),
        "risk_analysis_rows": int(len(risk_analysis)),
        "risk_sleeve_context_candidate_diagnostics": context_candidate_diagnostics,
        "selected_model": selected_model,
        "walk_forward": walk_forward_summary,
    }


def _risk_sleeve_context_candidate_diagnostics(
    *,
    all_policy_selection: pd.DataFrame,
    annual_walk_forward: pd.DataFrame,
) -> dict[str, Any]:
    if all_policy_selection.empty:
        return {
            "context_policy_candidate_rows": 0,
            "selected_context_trigger_years": [],
            "trigger_mode_year_summary": [],
            "best_context_validation_rows": [],
        }
    frame = all_policy_selection.copy()
    context = frame.loc[frame["trigger_mode"].astype(str).eq("context_stress_or_drawdown")].copy()
    selected_context_years = (
        sorted(
            int(value)
            for value in annual_walk_forward.loc[
                annual_walk_forward.get("trigger_mode", pd.Series(dtype=object)).astype(str).eq("context_stress_or_drawdown"),
                "test_year",
            ].dropna().tolist()
        )
        if not annual_walk_forward.empty
        else []
    )
    trigger_rows: list[dict[str, Any]] = []
    for (test_year, trigger_mode), group in frame.groupby(["test_year", "trigger_mode"], sort=True):
        trigger_rows.append(
            {
                "test_year": int(test_year),
                "trigger_mode": str(trigger_mode),
                "candidate_rows": int(len(group)),
                "validation_pass_count": int(group["passes_validation_target"].astype(bool).sum()),
                "test_pass_count": int(group["passes_test_target"].astype(bool).sum()),
                "best_validation_retention_pct": float(pd.to_numeric(group["validation_return_retention_pct"], errors="coerce").max()),
                "best_validation_max_drawdown_pct": float(pd.to_numeric(group["validation_max_drawdown_pct"], errors="coerce").max()),
            }
        )
    best_context_rows: list[dict[str, Any]] = []
    if not context.empty:
        sort_columns = [
            "test_year",
            "validation_return_retention_score",
            "validation_max_drawdown_pct",
            "selection_score",
        ]
        ranked = context.sort_values(sort_columns, ascending=[True, False, False, False])
        for _, row in ranked.groupby("test_year", sort=True).head(1).iterrows():
            best_context_rows.append(
                {
                    "test_year": int(row["test_year"]),
                    "variant": str(row["variant"]),
                    "context_trigger_source": str(row.get("context_trigger_source", "unknown")),
                    "cost_bps": float(row["cost_bps"]),
                    "validation_return_retention_pct": float(row["validation_return_retention_pct"]),
                    "validation_max_drawdown_pct": float(row["validation_max_drawdown_pct"]),
                    "validation_open_buy_dump_period_reduction": float(row["validation_open_buy_dump_period_reduction"]),
                    "validation_crash_period_reduction": float(row["validation_crash_period_reduction"]),
                    "passes_validation_target": bool(row["passes_validation_target"]),
                    "test_return_retention_pct": float(row["test_return_retention_pct"]),
                    "test_max_drawdown_pct": float(row["test_max_drawdown_pct"]),
                    "passes_test_target": bool(row["passes_test_target"]),
                }
            )
    return {
        "context_policy_candidate_rows": int(len(context)),
        "selected_context_trigger_years": selected_context_years,
        "trigger_mode_year_summary": _json_ready_records(pd.DataFrame(trigger_rows)),
        "best_context_validation_rows": _json_ready_records(pd.DataFrame(best_context_rows)),
    }


def _write_sidecars(
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    *,
    annual_walk_forward: pd.DataFrame,
    model_selection: pd.DataFrame,
    selected_policy_cost_sensitivity: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    feature_importance: pd.DataFrame,
    selected_model: dict[str, Any],
    analysis_summary: dict[str, Any],
) -> dict[str, str]:
    sidecars = {
        "annual_walk_forward": "annual_walk_forward.csv",
        "model_selection": "model_selection.csv",
        "selected_policy_cost_sensitivity": "selected_policy_cost_sensitivity.csv",
        "risk_analysis": "risk_analysis.csv",
        "feature_importance": "feature_importance.csv",
        "selected_model": "selected_model.json",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    annual_walk_forward.to_csv(spec.output_dir / sidecars["annual_walk_forward"], index=False)
    model_selection.to_csv(spec.output_dir / sidecars["model_selection"], index=False)
    selected_policy_cost_sensitivity.to_csv(spec.output_dir / sidecars["selected_policy_cost_sensitivity"], index=False)
    risk_analysis.to_csv(spec.output_dir / sidecars["risk_analysis"], index=False)
    feature_importance.to_csv(spec.output_dir / sidecars["feature_importance"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(
        "# Failure Modes\n\n"
        "- 本卡不是新的日内主 alpha；它保留日线 next-open 入场，只验证 5m 风控减仓/退出是否能减少买入后下杀。\n"
        "- 15m LGB/XGB/consensus 的 label 是 triple-barrier 1h/2h execution improvement，不等同于 open-to-open 收益预测。\n"
        "- 5m risk_score 字段沿用 15m 执行改善概率；风控触发方向是低概率触发，而不是高风险概率触发。\n"
        "- reenter-close 变体只降低触发日至收盘的日内风险，收盘恢复目标仓位以保留日线模型的隔夜段。\n"
        "- 5m 阈值只从 validation 年份生成；若 test 年效果差，不能用 test 重新调阈值。\n"
        "- 5m context stress 候选包含 SOXL/SOXX 本地 qlib bar 的 VWAP、SOXX 确认、波动扩张、杠杆偏离、gap reclaim、趋势质量和量价吸收/派发。\n"
        "- 外部 QQQ/SMH/SOXX lead-lag 与 NVDA/AVGO/AMD/TSM/ASML breadth 已按 15m bar 向后对齐到 5m replay，但外部历史不足完整 annual walk-forward 前只能作为诊断候选。\n"
        "- 风控 replay 已避免同一根 5m bar 收盘触发又按同一收盘成交；当前仍缺 bid/ask、盘口深度和真实成交容量，不能据此直接实盘。\n"
        "- 若 20/50bp 年度 retention、max drawdown 和风险降低不能同时过门槛，应继续作为诊断卡而非晋升。\n",
        encoding="utf-8",
    )
    return sidecars


def _write_workflow_yaml(
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
) -> Path:
    path = spec.output_dir / "workflow.yaml"
    path.write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                "qlib:",
                f"  recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
                f"  daily_provider_uri: {spec.daily_provider_uri.resolve().as_posix()}",
                f"  intraday_15min_provider_uri: {spec.intraday_provider_uri.resolve().as_posix()}",
                f"  intraday_5min_provider_uri: {spec.intraday_5min_provider_uri.resolve().as_posix()}",
                f"  external_context_15min_provider_uri: {spec.external_context_provider_uri.resolve().as_posix()}",
                "sources:",
                f"  daily_card: {spec.source_daily_card}",
                f"  daily_recorder_id: {daily_summary.get('recorder_id')}",
                f"  intraday_label_card: {spec.source_intraday_card}",
                f"  intraday_label_recorder_id: {intraday_summary.get('recorder_id')}",
                f"  intraday_score_card: {spec.source_local_fallback_card}",
                "label_policy:",
                f"  name: {LABEL_POLICY}",
                "  source: local_soxl_soxx_15m_lgb_xgb",
                "model_selection:",
                "  models: [lgb, xgb, lgb_xgb_consensus]",
                "  selection_segment: validation",
                "  report_segment: test",
                "  selection_uses_test_segment: false",
                "  required_cost_bps: [20.0, 50.0]",
                "execution:",
                f"  daily_timing: {DAILY_EXECUTION_TIMING}",
                f"  risk_sleeve_timing: {RISK_SLEEVE_EXECUTION_TIMING}",
                f"  risk_sleeve_execution_delay_bars: {RISK_SLEEVE_EXECUTION_DELAY_BARS}",
                f"  risk_sleeve_execution_price: next_5min_{RISK_SLEEVE_EXECUTION_PRICE}",
                f"  context_trigger_features: {list(CONTEXT_THRESHOLD_COLUMNS)}",
                f"  external_context_trigger_features: {list(EXTERNAL_CONTEXT_THRESHOLD_COLUMNS)}",
                "  daily_entry_is_not_skipped: true",
                "  risk_score_semantics: probability_of_positive_1h_2h_execution_improvement",
                "  risk_score_trigger_direction: low_probability_triggers_risk_action",
                f"  trigger_modes: {list(TRIGGER_MODES)}",
                "walk_forward:",
                "  policy: annual_train_to_T_minus_2_valid_T_minus_1_test_T",
                f"  required_start: {REQUIRED_WALK_FORWARD_START}",
                f"  required_test_years: {list(REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS)}",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "intraday_fold_predictions": "artifacts/five_min_risk_sleeve/intraday_fold_predictions.csv",
            "all_policy_selection": "artifacts/five_min_risk_sleeve/all_policy_selection.csv.gz",
            "selected_trades": "artifacts/five_min_risk_sleeve/selected_trades.csv",
            "daily_walk_forward_positions": "artifacts/five_min_risk_sleeve/daily_walk_forward_positions.csv",
        },
        "reproducible_from_recorder": True,
    }


def _promotion_decision(walk_forward_summary: dict[str, Any]) -> str:
    if walk_forward_summary.get("selection_uses_test_segment"):
        return "not_promoted_selection_leakage_detected"
    if not set(walk_forward_summary.get("required_test_years", [])).issubset(
        set(walk_forward_summary.get("feasible_test_years", []))
    ):
        return "not_promoted_5min_risk_sleeve_walk_forward_incomplete"
    if not set(walk_forward_summary.get("required_test_years", [])).issubset(
        set(walk_forward_summary.get("target_pass_test_years", []))
    ):
        return "not_promoted_5min_risk_sleeve_target_gate_failed"
    if not set(walk_forward_summary.get("required_test_years", [])).issubset(
        set(walk_forward_summary.get("risk_reduction_test_years", []))
    ):
        return "not_promoted_5min_risk_sleeve_risk_reduction_unstable"
    return "not_promoted_bid_ask_depth_replay_missing"


def _write_conclusion(
    spec: CognitiveAlphaIntraday5MinRiskSleeveSpec,
    *,
    summary: dict[str, Any],
    annual_walk_forward: pd.DataFrame,
) -> None:
    pass_count = len(summary["walk_forward"].get("target_pass_test_years", []))
    risk_count = len(summary["walk_forward"].get("risk_reduction_test_years", []))
    latest = annual_walk_forward.sort_values("test_year").iloc[-1].to_dict() if not annual_walk_forward.empty else {}
    external_context = summary.get("external_context", {})
    external_rows = int(external_context.get("external_context_available_5min_rows", 0) or 0)
    external_full_history = bool(external_context.get("full_walk_forward_history_available", False))
    text = (
        "# 中文结论\n\n"
        "- 做了什么：新增 5m 日内风控 sleeve。日线 open-to-open 模型仍按 next-open 入场；"
        "15m LGB/XGB/consensus triple-barrier 模型提供 1h/2h 执行改善概率，5m qlib bar 在低概率、破位或 context stress 触发后按下一根 5m 开盘做全平、半仓、1/4、1/8 减仓，或收盘恢复仓位的日内风险 replay；"
        "本轮新增 SOXL/SOXX 5m VWAP、SOXX 确认、波动扩张、杠杆偏离、gap reclaim、趋势质量和量价吸收/派发 context stress 候选，"
        "并接入 QQQ/SMH/SOXX 15m lead-lag 与半导体成分股 breadth 诊断候选。\n"
        f"- 外部 context 覆盖：5m replay 可对齐行数={external_rows}，full_walk_forward_history_available={external_full_history}；"
        "历史不足完整 2023-2026 annual walk-forward 前，外部因子只能解释/诊断，不能作为晋升门槛放宽依据。\n"
        "- 为什么日内模型没有超过日线模型：日线模型吃的是 open-to-open/隔夜/趋势段收益；"
        "日内模型学的是 1h/2h 执行改善。把它做入场 gate 会漏掉大趋势日，所以本卡改为保留入场、只控制买入后下杀。\n"
        f"- 年度 walk-forward：2023-2026 均按 train_to_T-2 / valid_T-1 / test_T；"
        f"20/50bp 同时过 retention 85%-95% 与 max_dd>-18% 的年份数={pass_count}；"
        f"降低 open-buy-dump 或 crash 暴露的年份数={risk_count}。\n"
        f"- 最新 fold：{latest.get('fold')} 选 {latest.get('selected_variant')} + {latest.get('trigger_mode')}，"
        f"20bp test retention={float(latest.get('20bp_test_return_retention_pct') or 0.0):.2f}%，"
        f"test max_dd={float(latest.get('20bp_test_max_drawdown_pct') or 0.0):.2f}%，"
        f"open-buy-dump reduction={float(latest.get('20bp_test_open_buy_dump_period_reduction') or 0.0):.0f}。\n"
        f"- 是否晋升：{summary['promotion_decision']}。若仍不过门槛，下一步不该继续调 test 阈值，"
        "而是补更长外部分钟上下文、bid/ask replay 和更明确的风险 label。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")
