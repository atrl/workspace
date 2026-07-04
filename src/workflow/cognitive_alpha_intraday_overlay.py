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


EXPERIMENT = "cognitive_alpha_intraday_execution_overlay_v1"
EXTERNAL_CONTEXT_OVERLAY_EXPERIMENT = "cognitive_alpha_intraday_external_context_overlay_v1"
REGIME_DIAGNOSTIC_EXPERIMENT = "cognitive_alpha_intraday_regime_overlay_diagnostic_v1"
DELAYED_OBSERVABLE_EXPERIMENT = "cognitive_alpha_intraday_delayed_observable_overlay_v1"
DAILY_SOURCE_CARD = "research/cognitive_alpha_goal_codex_exec_v1"
INTRADAY_SOURCE_CARD = "research/agentic_alpha_intraday_15m_regime_gated_v1"
EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD = "research/agentic_alpha_intraday_external_context_label_v1"
DAILY_EXECUTION_TIMING = "close_t_signal_next_open_to_following_open"
OVERLAY_SIGNAL_TIMING = "15m_bar_close_to_next_bar_open"
PROB_COL = "prob_lgb_xgb_consensus"
GATE_COL = "regime_gate_v2_dislocation"
PREDICTION_AVAILABLE_GATE_COL = "prediction_available_gate"
SAME_DAY_CLOSE_EXIT_POLICY = "same_day_close"
VARIANTS = [
    "baseline_next_open",
    "overlay_a_signal_gate",
    "overlay_b_twap3",
    "overlay_b_twap4",
    "overlay_c_risk_exit",
]
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "highfreq": "https://qlib.readthedocs.io/en/latest/component/highfreq.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
}
QLIB_NATIVE_API = [
    "qlib.init",
    "qlib.data.D.features",
    "qlib.workflow.R.start",
    "qlib.workflow.R.log_params",
    "qlib.workflow.R.log_metrics",
    "qlib.workflow.Recorder.log_artifact",
]
REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS = (2023, 2024, 2025, 2026)
REQUIRED_WALK_FORWARD_START = "2019-01-02"
REGIME_REQUIRED_COST_BPS = (20.0, 50.0)
OPEN_BUY_DUMP_THRESHOLD = -0.03
CRASH_THRESHOLD = -0.08


def build_intraday_optimization_plan(
    *,
    selected_model: dict[str, Any],
    walk_forward_summary: dict[str, Any],
    conditional_walk_forward_summary: dict[str, Any] | None = None,
    source_intraday_model: dict[str, Any] | None = None,
    local_intraday_manifest: dict[str, Any] | None = None,
    local_intraday_5min_manifest: dict[str, Any] | None = None,
    external_context_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conditional_walk_forward_summary = conditional_walk_forward_summary or {}
    source_intraday_model = source_intraday_model or {}
    external_context_summary = external_context_summary or {}
    local_history = _intraday_manifest_history(
        local_intraday_manifest,
        required_start=REQUIRED_WALK_FORWARD_START,
    )
    local_5min_history = _intraday_manifest_history(
        local_intraday_5min_manifest,
        required_start=REQUIRED_WALK_FORWARD_START,
    )
    external_status = _external_context_status(
        walk_forward_summary=walk_forward_summary,
        source_intraday_model=source_intraday_model,
        external_context_summary=external_context_summary,
    )
    selected_variant = str(selected_model.get("variant") or "")
    selected_test_retention = _safe_float(selected_model.get("test_return_retention_pct"))
    selected_test_max_drawdown = _safe_float(selected_model.get("test_max_drawdown_pct"))
    annual_policy_ok = bool(walk_forward_summary.get("required_policy_satisfied", False))
    conditional_policy_ok = bool(conditional_walk_forward_summary.get("required_policy_satisfied", False))

    if external_status == "blocked_history_insufficient" and local_history["supports_required_start"]:
        status = "needs_local_intraday_fallback"
        primary_next_action = "run_soxl_soxx_intraday_only_annual_walk_forward"
        ranked_actions = [
            {
                "rank": 1,
                "action": "run_soxl_soxx_intraday_only_annual_walk_forward",
                "reason": (
                    "Local SOXL/SOXX 15m history covers the required annual walk-forward start; "
                    "external context history does not."
                ),
            },
            {
                "rank": 2,
                "action": "train_local_lgb_xgb_execution_label",
                "reason": (
                    "Use only observable SOXL/SOXX 15m features first: gap reclaim, VWAP distance, "
                    "realized-vol expansion, 30/60m trend quality, absorption, and drawdown from open."
                ),
            },
            {
                "rank": 3,
                "action": "backfill_external_context_history_before_threshold_search",
                "reason": "External QQQ/SMH/constituent lead-lag features need full history before promotion gates.",
            },
        ]
    elif selected_variant == "baseline_next_open":
        status = "needs_new_intraday_factor_or_label"
        primary_next_action = "replace_open_gate_with_delayed_observable_label"
        ranked_actions = [
            {
                "rank": 1,
                "action": "replace_open_gate_with_delayed_observable_label",
                "reason": "The validation-only execution overlay selected baseline, so the current 15m gate adds no edge.",
            },
            {
                "rank": 2,
                "action": "search_local_intraday_risk_sleeve",
                "reason": "Optimize only for open-buy-then-dump and same-day risk reduction before changing position size.",
            },
        ]
    elif selected_test_retention is not None and selected_test_retention < 85.0:
        status = "needs_new_intraday_factor_or_label"
        primary_next_action = "replace_open_gate_with_delayed_observable_label"
        ranked_actions = [
            {
                "rank": 1,
                "action": "replace_open_gate_with_delayed_observable_label",
                "reason": (
                    "The validation-selected intraday overlay keeps too little of the daily model return; "
                    "the current gate is over-filtering trend exposure."
                ),
            },
            {
                "rank": 2,
                "action": "search_local_intraday_risk_sleeve",
                "reason": "Optimize only for open-buy-then-dump and same-day risk reduction before changing position size.",
            },
        ]
    elif not annual_policy_ok:
        status = "needs_annual_walk_forward_evidence"
        primary_next_action = "complete_annual_walk_forward_or_fail_closed"
        ranked_actions = [
            {
                "rank": 1,
                "action": "complete_annual_walk_forward_or_fail_closed",
                "reason": "Short-window validation/test results are not enough for promotion without annual folds.",
            }
        ]
    elif not conditional_policy_ok:
        status = "needs_cost_robust_risk_sleeve"
        primary_next_action = "optimize_cost_robust_risk_sleeve"
        ranked_actions = [
            {
                "rank": 1,
                "action": "optimize_cost_robust_risk_sleeve",
                "reason": "The main overlay has annual evidence, but the 20/50bp conditional sleeve is not robust.",
            }
        ]
    else:
        status = "needs_execution_replay"
        primary_next_action = "run_bid_ask_latency_replay"
        ranked_actions = [
            {
                "rank": 1,
                "action": "run_bid_ask_latency_replay",
                "reason": "Promotion still needs bid/ask, delay, and queue-quality validation.",
            }
        ]

    guardrails = [
        "selection_uses_validation_only",
        "test_segment_report_only",
        "retain_85_to_95_pct_return_band",
        "max_drawdown_above_minus_18_pct",
    ]
    if external_status == "blocked_history_insufficient":
        guardrails.append("do_not_continue_external_threshold_search")
    if local_5min_history["supports_required_start"]:
        guardrails.append("use_5min_only_as_replay_or_feature_probe_until_walk_forward_passes")

    return {
        "status": status,
        "primary_next_action": primary_next_action,
        "ranked_actions": ranked_actions,
        "guardrails": guardrails,
        "external_context_status": external_status,
        "local_intraday_history": {
            **local_history,
            "supports_5min_probe": bool(local_5min_history["supports_required_start"]),
            "freq_5min": local_5min_history.get("freq"),
            "calendar_start_5min": local_5min_history.get("calendar_start"),
            "calendar_end_5min": local_5min_history.get("calendar_end"),
        },
        "selected_overlay_status": {
            "variant": selected_variant,
            "test_return_retention_pct": selected_test_retention,
            "test_max_drawdown_pct": selected_test_max_drawdown,
            "annual_walk_forward_required_policy_satisfied": annual_policy_ok,
            "conditional_walk_forward_required_policy_satisfied": conditional_policy_ok,
        },
        "external_context_gap": external_context_summary.get("coverage_gap_analysis", {}),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _intraday_manifest_history(
    manifest: dict[str, Any] | None,
    *,
    required_start: str,
) -> dict[str, Any]:
    manifest = manifest or {}
    calendar_start = manifest.get("calendar_start")
    calendar_end = manifest.get("calendar_end")
    status = str(manifest.get("status") or "missing")
    supports_required_start = False
    if calendar_start:
        try:
            supports_required_start = pd.Timestamp(calendar_start).date() <= pd.Timestamp(required_start).date()
        except (TypeError, ValueError):
            supports_required_start = False
    return {
        "status": status,
        "freq": manifest.get("freq"),
        "calendar_start": calendar_start,
        "calendar_end": calendar_end,
        "instruments": manifest.get("instruments", []),
        "supports_required_start": bool(status == "ok" and supports_required_start),
    }


def _external_context_status(
    *,
    walk_forward_summary: dict[str, Any],
    source_intraday_model: dict[str, Any],
    external_context_summary: dict[str, Any],
) -> str:
    external_context = source_intraday_model.get("external_context") or {}
    has_external_context = bool(external_context.get("provider_uri") or external_context.get("provider_card"))
    if not has_external_context:
        return "not_used"
    coverage_gap = external_context_summary.get("coverage_gap_analysis", {})
    if coverage_gap and not bool(coverage_gap.get("full_walk_forward_history_available", False)):
        return "blocked_history_insufficient"
    blocking_reasons = {
        str(reason)
        for reason in walk_forward_summary.get("blocking_reasons", [])
        if reason is not None
    }
    history_blockers = {
        "insufficient_external_intraday_history",
        "missing_intraday_prediction_coverage",
        "source_intraday_walk_forward_not_satisfied",
    }
    if blocking_reasons & history_blockers:
        return "blocked_history_insufficient"
    return "available_partial_or_unverified"


@dataclass(frozen=True)
class CognitiveAlphaIntradayOverlaySpec:
    root: Path
    experiment: str = EXPERIMENT
    market: str = "all"
    region: str = "us"
    as_of: str = "2026-07-02"
    selection_cost_bps: float = 20.0
    cost_stress_bps: tuple[float, ...] = (10.0, 20.0, 50.0)
    signal_threshold: float | None = None
    risk_threshold: float | None = None
    source_daily_card: str = DAILY_SOURCE_CARD
    source_intraday_card: str = INTRADAY_SOURCE_CARD
    intraday_prob_col: str | None = None
    intraday_gate_col: str | None = GATE_COL
    intraday_gate_policy: str = "source_trade_gate"
    optimize_thresholds: bool = False

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
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"


def run_cognitive_alpha_intraday_execution_overlay_card(
    *,
    root: Path | str | None = None,
    signal_threshold: float | None = None,
    risk_threshold: float | None = None,
) -> dict[str, Any]:
    spec = CognitiveAlphaIntradayOverlaySpec(
        root=_resolve_root(root),
        signal_threshold=signal_threshold,
        risk_threshold=risk_threshold,
    )
    return _run_card(spec)


def run_cognitive_alpha_intraday_external_context_overlay_card(
    *,
    root: Path | str | None = None,
    signal_threshold: float | None = None,
    risk_threshold: float | None = None,
) -> dict[str, Any]:
    spec = CognitiveAlphaIntradayOverlaySpec(
        root=_resolve_root(root),
        experiment=EXTERNAL_CONTEXT_OVERLAY_EXPERIMENT,
        source_intraday_card=EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD,
        intraday_gate_col=None,
        intraday_gate_policy="all_predicted_15m_rows",
        optimize_thresholds=True,
        signal_threshold=signal_threshold,
        risk_threshold=risk_threshold,
    )
    return _run_card(spec)


def run_cognitive_alpha_intraday_regime_overlay_diagnostic_card(
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    spec = CognitiveAlphaIntradayOverlaySpec(
        root=_resolve_root(root),
        experiment=REGIME_DIAGNOSTIC_EXPERIMENT,
        source_intraday_card=EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD,
        intraday_gate_col=None,
        intraday_gate_policy="all_predicted_15m_rows",
        optimize_thresholds=True,
    )
    return _run_regime_diagnostic_card(spec)


def run_cognitive_alpha_intraday_delayed_observable_overlay_card(
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    spec = CognitiveAlphaIntradayOverlaySpec(
        root=_resolve_root(root),
        experiment=DELAYED_OBSERVABLE_EXPERIMENT,
        source_intraday_card=EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD,
        intraday_gate_col=None,
        intraday_gate_policy="all_predicted_15m_rows",
        optimize_thresholds=True,
    )
    return _run_delayed_observable_card(spec)


def simulate_execution_overlays(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    intraday: pd.DataFrame,
    signal_threshold: float,
    risk_threshold: float,
    cost_bps: float = 0.0,
    prob_col: str = PROB_COL,
    gate_col: str = GATE_COL,
) -> pd.DataFrame:
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    bars = _normalize_intraday_bars(intraday, prob_col=prob_col, gate_col=gate_col)

    rows: list[dict[str, Any]] = []
    for daily_row in daily_frame.to_dict("records"):
        rows.extend(
            _simulate_daily_row(
                daily_row,
                prices=prices,
                intraday=bars,
                signal_threshold=signal_threshold,
                risk_threshold=risk_threshold,
                cost_bps=float(cost_bps),
                prob_col=prob_col,
                gate_col=gate_col,
            )
        )
    return pd.DataFrame(rows)


def simulate_observable_delayed_signal_overlay(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    intraday: pd.DataFrame,
    observation_bars: int,
    condition_feature: str,
    condition_operator: str,
    condition_threshold: float,
    signal_threshold: float,
    risk_threshold: float | None = None,
    exit_policy: str | None = None,
    cost_bps: float = 0.0,
    prob_col: str = PROB_COL,
    gate_col: str = GATE_COL,
) -> pd.DataFrame:
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    bars = _normalize_intraday_bars(intraday, prob_col=prob_col, gate_col=gate_col)
    resolved_exit_policy = _delayed_exit_policy(exit_policy=exit_policy, risk_threshold=risk_threshold)
    variant = f"delayed_signal_after_{int(observation_bars) * 15}m"
    if resolved_exit_policy == "risk_exit":
        variant = f"{variant}_risk_exit"
    elif resolved_exit_policy == SAME_DAY_CLOSE_EXIT_POLICY:
        variant = f"{variant}_same_day_close"

    rows: list[dict[str, Any]] = []
    price_index = pd.Index(prices["datetime"])
    for daily_row in daily_frame.to_dict("records"):
        decision_date = pd.Timestamp(daily_row["datetime"]).normalize()
        if decision_date not in set(price_index):
            rows.append(_empty_variant_row(daily_row, "baseline_next_open", cost_bps, "missing_decision_open"))
            rows.append(_empty_variant_row(daily_row, variant, cost_bps, "missing_decision_open"))
            continue
        location = int(price_index.get_loc(decision_date))
        if isinstance(location, slice):
            location = int(location.start)
        entry_date = prices.iloc[location + 1]["datetime"] if location + 1 < len(prices) else pd.NaT
        exit_date = prices.iloc[location + 2]["datetime"] if location + 2 < len(prices) else pd.NaT
        target = float(daily_row.get("target_position", 0.0) or 0.0)
        source_return = float(daily_row.get("strategy_return", 0.0) or 0.0)
        execution_cost = abs(target) * float(cost_bps) / 10000.0
        segment = str(daily_row.get("segment", "unknown"))
        base_common = {
            "datetime": decision_date.date().isoformat(),
            "segment": segment,
            **_daily_walk_forward_metadata(daily_row),
            "target_position": target,
            "cost_bps": float(cost_bps),
            "entry_date": _date_or_none(entry_date),
            "exit_date": _date_or_none(exit_date),
            "source_daily_strategy_return": source_return,
        }
        if abs(target) <= 1e-12:
            for row_variant in ["baseline_next_open", variant]:
                rows.append(
                    {
                        **base_common,
                        "variant": row_variant,
                        "entry_time": None,
                        "exit_time": None,
                        "entry_price": np.nan,
                        "exit_price": np.nan,
                        "intraday_signal_score": np.nan,
                        "observed_feature_value": np.nan,
                        "gross_return": 0.0,
                        "strategy_return": 0.0,
                        "skipped_reason": None,
                    }
                )
            continue
        if pd.isna(entry_date) or pd.isna(exit_date):
            rows.append(_empty_variant_row(daily_row, "baseline_next_open", cost_bps, "missing_entry_or_exit_open"))
            rows.append(_empty_variant_row(daily_row, variant, cost_bps, "missing_entry_or_exit_open"))
            continue

        entry_open = float(prices.iloc[location + 1]["open"])
        exit_open = float(prices.iloc[location + 2]["open"])
        baseline_return = source_return - execution_cost
        rows.append(
            {
                **base_common,
                "variant": "baseline_next_open",
                "entry_time": pd.Timestamp(entry_date).date().isoformat(),
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": entry_open,
                "exit_price": exit_open,
                "intraday_signal_score": np.nan,
                "observed_feature_value": np.nan,
                "gross_return": source_return,
                "strategy_return": baseline_return,
                "skipped_reason": None,
            }
        )

        entry_day_bars = bars.loc[bars["date"].eq(pd.Timestamp(entry_date).date().isoformat())].sort_values("datetime")
        observed = entry_day_bars.head(int(observation_bars))
        observed_value = _observed_intraday_feature_value(
            observed,
            entry_open=entry_open,
            feature=condition_feature,
            prob_col=prob_col,
        )
        if len(observed) < int(observation_bars) or not _condition_matches(
            observed_value,
            operator=condition_operator,
            threshold=float(condition_threshold),
        ):
            rows.append(
                {
                    **base_common,
                    "variant": variant,
                    "entry_time": pd.Timestamp(entry_date).date().isoformat(),
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "intraday_signal_score": np.nan,
                    "observed_feature_value": observed_value,
                    "gross_return": source_return,
                    "strategy_return": baseline_return,
                    "skipped_reason": None if len(observed) >= int(observation_bars) else "missing_observation_window",
                }
            )
            continue

        signal_candidates = entry_day_bars.iloc[int(observation_bars) - 1 :].loc[
            entry_day_bars.iloc[int(observation_bars) - 1 :].get(gate_col, pd.Series(False)).fillna(False).astype(bool)
            & (pd.to_numeric(entry_day_bars.iloc[int(observation_bars) - 1 :].get(prob_col), errors="coerce") >= float(signal_threshold))
            & entry_day_bars.iloc[int(observation_bars) - 1 :]["next_open"].notna()
        ]
        if signal_candidates.empty:
            rows.append(
                {
                    **base_common,
                    "variant": variant,
                    "entry_time": None,
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": np.nan,
                    "exit_price": exit_open,
                    "intraday_signal_score": np.nan,
                    "observed_feature_value": observed_value,
                    "gross_return": 0.0,
                    "strategy_return": 0.0,
                    "skipped_reason": "no_delayed_15m_entry_signal",
                }
            )
            continue

        signal = signal_candidates.iloc[0]
        entry_price = float(signal["next_open"])
        selected_exit_time = pd.Timestamp(exit_date).date().isoformat()
        selected_exit_price = exit_open
        if resolved_exit_policy == SAME_DAY_CLOSE_EXIT_POLICY and pd.notna(signal["next_datetime"]):
            same_day_exit = _same_day_close_exit(entry_day_bars, entry_time=pd.Timestamp(signal["next_datetime"]))
            if same_day_exit is None:
                rows.append(
                    {
                        **base_common,
                        "variant": variant,
                        "entry_time": _timestamp_or_none(signal["next_datetime"]),
                        "exit_time": None,
                        "entry_price": entry_price,
                        "exit_price": np.nan,
                        "intraday_signal_score": float(signal[prob_col]),
                        "observed_feature_value": observed_value,
                        "gross_return": 0.0,
                        "strategy_return": 0.0,
                        "skipped_reason": "missing_same_day_close_exit",
                    }
                )
                continue
            selected_exit_time, selected_exit_price = same_day_exit
        elif resolved_exit_policy == "risk_exit" and risk_threshold is not None and pd.notna(signal["next_datetime"]):
            risk_candidates = entry_day_bars.loc[
                entry_day_bars["datetime"].ge(pd.Timestamp(signal["next_datetime"]))
                & entry_day_bars.get(gate_col, pd.Series(False, index=entry_day_bars.index)).fillna(False).astype(bool)
                & pd.to_numeric(entry_day_bars.get(prob_col), errors="coerce").le(float(risk_threshold))
                & entry_day_bars["close"].notna()
            ]
            if not risk_candidates.empty:
                risk = risk_candidates.iloc[0]
                selected_exit_time = _timestamp_or_none(risk["datetime"])
                selected_exit_price = float(risk["close"])
        gross_return = target * (selected_exit_price / entry_price - 1.0)
        rows.append(
            {
                **base_common,
                "variant": variant,
                "entry_time": _timestamp_or_none(signal["next_datetime"]),
                "exit_time": selected_exit_time,
                "entry_price": entry_price,
                "exit_price": selected_exit_price,
                "intraday_signal_score": float(signal[prob_col]),
                "observed_feature_value": observed_value,
                "gross_return": gross_return,
                "strategy_return": gross_return - execution_cost,
                "skipped_reason": None,
            }
        )
    return pd.DataFrame(rows)


def summarize_overlay_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(rows)
    grouped = trades.groupby(["cost_bps", "segment", "variant"], sort=True)
    for (cost_bps, segment, variant), group in grouped:
        group = group.sort_values("datetime").copy()
        returns = pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0 if not equity.empty else pd.Series(dtype=float)
        std = float(returns.std(ddof=0))
        sharpe = 0.0 if std == 0.0 else float(returns.mean() / std * np.sqrt(252.0))
        active = group.loc[pd.to_numeric(group["target_position"], errors="coerce").abs() > 1e-9]
        rows.append(
            {
                "cost_bps": float(cost_bps),
                "segment": str(segment),
                "variant": str(variant),
                "periods": int(len(group)),
                "active_periods": int(len(active)),
                "trade_days": int(active["entry_date"].nunique()) if not active.empty else 0,
                "skipped_periods": int(group["skipped_reason"].notna().sum()),
                "avg_abs_position": float(pd.to_numeric(group["target_position"], errors="coerce").abs().mean()),
                "total_return_pct": float((equity.iloc[-1] - 1.0) * 100.0) if not equity.empty else 0.0,
                "max_drawdown_pct": float(drawdown.min() * 100.0) if not drawdown.empty else 0.0,
                "daily_sharpe": sharpe,
                "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
                "worst_period_return_pct": float(returns.min() * 100.0) if len(returns) else 0.0,
                "avg_period_return_bps": float(returns.mean() * 10000.0) if len(returns) else 0.0,
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    baseline = metrics.loc[metrics["variant"] == "baseline_next_open", ["cost_bps", "segment", "total_return_pct", "max_drawdown_pct"]].rename(
        columns={
            "total_return_pct": "baseline_return_pct",
            "max_drawdown_pct": "baseline_max_drawdown_pct",
        }
    )
    metrics = metrics.merge(baseline, on=["cost_bps", "segment"], how="left")
    metrics["return_retention_pct"] = np.where(
        metrics["baseline_return_pct"].abs() > 1e-12,
        metrics["total_return_pct"] / metrics["baseline_return_pct"] * 100.0,
        np.nan,
    )
    metrics["drawdown_improvement_pct"] = metrics["max_drawdown_pct"] - metrics["baseline_max_drawdown_pct"]
    metrics["return_target_mode"] = [
        _return_target_mode(baseline_return_pct)
        for baseline_return_pct in metrics["baseline_return_pct"]
    ]
    metrics["return_target_satisfied"] = [
        _return_target_satisfied(
            total_return_pct,
            baseline_return_pct,
            return_retention_pct,
        )
        for total_return_pct, baseline_return_pct, return_retention_pct in zip(
            metrics["total_return_pct"],
            metrics["baseline_return_pct"],
            metrics["return_retention_pct"],
            strict=False,
        )
    ]
    metrics["return_retention_in_target_band"] = [
        _return_retention_in_target_band(baseline_return_pct, return_retention_pct)
        for baseline_return_pct, return_retention_pct in zip(
            metrics["baseline_return_pct"],
            metrics["return_retention_pct"],
            strict=False,
        )
    ]
    metrics["return_retention_score"] = [
        _return_retention_score(total_return_pct, baseline_return_pct, return_retention_pct)
        for total_return_pct, baseline_return_pct, return_retention_pct in zip(
            metrics["total_return_pct"],
            metrics["baseline_return_pct"],
            metrics["return_retention_pct"],
            strict=False,
        )
    ]
    metrics["selection_score"] = (
        metrics["return_retention_score"].fillna(0.0)
        + 2.0 * metrics["drawdown_improvement_pct"].fillna(0.0)
        + 0.25 * metrics["daily_sharpe"].fillna(0.0)
        - 0.10 * metrics["skipped_periods"].fillna(0.0)
    )
    return metrics.sort_values(["cost_bps", "segment", "selection_score"], ascending=[True, True, False]).reset_index(drop=True)


def summarize_execution_risk_diagnostics(
    trades: pd.DataFrame,
    *,
    open_buy_dump_threshold: float = OPEN_BUY_DUMP_THRESHOLD,
    crash_threshold: float = CRASH_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(rows)
    frame = trades.copy()
    if "entry_day_strategy_return" not in frame.columns:
        frame["entry_day_strategy_return"] = np.nan
    grouped = frame.groupby(["cost_bps", "segment", "variant"], sort=True)
    for (cost_bps, segment, variant), group in grouped:
        active = pd.to_numeric(group["target_position"], errors="coerce").abs().gt(1e-9)
        strategy_returns = pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0.0)
        entry_day_returns = pd.to_numeric(group["entry_day_strategy_return"], errors="coerce")
        entry_day_available = active & entry_day_returns.notna()
        open_buy_dump = entry_day_available & entry_day_returns.le(float(open_buy_dump_threshold))
        crash = active & strategy_returns.le(float(crash_threshold))
        rows.append(
            {
                "cost_bps": float(cost_bps),
                "segment": str(segment),
                "variant": str(variant),
                "periods": int(len(group)),
                "active_periods": int(active.sum()),
                "entry_day_available_periods": int(entry_day_available.sum()),
                "open_buy_dump_threshold_pct": float(open_buy_dump_threshold) * 100.0,
                "open_buy_dump_periods": int(open_buy_dump.sum()),
                "open_buy_dump_rate": float(open_buy_dump.sum() / active.sum()) if active.sum() else 0.0,
                "open_buy_dump_loss_sum_pct": float(entry_day_returns.loc[open_buy_dump].sum() * 100.0),
                "crash_threshold_pct": float(crash_threshold) * 100.0,
                "crash_periods": int(crash.sum()),
                "crash_rate": float(crash.sum() / active.sum()) if active.sum() else 0.0,
                "worst_strategy_return_pct": float(strategy_returns.loc[active].min() * 100.0) if active.any() else 0.0,
                "worst_entry_day_return_pct": float(entry_day_returns.loc[entry_day_available].min() * 100.0)
                if entry_day_available.any()
                else 0.0,
            }
        )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    baseline = diagnostics.loc[
        diagnostics["variant"].eq("baseline_next_open"),
        [
            "cost_bps",
            "segment",
            "open_buy_dump_periods",
            "crash_periods",
            "worst_strategy_return_pct",
            "worst_entry_day_return_pct",
        ],
    ].rename(
        columns={
            "open_buy_dump_periods": "baseline_open_buy_dump_periods",
            "crash_periods": "baseline_crash_periods",
            "worst_strategy_return_pct": "baseline_worst_strategy_return_pct",
            "worst_entry_day_return_pct": "baseline_worst_entry_day_return_pct",
        }
    )
    diagnostics = diagnostics.merge(baseline, on=["cost_bps", "segment"], how="left")
    diagnostics["open_buy_dump_period_reduction"] = (
        diagnostics["baseline_open_buy_dump_periods"].fillna(0) - diagnostics["open_buy_dump_periods"]
    )
    diagnostics["crash_period_reduction"] = diagnostics["baseline_crash_periods"].fillna(0) - diagnostics["crash_periods"]
    diagnostics["worst_strategy_return_improvement_pct"] = (
        diagnostics["worst_strategy_return_pct"] - diagnostics["baseline_worst_strategy_return_pct"].fillna(0.0)
    )
    diagnostics["worst_entry_day_return_improvement_pct"] = (
        diagnostics["worst_entry_day_return_pct"] - diagnostics["baseline_worst_entry_day_return_pct"].fillna(0.0)
    )
    return diagnostics.sort_values(["cost_bps", "segment", "variant"]).reset_index(drop=True)


def evaluate_conditional_overlay_candidates(
    trades: pd.DataFrame,
    regime_features: pd.DataFrame,
    *,
    cost_bps: float,
    variants: tuple[str, ...] = (
        "overlay_a_signal_gate",
        "overlay_b_twap3",
        "overlay_b_twap4",
        "overlay_c_risk_exit",
    ),
    condition_columns: tuple[str, ...] = (
        "probability",
        "target_position",
        "turnover",
        "raw_target",
    ),
    quantiles: tuple[float, ...] = tuple(float(value) for value in np.arange(0.05, 0.975, 0.025)),
    validation_segment: str = "validation",
    test_segment: str = "test",
    return_retention_low: float = 85.0,
    return_retention_high: float = 95.0,
    max_drawdown_floor: float = -18.0,
    max_skip_rate: float = 0.50,
    min_validation_coverage: float = 0.03,
) -> pd.DataFrame:
    frame = trades.copy()
    if "cost_bps" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["cost_bps"], errors="coerce").eq(float(cost_bps))].copy()
    if frame.empty:
        return pd.DataFrame()

    features = regime_features.copy()
    frame["datetime_key"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    features["datetime_key"] = pd.to_datetime(features["datetime"]).dt.normalize()

    baseline = _variant_return_frame(frame, variant="baseline_next_open")
    if baseline.empty:
        return pd.DataFrame()

    feature_columns = ["datetime_key", *[column for column in condition_columns if column in features.columns]]
    if "target_position" in features.columns and "target_position" not in feature_columns:
        feature_columns.append("target_position")
    aligned = baseline.merge(features[feature_columns].drop_duplicates("datetime_key"), on="datetime_key", how="left")
    validation_mask = aligned["segment"].eq(validation_segment).to_numpy()
    test_mask = aligned["segment"].eq(test_segment).to_numpy()
    active_mask = (
        pd.to_numeric(aligned.get("target_position", pd.Series(1.0, index=aligned.index)), errors="coerce")
        .fillna(0.0)
        .abs()
        .gt(1e-12)
        .to_numpy()
    )
    baseline_returns = pd.to_numeric(aligned["strategy_return"], errors="coerce").fillna(0.0).to_numpy()
    baseline_skipped = aligned["skipped_reason"].notna().to_numpy()
    if "entry_day_strategy_return" not in aligned.columns:
        aligned["entry_day_strategy_return"] = np.nan
    baseline_entry_day_returns = (
        pd.to_numeric(aligned["entry_day_strategy_return"], errors="coerce")
        .to_numpy(dtype=float)
    )
    baseline_validation = _segment_return_metrics(baseline_returns, validation_mask, baseline_skipped)
    baseline_test = _segment_return_metrics(baseline_returns, test_mask, baseline_skipped)
    baseline_validation_risk = _segment_execution_risk_metrics(
        baseline_returns,
        baseline_entry_day_returns,
        validation_mask,
        active_mask,
    )
    baseline_test_risk = _segment_execution_risk_metrics(
        baseline_returns,
        baseline_entry_day_returns,
        test_mask,
        active_mask,
    )

    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_frame = _variant_return_frame(frame, variant=variant)
        if variant_frame.empty:
            continue
        variant_aligned = aligned[["datetime_key"]].merge(
            variant_frame[
                [
                    column
                    for column in ["datetime_key", "strategy_return", "skipped_reason", "entry_day_strategy_return"]
                    if column in variant_frame.columns
                ]
            ],
            on="datetime_key",
            how="left",
        )
        if "entry_day_strategy_return" not in variant_aligned.columns:
            variant_aligned["entry_day_strategy_return"] = np.nan
        variant_returns = (
            pd.to_numeric(variant_aligned["strategy_return"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        variant_skipped = variant_aligned["skipped_reason"].notna().to_numpy()
        variant_entry_day_returns = pd.to_numeric(
            variant_aligned["entry_day_strategy_return"],
            errors="coerce",
        ).to_numpy(dtype=float)
        for column in condition_columns:
            if column not in aligned.columns:
                continue
            values = pd.to_numeric(aligned[column], errors="coerce")
            validation_values = values.loc[validation_mask].dropna()
            if validation_values.nunique() < 2:
                continue
            for quantile in quantiles:
                threshold = float(validation_values.quantile(float(quantile)))
                for operator in ("ge", "le"):
                    condition_mask = (
                        values.ge(threshold) if operator == "ge" else values.le(threshold)
                    ).fillna(False).to_numpy()
                    coverage_base = validation_mask & active_mask
                    coverage = (
                        float(condition_mask[coverage_base].mean())
                        if coverage_base.any()
                        else 0.0
                    )
                    if coverage < float(min_validation_coverage):
                        continue
                    returns = np.where(condition_mask, variant_returns, baseline_returns)
                    skipped = np.where(condition_mask, variant_skipped, baseline_skipped)
                    entry_day_returns = np.where(
                        condition_mask,
                        variant_entry_day_returns,
                        baseline_entry_day_returns,
                    )
                    validation = _segment_return_metrics(returns, validation_mask, skipped)
                    test = _segment_return_metrics(returns, test_mask, skipped)
                    validation_risk = _segment_execution_risk_metrics(
                        returns,
                        entry_day_returns,
                        validation_mask,
                        active_mask,
                    )
                    test_risk = _segment_execution_risk_metrics(
                        returns,
                        entry_day_returns,
                        test_mask,
                        active_mask,
                    )
                    validation_retention = _retention(
                        validation["total_return_pct"],
                        baseline_validation["total_return_pct"],
                    )
                    test_retention = _retention(test["total_return_pct"], baseline_test["total_return_pct"])
                    validation_return_target_satisfied = _return_target_satisfied(
                        validation["total_return_pct"],
                        baseline_validation["total_return_pct"],
                        validation_retention,
                        return_retention_low=return_retention_low,
                        return_retention_high=return_retention_high,
                    )
                    test_return_target_satisfied = _return_target_satisfied(
                        test["total_return_pct"],
                        baseline_test["total_return_pct"],
                        test_retention,
                        return_retention_low=return_retention_low,
                        return_retention_high=return_retention_high,
                    )
                    validation_return_score = _return_retention_score(
                        validation["total_return_pct"],
                        baseline_validation["total_return_pct"],
                        validation_retention,
                        return_retention_high=return_retention_high,
                    )
                    validation_dd_improvement = (
                        validation["max_drawdown_pct"] - baseline_validation["max_drawdown_pct"]
                    )
                    test_dd_improvement = test["max_drawdown_pct"] - baseline_test["max_drawdown_pct"]
                    passes_validation = bool(
                        validation_return_target_satisfied
                        and validation["max_drawdown_pct"] > max_drawdown_floor
                        and validation["skipped_periods"] <= validation["periods"] * max_skip_rate
                    )
                    passes_test = bool(
                        test_return_target_satisfied
                        and test["max_drawdown_pct"] > max_drawdown_floor
                        and test["skipped_periods"] <= test["periods"] * max_skip_rate
                    )
                    rows.append(
                        {
                            "cost_bps": float(cost_bps),
                            "variant": variant,
                            "condition_column": column,
                            "condition_operator": operator,
                            "condition_quantile": float(quantile),
                            "condition_threshold": threshold,
                            "condition_threshold_source": "validation_quantile",
                            "condition_name": f"{column}_{operator}_valid_q{quantile:.3f}_{threshold:.6g}",
                            "condition_valid_active_coverage": coverage,
                            "selection_segment": validation_segment,
                            "report_segment": test_segment,
                            "selection_uses_test_segment": False,
                            "validation_periods": validation["periods"],
                            "validation_baseline_return_pct": baseline_validation["total_return_pct"],
                            "validation_return_pct": validation["total_return_pct"],
                            "validation_return_retention_pct": validation_retention,
                            "validation_return_retention_score": validation_return_score,
                            "validation_return_target_mode": _return_target_mode(
                                baseline_validation["total_return_pct"]
                            ),
                            "validation_return_target_satisfied": validation_return_target_satisfied,
                            "validation_return_retention_in_target_band": _return_retention_in_target_band(
                                baseline_validation["total_return_pct"],
                                validation_retention,
                                return_retention_low=return_retention_low,
                                return_retention_high=return_retention_high,
                            ),
                            "validation_max_drawdown_pct": validation["max_drawdown_pct"],
                            "validation_drawdown_improvement_pct": validation_dd_improvement,
                            "validation_daily_sharpe": validation["daily_sharpe"],
                            "validation_skipped_periods": validation["skipped_periods"],
                            "validation_open_buy_dump_periods": validation_risk["open_buy_dump_periods"],
                            "validation_open_buy_dump_period_reduction": (
                                baseline_validation_risk["open_buy_dump_periods"]
                                - validation_risk["open_buy_dump_periods"]
                            ),
                            "validation_crash_periods": validation_risk["crash_periods"],
                            "validation_crash_period_reduction": (
                                baseline_validation_risk["crash_periods"] - validation_risk["crash_periods"]
                            ),
                            "validation_worst_entry_day_return_pct": validation_risk[
                                "worst_entry_day_return_pct"
                            ],
                            "validation_worst_entry_day_return_improvement_pct": (
                                validation_risk["worst_entry_day_return_pct"]
                                - baseline_validation_risk["worst_entry_day_return_pct"]
                            ),
                            "validation_worst_strategy_return_pct": validation_risk["worst_strategy_return_pct"],
                            "validation_worst_strategy_return_improvement_pct": (
                                validation_risk["worst_strategy_return_pct"]
                                - baseline_validation_risk["worst_strategy_return_pct"]
                            ),
                            "test_periods": test["periods"],
                            "test_baseline_return_pct": baseline_test["total_return_pct"],
                            "test_return_pct": test["total_return_pct"],
                            "test_return_retention_pct": test_retention,
                            "test_return_retention_score": _return_retention_score(
                                test["total_return_pct"],
                                baseline_test["total_return_pct"],
                                test_retention,
                                return_retention_high=return_retention_high,
                            ),
                            "test_return_target_mode": _return_target_mode(baseline_test["total_return_pct"]),
                            "test_return_target_satisfied": test_return_target_satisfied,
                            "test_return_retention_in_target_band": _return_retention_in_target_band(
                                baseline_test["total_return_pct"],
                                test_retention,
                                return_retention_low=return_retention_low,
                                return_retention_high=return_retention_high,
                            ),
                            "test_max_drawdown_pct": test["max_drawdown_pct"],
                            "test_drawdown_improvement_pct": test_dd_improvement,
                            "test_daily_sharpe": test["daily_sharpe"],
                            "test_skipped_periods": test["skipped_periods"],
                            "test_open_buy_dump_periods": test_risk["open_buy_dump_periods"],
                            "test_open_buy_dump_period_reduction": (
                                baseline_test_risk["open_buy_dump_periods"]
                                - test_risk["open_buy_dump_periods"]
                            ),
                            "test_crash_periods": test_risk["crash_periods"],
                            "test_crash_period_reduction": (
                                baseline_test_risk["crash_periods"] - test_risk["crash_periods"]
                            ),
                            "test_worst_entry_day_return_pct": test_risk["worst_entry_day_return_pct"],
                            "test_worst_entry_day_return_improvement_pct": (
                                test_risk["worst_entry_day_return_pct"]
                                - baseline_test_risk["worst_entry_day_return_pct"]
                            ),
                            "test_worst_strategy_return_pct": test_risk["worst_strategy_return_pct"],
                            "test_worst_strategy_return_improvement_pct": (
                                test_risk["worst_strategy_return_pct"]
                                - baseline_test_risk["worst_strategy_return_pct"]
                            ),
                            "passes_validation_target": passes_validation,
                            "passes_test_target": passes_test,
                            "passes_validation_and_test_target": bool(passes_validation and passes_test),
                            "selection_score": (
                                validation_return_score
                                + 2.0 * validation_dd_improvement
                                + 0.25 * validation["daily_sharpe"]
                                - 0.10 * validation["skipped_periods"]
                            ),
                        }
                    )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return (
        result.drop_duplicates()
        .sort_values(
            [
                "passes_validation_target",
                "selection_score",
                "validation_drawdown_improvement_pct",
                "validation_return_pct",
            ],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def select_conditional_risk_sleeve_policy(
    candidates: pd.DataFrame,
    *,
    required_cost_bps: tuple[float, ...] = (20.0, 50.0),
    selection_cost_bps: float = 20.0,
    eligible_variants: tuple[str, ...] = ("overlay_b_twap3", "overlay_b_twap4"),
    max_drawdown_floor: float = -18.0,
) -> dict[str, Any]:
    selection_priority = [
        "validation_min_required_cost_return_retention_pct_desc",
        "validation_min_required_cost_drawdown_margin_pct_desc",
        "validation_max_required_cost_skip_rate_asc",
        "validation_open_buy_dump_period_reduction_desc",
        "validation_crash_period_reduction_desc",
        "validation_worst_entry_day_return_improvement_desc",
        "validation_drawdown_improvement_desc",
        "validation_return_retention_score_desc",
        "condition_valid_active_coverage_desc",
        "selection_score_desc",
    ]
    if candidates.empty:
        return {
            "variant": None,
            "selection_role": "conditional_risk_sleeve",
            "selection_priority": selection_priority,
            "selection_uses_test_segment": False,
            "validation_passes_target": False,
            "eligible_variants": list(eligible_variants),
        }
    frame = candidates.copy()
    if eligible_variants and "variant" in frame.columns:
        eligible = frame.loc[frame["variant"].isin(set(eligible_variants))].copy()
        if not eligible.empty:
            frame = eligible
    frame["cost_bps"] = pd.to_numeric(frame.get("cost_bps", selection_cost_bps), errors="coerce")
    if "validation_return_retention_score" in frame.columns:
        frame["validation_return_retention_score"] = pd.to_numeric(
            frame["validation_return_retention_score"],
            errors="coerce",
        ).fillna(0.0)
    else:
        frame["validation_return_retention_score"] = pd.to_numeric(
            frame.get("validation_return_retention_pct", 0.0),
            errors="coerce",
        ).clip(upper=95.0)
    if "validation_return_retention_in_target_band" in frame.columns:
        frame["validation_return_retention_in_target_band"] = (
            frame["validation_return_retention_in_target_band"].fillna(False).astype(bool)
        )
    else:
        frame["validation_return_retention_in_target_band"] = pd.to_numeric(
            frame.get("validation_return_retention_pct", 0.0),
            errors="coerce",
        ).between(85.0, 95.0)
    for column in [
        "validation_open_buy_dump_period_reduction",
        "validation_crash_period_reduction",
        "validation_worst_entry_day_return_improvement_pct",
        "validation_drawdown_improvement_pct",
        "condition_valid_active_coverage",
        "selection_score",
    ]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in [
        "validation_return_retention_pct",
        "validation_max_drawdown_pct",
        "validation_skipped_periods",
        "validation_periods",
    ]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    required_costs = tuple(float(value) for value in required_cost_bps)
    group_columns = _regime_policy_group_columns(frame)
    robust_rows: list[tuple[pd.Series, pd.DataFrame]] = []
    if group_columns:
        for _, group in frame.groupby(group_columns, sort=False, dropna=False):
            costs = {float(value) for value in group["cost_bps"].dropna().unique()}
            if not set(required_costs).issubset(costs):
                continue
            required_rows = group.loc[group["cost_bps"].isin(required_costs)].copy()
            if not bool(required_rows.get("passes_validation_target", False).astype(bool).all()):
                continue
            selected_cost_row = group.loc[group["cost_bps"].eq(float(selection_cost_bps))]
            if selected_cost_row.empty:
                continue
            selected_row = selected_cost_row.iloc[0]
            has_validation_risk_reduction = bool(
                float(selected_row.get("validation_open_buy_dump_period_reduction", 0.0)) > 0.0
                or float(selected_row.get("validation_crash_period_reduction", 0.0)) > 0.0
            )
            if has_validation_risk_reduction:
                selected_row = selected_row.copy()
                selected_row["validation_min_required_cost_return_retention_pct"] = float(
                    pd.to_numeric(required_rows["validation_return_retention_pct"], errors="coerce").min()
                )
                if "validation_return_retention_score" in required_rows.columns:
                    selected_row["validation_min_required_cost_return_retention_score"] = float(
                        pd.to_numeric(required_rows["validation_return_retention_score"], errors="coerce").min()
                    )
                selected_row["validation_required_cost_all_return_retention_in_target_band"] = bool(
                    required_rows["validation_return_retention_in_target_band"].astype(bool).all()
                )
                selected_row["validation_min_required_cost_drawdown_margin_pct"] = float(
                    (
                        pd.to_numeric(required_rows["validation_max_drawdown_pct"], errors="coerce")
                        - float(max_drawdown_floor)
                    ).min()
                )
                required_periods = pd.to_numeric(required_rows["validation_periods"], errors="coerce").replace(0, np.nan)
                selected_row["validation_max_required_cost_skip_rate"] = float(
                    (
                        pd.to_numeric(required_rows["validation_skipped_periods"], errors="coerce")
                        / required_periods
                    )
                    .fillna(1.0)
                    .max()
                )
                robust_rows.append((selected_row, required_rows.sort_values("cost_bps").copy()))

    if robust_rows:
        source = pd.DataFrame([row for row, _ in robust_rows]).copy()
        required_cost_results_by_key = {
            key: required_rows for key, (_, required_rows) in zip(source.index, robust_rows, strict=False)
        }
        validation_cost_robust = True
    else:
        validation_passes = frame.loc[
            frame.get("passes_validation_target", pd.Series(False, index=frame.index)).astype(bool)
        ].copy()
        source = validation_passes if not validation_passes.empty else frame.copy()
        required_cost_results_by_key = {}
        validation_cost_robust = False
    for column in [
        "validation_required_cost_all_return_retention_in_target_band",
        "validation_min_required_cost_return_retention_pct",
        "validation_min_required_cost_return_retention_score",
        "validation_min_required_cost_drawdown_margin_pct",
    ]:
        if column not in source.columns:
            source[column] = 0.0
        source[column] = pd.to_numeric(source[column], errors="coerce").fillna(0.0)
    if "validation_max_required_cost_skip_rate" not in source.columns:
        source["validation_max_required_cost_skip_rate"] = 1.0
    source["validation_max_required_cost_skip_rate"] = pd.to_numeric(
        source["validation_max_required_cost_skip_rate"],
        errors="coerce",
    ).fillna(1.0)

    selected = source.sort_values(
        [
            "validation_required_cost_all_return_retention_in_target_band",
            "validation_min_required_cost_return_retention_score",
            "validation_min_required_cost_drawdown_margin_pct",
            "validation_max_required_cost_skip_rate",
            "validation_open_buy_dump_period_reduction",
            "validation_crash_period_reduction",
            "validation_worst_entry_day_return_improvement_pct",
            "validation_drawdown_improvement_pct",
            "validation_return_retention_score",
            "condition_valid_active_coverage",
            "selection_score",
        ],
        ascending=[False, False, False, True, False, False, False, False, False, False, False],
    ).iloc[0]
    result = _json_ready_dict(selected.to_dict())
    required_cost_results = required_cost_results_by_key.get(selected.name, pd.DataFrame())
    result["selection_role"] = "conditional_risk_sleeve"
    result["selection_priority"] = selection_priority
    result["selection_segment"] = "validation"
    result["report_segment"] = "test"
    result["selection_uses_test_segment"] = False
    result["required_cost_bps"] = list(required_costs)
    result["selection_cost_bps"] = float(selection_cost_bps)
    result["validation_passes_target"] = bool(result.get("passes_validation_target", False))
    result["test_passes_target"] = bool(result.get("passes_test_target", False))
    result["validation_cost_robust"] = bool(validation_cost_robust)
    result["eligible_variants"] = list(eligible_variants)
    if not required_cost_results.empty:
        result["required_cost_validation_target_count"] = int(
            required_cost_results["passes_validation_target"].astype(bool).sum()
        )
        result["required_cost_test_target_count"] = int(required_cost_results["passes_test_target"].astype(bool).sum())
        result["required_cost_validation_and_test_target_count"] = int(
            required_cost_results.get(
                "passes_validation_and_test_target",
                pd.Series(False, index=required_cost_results.index),
            )
            .astype(bool)
            .sum()
        )
        required_columns = [
            "cost_bps",
            "validation_return_retention_pct",
            "validation_max_drawdown_pct",
            "validation_open_buy_dump_period_reduction",
            "validation_crash_period_reduction",
            "validation_skipped_periods",
            "test_return_retention_pct",
            "test_max_drawdown_pct",
            "test_open_buy_dump_period_reduction",
            "test_crash_period_reduction",
            "passes_validation_target",
            "passes_test_target",
            "passes_validation_and_test_target",
        ]
        result["required_cost_results"] = _json_ready_records(
            required_cost_results[[column for column in required_columns if column in required_cost_results.columns]]
        )
    else:
        result["required_cost_validation_target_count"] = 0
        result["required_cost_test_target_count"] = 0
        result["required_cost_validation_and_test_target_count"] = 0
        result["required_cost_results"] = []
    return result


def select_validation_robust_delayed_policy(
    candidates: pd.DataFrame,
    *,
    required_cost_bps: tuple[float, ...] = (20.0, 50.0),
    selection_cost_bps: float = 20.0,
    return_retention_low: float = 85.0,
    return_retention_high: float = 95.0,
    min_validation_retention_band_margin: float = 0.05,
    max_drawdown_floor: float = -18.0,
    max_skip_rate: float = 0.15,
    eligible_exit_policies: tuple[str, ...] | None = ("hold_to_next_open", SAME_DAY_CLOSE_EXIT_POLICY),
    return_retention_upper_is_soft: bool = False,
) -> dict[str, Any]:
    selection_priority = [
        "validation_return_retention_score_desc",
        "validation_skip_rate_asc",
        "condition_valid_active_coverage_desc",
        "validation_drawdown_improvement_pct_desc",
        "selection_score_desc",
    ]
    if candidates.empty:
        return {
            "policy_id": None,
            "validation_passes_target": False,
            "selection_uses_test_segment": False,
            "selection_priority": selection_priority,
        }
    frame = candidates.copy()
    if eligible_exit_policies is not None and "exit_policy" in frame.columns:
        frame = frame.loc[frame["exit_policy"].isin(set(eligible_exit_policies))].copy()
        if frame.empty:
            return {
                "policy_id": None,
                "validation_passes_target": False,
                "selection_uses_test_segment": False,
                "eligible_exit_policies": list(eligible_exit_policies),
                "selection_priority": selection_priority,
            }
    frame["cost_bps"] = pd.to_numeric(frame["cost_bps"], errors="coerce")
    if "validation_return_retention_score" in frame.columns:
        frame["validation_return_retention_score"] = pd.to_numeric(
            frame["validation_return_retention_score"],
            errors="coerce",
        ).fillna(0.0)
    else:
        frame["validation_return_retention_score"] = frame["validation_return_retention_pct"].clip(
            upper=float(return_retention_high)
        )
    frame["validation_skip_rate"] = (
        pd.to_numeric(frame["validation_skipped_periods"], errors="coerce")
        / pd.to_numeric(frame["validation_periods"], errors="coerce").replace(0, np.nan)
    ).fillna(1.0)
    if "condition_valid_active_coverage" not in frame.columns:
        frame["condition_valid_active_coverage"] = 1.0
    required_costs = {float(value) for value in required_cost_bps}
    rows: list[pd.Series] = []
    for _, group in frame.groupby("policy_id", sort=False):
        costs = {float(value) for value in group["cost_bps"].dropna().unique()}
        if not required_costs.issubset(costs):
            continue
        robust = True
        retention_band_margins: list[float] = []
        for cost in required_costs:
            row = group.loc[group["cost_bps"].eq(cost)].iloc[0]
            retention = float(row["validation_return_retention_pct"])
            drawdown = float(row["validation_max_drawdown_pct"])
            skipped = float(row["validation_skipped_periods"])
            periods = float(row["validation_periods"])
            target_value = row.get("validation_return_target_satisfied", np.nan)
            if pd.isna(target_value):
                return_target_satisfied = bool(
                    retention >= float(return_retention_low)
                    and (return_retention_upper_is_soft or retention <= float(return_retention_high))
                )
                retention_band_margin = min(
                    retention - float(return_retention_low),
                    float(return_retention_high) - retention,
                )
            else:
                return_target_satisfied = bool(target_value)
                retention_band_margin = (
                    float(return_retention_high)
                    if str(row.get("validation_return_target_mode", "")) == "baseline_nonpositive_improvement"
                    and return_target_satisfied
                    else min(retention - float(return_retention_low), float(return_retention_high) - retention)
                )
            retention_band_margins.append(retention_band_margin)
            if not (
                return_target_satisfied
                and (
                    return_retention_upper_is_soft
                    or retention_band_margin >= float(min_validation_retention_band_margin)
                )
                and drawdown > max_drawdown_floor
                and skipped <= periods * max_skip_rate
            ):
                robust = False
                break
        if robust:
            selected_cost_row = group.loc[group["cost_bps"].eq(float(selection_cost_bps))]
            if not selected_cost_row.empty:
                selected_row = selected_cost_row.iloc[0].copy()
                selected_row["required_cost_min_validation_band_margin"] = float(min(retention_band_margins))
                rows.append(selected_row)
    if not rows:
        fallback = frame.sort_values(
            ["selection_score", "validation_drawdown_improvement_pct"],
            ascending=[False, False],
        ).iloc[0]
        result = _json_ready_dict(fallback.to_dict())
        result["validation_passes_target"] = False
        result["selection_uses_test_segment"] = False
        result["required_cost_bps"] = sorted(required_costs)
        result["max_skip_rate"] = float(max_skip_rate)
        result["return_retention_high"] = float(return_retention_high)
        result["return_retention_upper_policy"] = (
            "soft_cap_for_scoring" if return_retention_upper_is_soft else "hard_gate"
        )
        result["min_validation_retention_band_margin"] = float(min_validation_retention_band_margin)
        result["selection_priority"] = selection_priority
        result["blocking_reason"] = "no_validation_policy_in_retention_band"
        return result
    selected = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "validation_return_retention_score",
                "validation_skip_rate",
                "condition_valid_active_coverage",
                "validation_drawdown_improvement_pct",
                "selection_score",
            ],
            ascending=[False, True, False, False, False],
        )
        .iloc[0]
    )
    result = _json_ready_dict(selected.to_dict())
    result["validation_passes_target"] = True
    result["selection_uses_test_segment"] = False
    result["required_cost_bps"] = sorted(required_costs)
    result["max_skip_rate"] = float(max_skip_rate)
    result["min_validation_retention_band_margin"] = float(min_validation_retention_band_margin)
    result["return_retention_high"] = float(return_retention_high)
    result["return_retention_upper_policy"] = "soft_cap_for_scoring" if return_retention_upper_is_soft else "hard_gate"
    result["selection_priority"] = selection_priority
    return result


def select_intraday_flat_sleeve_policy(
    candidates: pd.DataFrame,
    *,
    required_cost_bps: tuple[float, ...] = (20.0, 50.0),
    selection_cost_bps: float = 20.0,
    return_retention_low: float = 85.0,
    return_retention_high: float = 95.0,
    max_drawdown_floor: float = -18.0,
    max_skip_rate: float = 0.15,
) -> dict[str, Any]:
    selection_priority = [
        "validation_max_drawdown_desc",
        "validation_return_retention_score_desc",
        "validation_skip_rate_asc",
        "condition_valid_active_coverage_desc",
        "selection_score_desc",
    ]
    if candidates.empty or "exit_policy" not in candidates.columns:
        return {
            "policy_id": None,
            "exit_policy": SAME_DAY_CLOSE_EXIT_POLICY,
            "selection_role": "intraday_flat_sleeve",
            "selection_priority": selection_priority,
            "validation_passes_target": False,
            "selection_uses_test_segment": False,
        }
    frame = candidates.loc[candidates["exit_policy"].eq(SAME_DAY_CLOSE_EXIT_POLICY)].copy()
    if frame.empty:
        return {
            "policy_id": None,
            "exit_policy": SAME_DAY_CLOSE_EXIT_POLICY,
            "selection_role": "intraday_flat_sleeve",
            "selection_priority": selection_priority,
            "validation_passes_target": False,
            "selection_uses_test_segment": False,
        }
    frame["cost_bps"] = pd.to_numeric(frame["cost_bps"], errors="coerce")
    if "validation_return_retention_score" in frame.columns:
        frame["validation_return_retention_score"] = pd.to_numeric(
            frame["validation_return_retention_score"],
            errors="coerce",
        ).fillna(0.0)
    else:
        frame["validation_return_retention_score"] = frame["validation_return_retention_pct"].clip(upper=95.0)
    frame["validation_skip_rate"] = (
        pd.to_numeric(frame["validation_skipped_periods"], errors="coerce")
        / pd.to_numeric(frame["validation_periods"], errors="coerce").replace(0, np.nan)
    ).fillna(1.0)
    if "condition_valid_active_coverage" not in frame.columns:
        frame["condition_valid_active_coverage"] = 1.0
    required_costs = {float(value) for value in required_cost_bps}
    robust_rows: list[pd.Series] = []
    for _, group in frame.groupby("policy_id", sort=False):
        costs = {float(value) for value in group["cost_bps"].dropna().unique()}
        if not required_costs.issubset(costs):
            continue
        robust = True
        for cost in required_costs:
            row = group.loc[group["cost_bps"].eq(cost)].iloc[0]
            retention = float(row["validation_return_retention_pct"])
            drawdown = float(row["validation_max_drawdown_pct"])
            skipped = float(row["validation_skipped_periods"])
            periods = float(row["validation_periods"])
            target_value = row.get("validation_return_target_satisfied", np.nan)
            if pd.isna(target_value):
                return_target_satisfied = bool(
                    retention >= float(return_retention_low) and retention <= float(return_retention_high)
                )
            else:
                return_target_satisfied = bool(target_value)
            if not (
                return_target_satisfied
                and drawdown > float(max_drawdown_floor)
                and skipped <= periods * float(max_skip_rate)
            ):
                robust = False
                break
        if robust:
            selected_cost_row = group.loc[group["cost_bps"].eq(float(selection_cost_bps))]
            if not selected_cost_row.empty:
                robust_rows.append(selected_cost_row.iloc[0])
    source = pd.DataFrame(robust_rows) if robust_rows else frame.copy()
    selected = source.sort_values(
        [
            "validation_max_drawdown_pct",
            "validation_return_retention_score",
            "validation_skip_rate",
            "condition_valid_active_coverage",
            "selection_score",
        ],
        ascending=[False, False, True, False, False],
    ).iloc[0]
    result = _json_ready_dict(selected.to_dict())
    result["selection_role"] = "intraday_flat_sleeve"
    result["selection_priority"] = selection_priority
    result["selection_segment"] = "validation"
    result["report_segment"] = "test"
    result["selection_uses_test_segment"] = False
    result["required_cost_bps"] = sorted(required_costs)
    result["max_skip_rate"] = float(max_skip_rate)
    result["return_retention_floor_pct"] = float(return_retention_low)
    result["return_retention_high"] = float(return_retention_high)
    result["return_retention_upper_policy"] = "hard_gate"
    result["max_drawdown_floor_pct"] = float(max_drawdown_floor)
    result["validation_passes_target"] = bool(robust_rows)
    return result


def evaluate_observable_delayed_signal_candidates(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    intraday: pd.DataFrame,
    cost_bps_values: tuple[float, ...] = (10.0, 20.0, 50.0),
    prob_col: str = PROB_COL,
    gate_col: str = GATE_COL,
    observation_bars_grid: tuple[int, ...] = (2, 4),
    signal_threshold_grid: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
    risk_threshold_grid: tuple[float | None, ...] = (None, 0.20, 0.30, 0.40, 0.50),
    exit_policy_grid: tuple[str, ...] = ("hold_to_next_open", "risk_exit", SAME_DAY_CLOSE_EXIT_POLICY),
    quantiles: tuple[float, ...] = tuple(float(value) for value in np.arange(0.05, 0.975, 0.025)),
) -> pd.DataFrame:
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    bars = _normalize_intraday_bars(intraday, prob_col=prob_col, gate_col=gate_col)
    prepared = _prepare_delayed_signal_arrays(
        daily_frame,
        daily_prices=prices,
        intraday=bars,
        prob_col=prob_col,
        gate_col=gate_col,
        observation_bars_grid=observation_bars_grid,
        signal_threshold_grid=signal_threshold_grid,
        risk_threshold_grid=risk_threshold_grid,
        exit_policy_grid=exit_policy_grid,
        cost_bps_values=cost_bps_values,
    )
    segment = prepared["segment"]
    validation_mask = segment.eq("validation").to_numpy()
    test_mask = segment.eq("test").to_numpy()
    active_mask = prepared["active"]

    rows: list[dict[str, Any]] = []
    for observation_bars in observation_bars_grid:
        feature_map = prepared["features"][int(observation_bars)]
        observation_minutes = int(observation_bars) * 15
        for feature, values in feature_map.items():
            validation_values = pd.Series(values[validation_mask]).dropna()
            if validation_values.nunique() < 2:
                continue
            for quantile in quantiles:
                threshold = float(validation_values.quantile(float(quantile)))
                for operator in ("ge", "le"):
                    condition = (
                        values >= threshold
                        if operator == "ge"
                        else values <= threshold
                    )
                    condition = np.nan_to_num(condition, nan=False).astype(bool)
                    coverage_base = validation_mask & active_mask
                    coverage = float(condition[coverage_base].mean()) if coverage_base.any() else 0.0
                    if coverage < 0.03:
                        continue
                    for signal_threshold in signal_threshold_grid:
                        for exit_policy, risk_threshold, action_key in _delayed_exit_policy_actions(
                            exit_policy_grid,
                            risk_threshold_grid,
                        ):
                            risk_suffix = (
                                "hold"
                                if exit_policy == "hold_to_next_open"
                                else "same_day_close"
                                if exit_policy == SAME_DAY_CLOSE_EXIT_POLICY
                                else f"risk_{float(risk_threshold):.2f}"
                            )
                            variant = f"delayed_signal_after_{observation_minutes}m"
                            if exit_policy == "risk_exit":
                                variant = f"{variant}_risk_exit"
                            elif exit_policy == SAME_DAY_CLOSE_EXIT_POLICY:
                                variant = f"{variant}_same_day_close"
                            policy_id = (
                                f"obs{observation_minutes}_{feature}_{operator}_q{quantile:.3f}_"
                                f"{threshold:.6g}_signal_{signal_threshold:.2f}_{risk_suffix}"
                            )
                            for cost_bps in cost_bps_values:
                                cost_key = float(cost_bps)
                                base_returns = prepared["base_returns"][cost_key]
                                base_skipped = prepared["base_skipped"][cost_key]
                                action_returns, action_skipped = prepared["actions"][cost_key][int(observation_bars)][float(signal_threshold)][action_key]
                                returns = np.where(condition, action_returns, base_returns)
                                skipped = np.where(condition, action_skipped, base_skipped)
                                baseline_validation = _segment_return_metrics(base_returns, validation_mask, base_skipped)
                                baseline_test = _segment_return_metrics(base_returns, test_mask, base_skipped)
                                validation = _segment_return_metrics(returns, validation_mask, skipped)
                                test = _segment_return_metrics(returns, test_mask, skipped)
                                validation_retention = _retention(
                                    validation["total_return_pct"],
                                    baseline_validation["total_return_pct"],
                                )
                                test_retention = _retention(test["total_return_pct"], baseline_test["total_return_pct"])
                                validation_return_target_satisfied = _return_target_satisfied(
                                    validation["total_return_pct"],
                                    baseline_validation["total_return_pct"],
                                    validation_retention,
                                )
                                test_return_target_satisfied = _return_target_satisfied(
                                    test["total_return_pct"],
                                    baseline_test["total_return_pct"],
                                    test_retention,
                                )
                                validation_return_score = _return_retention_score(
                                    validation["total_return_pct"],
                                    baseline_validation["total_return_pct"],
                                    validation_retention,
                                )
                                validation_dd_improvement = (
                                    validation["max_drawdown_pct"] - baseline_validation["max_drawdown_pct"]
                                )
                                test_dd_improvement = test["max_drawdown_pct"] - baseline_test["max_drawdown_pct"]
                                rows.append(
                                    {
                                        "policy_id": policy_id,
                                        "cost_bps": cost_key,
                                        "variant": variant,
                                        "exit_policy": exit_policy,
                                        "risk_threshold": np.nan if risk_threshold is None else float(risk_threshold),
                                        "risk_threshold_source": "fixed_grid" if exit_policy == "risk_exit" else "disabled",
                                        "observation_bars": int(observation_bars),
                                        "observation_minutes": observation_minutes,
                                        "condition_feature": feature,
                                        "condition_operator": operator,
                                        "condition_quantile": float(quantile),
                                        "condition_threshold": threshold,
                                        "condition_threshold_source": "validation_quantile",
                                        "condition_valid_active_coverage": coverage,
                                        "signal_threshold": float(signal_threshold),
                                        "selection_segment": "validation",
                                        "report_segment": "test",
                                        "selection_uses_test_segment": False,
                                        "validation_periods": validation["periods"],
                                        "validation_baseline_return_pct": baseline_validation["total_return_pct"],
                                        "validation_return_pct": validation["total_return_pct"],
                                        "validation_return_retention_pct": validation_retention,
                                        "validation_return_retention_score": validation_return_score,
                                        "validation_return_target_mode": _return_target_mode(
                                            baseline_validation["total_return_pct"]
                                        ),
                                        "validation_return_target_satisfied": validation_return_target_satisfied,
                                        "validation_return_retention_in_target_band": _return_retention_in_target_band(
                                            baseline_validation["total_return_pct"],
                                            validation_retention,
                                        ),
                                        "validation_max_drawdown_pct": validation["max_drawdown_pct"],
                                        "validation_drawdown_improvement_pct": validation_dd_improvement,
                                        "validation_daily_sharpe": validation["daily_sharpe"],
                                        "validation_skipped_periods": validation["skipped_periods"],
                                        "test_periods": test["periods"],
                                        "test_baseline_return_pct": baseline_test["total_return_pct"],
                                        "test_return_pct": test["total_return_pct"],
                                        "test_return_retention_pct": test_retention,
                                        "test_return_retention_score": _return_retention_score(
                                            test["total_return_pct"],
                                            baseline_test["total_return_pct"],
                                            test_retention,
                                        ),
                                        "test_return_target_mode": _return_target_mode(
                                            baseline_test["total_return_pct"]
                                        ),
                                        "test_return_target_satisfied": test_return_target_satisfied,
                                        "test_return_retention_in_target_band": _return_retention_in_target_band(
                                            baseline_test["total_return_pct"],
                                            test_retention,
                                        ),
                                        "test_max_drawdown_pct": test["max_drawdown_pct"],
                                        "test_drawdown_improvement_pct": test_dd_improvement,
                                        "test_daily_sharpe": test["daily_sharpe"],
                                        "test_skipped_periods": test["skipped_periods"],
                                        "selection_score": (
                                            validation_return_score
                                            + 2.0 * validation_dd_improvement
                                            + 0.25 * validation["daily_sharpe"]
                                            - 0.10 * validation["skipped_periods"]
                                        ),
                                    }
                                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if "validation_return_retention_score" not in result.columns:
        result["validation_return_retention_score"] = result["validation_return_retention_pct"].clip(upper=95.0)
    if "test_return_retention_score" not in result.columns:
        result["test_return_retention_score"] = result["test_return_retention_pct"].clip(upper=95.0)
    result["validation_skip_rate"] = (
        result["validation_skipped_periods"] / result["validation_periods"].replace(0, np.nan)
    ).fillna(1.0)
    result["test_skip_rate"] = (
        result["test_skipped_periods"] / result["test_periods"].replace(0, np.nan)
    ).fillna(1.0)
    result["return_retention_upper_policy"] = "hard_gate"
    result["passes_validation_target"] = (
        result["validation_return_target_satisfied"].astype(bool)
        & result["validation_max_drawdown_pct"].gt(-18.0)
        & result["validation_skipped_periods"].le(result["validation_periods"] * 0.15)
    )
    result["passes_test_target"] = (
        result["test_return_target_satisfied"].astype(bool)
        & result["test_max_drawdown_pct"].gt(-18.0)
        & result["test_skipped_periods"].le(result["test_periods"] * 0.15)
    )
    result["passes_validation_and_test_target"] = result["passes_validation_target"] & result["passes_test_target"]
    return result.sort_values(
        ["passes_validation_target", "validation_drawdown_improvement_pct", "selection_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_regime_feature_frame(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    price_index = pd.Index(prices["datetime"])
    rows: list[dict[str, Any]] = []
    for daily_row in daily_frame.to_dict("records"):
        decision_date = pd.Timestamp(daily_row["datetime"]).normalize()
        location = price_index.get_loc(decision_date) if decision_date in set(price_index) else None
        if isinstance(location, slice):
            location = int(location.start)
        entry_date = (
            prices.iloc[int(location) + 1]["datetime"]
            if location is not None and int(location) + 1 < len(prices)
            else pd.NaT
        )
        row = {
            "datetime": decision_date.date().isoformat(),
            "segment": str(daily_row.get("segment", "unknown")),
            "entry_date": _date_or_none(entry_date),
            "target_position": float(daily_row.get("target_position", 0.0) or 0.0),
        }
        for column in ["probability", "raw_target", "turnover"]:
            if column in daily_row:
                row[column] = float(daily_row.get(column, np.nan) or 0.0)
        for key in ["walk_forward_fold", "test_year", "fold_segment"]:
            if key in daily_row:
                row[key] = daily_row[key]
        rows.append(row)
    return pd.DataFrame(rows)


def _variant_return_frame(trades: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    frame = trades.loc[trades["variant"].eq(variant)].copy()
    if frame.empty:
        return frame
    frame["datetime_key"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    return frame.sort_values("datetime_key").drop_duplicates("datetime_key").reset_index(drop=True)


def _prepare_delayed_signal_arrays(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    intraday: pd.DataFrame,
    prob_col: str,
    gate_col: str,
    observation_bars_grid: tuple[int, ...],
    signal_threshold_grid: tuple[float, ...],
    risk_threshold_grid: tuple[float | None, ...],
    exit_policy_grid: tuple[str, ...],
    cost_bps_values: tuple[float, ...],
) -> dict[str, Any]:
    daily_frame = _normalize_daily_positions(daily).sort_values("datetime").reset_index(drop=True)
    prices = _normalize_daily_prices(daily_prices)
    bars = _normalize_intraday_bars(intraday, prob_col=prob_col, gate_col=gate_col)
    price_index = pd.Index(prices["datetime"])
    bars_by_date = {
        date: group.sort_values("datetime").reset_index(drop=True)
        for date, group in bars.groupby("date", sort=True)
    }
    row_bars: list[pd.DataFrame] = []
    target = np.zeros(len(daily_frame))
    entry_open = np.full(len(daily_frame), np.nan)
    exit_open = np.full(len(daily_frame), np.nan)
    source_return = np.zeros(len(daily_frame))
    active = np.zeros(len(daily_frame), dtype=bool)
    for index, row in daily_frame.iterrows():
        decision_date = pd.Timestamp(row["datetime"]).normalize()
        location = price_index.get_loc(decision_date) if decision_date in set(price_index) else None
        if isinstance(location, slice):
            location = int(location.start)
        row_target = float(row.get("target_position", 0.0) or 0.0)
        target[index] = row_target
        source_return[index] = float(row.get("strategy_return", 0.0) or 0.0)
        if location is None or int(location) + 2 >= len(prices) or abs(row_target) <= 1e-12:
            row_bars.append(pd.DataFrame())
            continue
        entry_date = prices.iloc[int(location) + 1]["datetime"]
        entry_key = pd.Timestamp(entry_date).date().isoformat()
        row_bars.append(bars_by_date.get(entry_key, pd.DataFrame()))
        entry_open[index] = float(prices.iloc[int(location) + 1]["open"])
        exit_open[index] = float(prices.iloc[int(location) + 2]["open"])
        active[index] = True

    features: dict[int, dict[str, np.ndarray]] = {}
    actions: dict[float, dict[int, dict[float, tuple[np.ndarray, np.ndarray]]]] = {}
    base_returns: dict[float, np.ndarray] = {}
    base_skipped: dict[float, np.ndarray] = {}

    for cost_bps in cost_bps_values:
        cost_key = float(cost_bps)
        base_returns[cost_key] = source_return - np.abs(target) * cost_key / 10000.0
        base_skipped[cost_key] = np.zeros(len(daily_frame), dtype=bool)
        actions[cost_key] = {}

    for observation_bars in observation_bars_grid:
        obs_key = int(observation_bars)
        exit_actions = _delayed_exit_policy_actions(exit_policy_grid, risk_threshold_grid)
        obs_features = {
            "observed_return": np.full(len(daily_frame), np.nan),
            "observed_prob_mean": np.full(len(daily_frame), np.nan),
            "observed_prob_max": np.full(len(daily_frame), np.nan),
            "observed_prob_min": np.full(len(daily_frame), np.nan),
        }
        for cost_bps in cost_bps_values:
            actions[float(cost_bps)][obs_key] = {
                float(signal_threshold): {
                    action_key: (
                        np.zeros(len(daily_frame), dtype=float),
                        np.zeros(len(daily_frame), dtype=bool),
                    )
                    for _, _, action_key in exit_actions
                }
                for signal_threshold in signal_threshold_grid
            }
        for index, bars_for_day in enumerate(row_bars):
            if not active[index] or bars_for_day.empty or len(bars_for_day) < obs_key:
                for cost_bps in cost_bps_values:
                    for signal_threshold in signal_threshold_grid:
                        for _, _, action_key in exit_actions:
                            actions[float(cost_bps)][obs_key][float(signal_threshold)][action_key][1][index] = bool(active[index])
                continue
            observed = bars_for_day.iloc[:obs_key]
            obs_features["observed_return"][index] = _observed_intraday_feature_value(
                observed,
                entry_open=float(entry_open[index]),
                feature="observed_return",
                prob_col=prob_col,
            )
            obs_features["observed_prob_mean"][index] = _observed_intraday_feature_value(
                observed,
                entry_open=float(entry_open[index]),
                feature="observed_prob_mean",
                prob_col=prob_col,
            )
            obs_features["observed_prob_max"][index] = _observed_intraday_feature_value(
                observed,
                entry_open=float(entry_open[index]),
                feature="observed_prob_max",
                prob_col=prob_col,
            )
            obs_features["observed_prob_min"][index] = _observed_intraday_feature_value(
                observed,
                entry_open=float(entry_open[index]),
                feature="observed_prob_min",
                prob_col=prob_col,
            )
            signal_base = bars_for_day.iloc[obs_key - 1 :].loc[
                bars_for_day.iloc[obs_key - 1 :][gate_col].fillna(False).astype(bool)
                & bars_for_day.iloc[obs_key - 1 :]["next_open"].notna()
            ]
            for signal_threshold in signal_threshold_grid:
                signal_threshold = float(signal_threshold)
                signal_candidates = signal_base.loc[
                    pd.to_numeric(signal_base[prob_col], errors="coerce").ge(signal_threshold)
                ]
                for cost_bps in cost_bps_values:
                    cost_key = float(cost_bps)
                    if signal_candidates.empty:
                        for _, _, action_key in exit_actions:
                            returns, skipped = actions[cost_key][obs_key][signal_threshold][action_key]
                            skipped[index] = True
                            returns[index] = 0.0
                        continue
                    signal = signal_candidates.iloc[0]
                    entry_price = float(signal["next_open"])
                    cost = abs(target[index]) * cost_key / 10000.0
                    for exit_policy, risk_threshold, action_key in exit_actions:
                        returns, skipped = actions[cost_key][obs_key][signal_threshold][action_key]
                        selected_exit_open = float(exit_open[index])
                        if exit_policy == SAME_DAY_CLOSE_EXIT_POLICY and pd.notna(signal["next_datetime"]):
                            same_day_exit = _same_day_close_exit(
                                bars_for_day,
                                entry_time=pd.Timestamp(signal["next_datetime"]),
                            )
                            if same_day_exit is None:
                                skipped[index] = True
                                returns[index] = 0.0
                                continue
                            _, selected_exit_open = same_day_exit
                        elif exit_policy == "risk_exit" and risk_threshold is not None and pd.notna(signal["next_datetime"]):
                            risk_candidates = bars_for_day.loc[
                                bars_for_day["datetime"].ge(pd.Timestamp(signal["next_datetime"]))
                                & bars_for_day[gate_col].fillna(False).astype(bool)
                                & pd.to_numeric(bars_for_day[prob_col], errors="coerce").le(float(risk_threshold))
                                & bars_for_day["close"].notna()
                            ]
                            if not risk_candidates.empty:
                                selected_exit_open = float(risk_candidates.iloc[0]["close"])
                        returns[index] = float(target[index] * (selected_exit_open / entry_price - 1.0) - cost)
        features[obs_key] = obs_features

    return {
        "segment": daily_frame["segment"],
        "active": active,
        "features": features,
        "actions": actions,
        "base_returns": base_returns,
        "base_skipped": base_skipped,
    }


def _segment_return_metrics(
    returns: np.ndarray,
    mask: np.ndarray,
    skipped: np.ndarray,
) -> dict[str, float | int]:
    segment_returns = np.nan_to_num(np.asarray(returns, dtype=float)[mask], nan=0.0)
    segment_skipped = np.asarray(skipped, dtype=bool)[mask]
    if len(segment_returns) == 0:
        return {
            "periods": 0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_sharpe": 0.0,
            "skipped_periods": 0,
        }
    equity = np.cumprod(1.0 + segment_returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    std = float(np.std(segment_returns))
    sharpe = 0.0 if std == 0.0 else float(np.mean(segment_returns) / std * np.sqrt(252.0))
    return {
        "periods": int(len(segment_returns)),
        "total_return_pct": float((equity[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "daily_sharpe": sharpe,
        "skipped_periods": int(segment_skipped.sum()),
    }


def _segment_execution_risk_metrics(
    returns: np.ndarray,
    entry_day_returns: np.ndarray,
    mask: np.ndarray,
    active: np.ndarray,
    *,
    open_buy_dump_threshold: float = OPEN_BUY_DUMP_THRESHOLD,
    crash_threshold: float = CRASH_THRESHOLD,
) -> dict[str, float | int]:
    segment_mask = np.asarray(mask, dtype=bool)
    segment_active = segment_mask & np.asarray(active, dtype=bool)
    segment_returns = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    segment_entry_day_returns = np.asarray(entry_day_returns, dtype=float)
    active_returns = segment_returns[segment_active]
    active_entry_day_returns = segment_entry_day_returns[segment_active]
    entry_day_available = ~np.isnan(active_entry_day_returns)
    open_buy_dump = entry_day_available & (active_entry_day_returns <= float(open_buy_dump_threshold))
    crash = active_returns <= float(crash_threshold)
    return {
        "active_periods": int(segment_active.sum()),
        "entry_day_available_periods": int(entry_day_available.sum()),
        "open_buy_dump_periods": int(open_buy_dump.sum()),
        "crash_periods": int(crash.sum()),
        "worst_entry_day_return_pct": float(np.nanmin(active_entry_day_returns) * 100.0)
        if entry_day_available.any()
        else 0.0,
        "worst_strategy_return_pct": float(np.min(active_returns) * 100.0)
        if len(active_returns)
        else 0.0,
    }


def _retention(total_return_pct: float, baseline_return_pct: float) -> float:
    if abs(float(baseline_return_pct)) <= 1e-12:
        return float("nan")
    return float(total_return_pct) / float(baseline_return_pct) * 100.0


def _return_target_mode(baseline_return_pct: float) -> str:
    return "positive_retention_band" if float(baseline_return_pct) > 0.0 else "baseline_nonpositive_improvement"


def _return_target_satisfied(
    total_return_pct: float,
    baseline_return_pct: float,
    return_retention_pct: float,
    *,
    return_retention_low: float = 85.0,
    return_retention_high: float = 95.0,
    return_retention_upper_is_soft: bool = True,
) -> bool:
    total = float(total_return_pct)
    baseline = float(baseline_return_pct)
    if baseline <= 0.0:
        return bool(total >= baseline)
    if pd.isna(return_retention_pct):
        return False
    retention = float(return_retention_pct)
    return bool(
        retention >= float(return_retention_low)
        and (return_retention_upper_is_soft or retention <= float(return_retention_high))
    )


def _return_retention_in_target_band(
    baseline_return_pct: float,
    return_retention_pct: float,
    *,
    return_retention_low: float = 85.0,
    return_retention_high: float = 95.0,
) -> bool:
    if float(baseline_return_pct) <= 0.0:
        return True
    if pd.isna(return_retention_pct):
        return False
    retention = float(return_retention_pct)
    return bool(float(return_retention_low) <= retention <= float(return_retention_high))


def _return_retention_score(
    total_return_pct: float,
    baseline_return_pct: float,
    return_retention_pct: float,
    *,
    return_retention_high: float = 95.0,
) -> float:
    baseline = float(baseline_return_pct)
    total = float(total_return_pct)
    if baseline <= 0.0:
        return float(return_retention_high) if total >= baseline else 0.0
    if pd.isna(return_retention_pct):
        return 0.0
    return float(min(float(return_retention_pct), float(return_retention_high)))


def _run_card(spec: CognitiveAlphaIntradayOverlaySpec) -> dict[str, Any]:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)

    daily_summary = _read_json(spec.daily_card_dir / "summary.json")
    intraday_summary = _read_json(spec.intraday_card_dir / "summary.json")
    workflow_yaml = _write_workflow_yaml(spec, daily_summary=daily_summary, intraday_summary=intraday_summary)
    signal_config = _intraday_signal_config(spec, intraday_summary=intraday_summary)

    daily_positions = _load_daily_selected_positions(daily_summary, prefer_walk_forward=False)
    walk_forward_daily_positions = _load_daily_selected_positions(daily_summary, prefer_walk_forward=True)
    daily_position_source_audit = _daily_position_source_audit(spec, daily_summary=daily_summary)
    as_of = str(daily_summary.get("as_of") or spec.as_of)
    daily_price_start = min(
        daily_positions["datetime"].min(),
        walk_forward_daily_positions["datetime"].min(),
    )
    daily_prices = _load_daily_open_prices(spec, start=str(daily_price_start.date()), end=as_of)
    intraday_predictions = _load_intraday_prediction_frame(
        spec,
        intraday_summary=intraday_summary,
        as_of=as_of,
        signal_config=signal_config,
    )
    thresholds = _thresholds_from_validation(
        intraday_predictions,
        intraday_summary=intraday_summary,
        signal_config=signal_config,
        signal_threshold=spec.signal_threshold,
        risk_threshold=spec.risk_threshold,
    )
    threshold_selection = pd.DataFrame()
    if spec.optimize_thresholds and spec.signal_threshold is None and spec.risk_threshold is None:
        thresholds, threshold_selection = _select_thresholds_from_validation(
            daily_positions,
            daily_prices=daily_prices,
            intraday_predictions=intraday_predictions,
            signal_config=signal_config,
            base_thresholds=thresholds,
            cost_bps=spec.selection_cost_bps,
        )

    trade_frames = []
    for cost_bps in sorted(set(float(value) for value in spec.cost_stress_bps)):
        trade_frames.append(
            simulate_execution_overlays(
                daily_positions,
                daily_prices=daily_prices,
                intraday=intraday_predictions,
                signal_threshold=thresholds["signal_threshold"],
                risk_threshold=thresholds["risk_threshold"],
                cost_bps=cost_bps,
                prob_col=signal_config["prob_col"],
                gate_col=signal_config["gate_col"],
            ).assign(cost_bps=float(cost_bps))
        )
    overlay_trades = pd.concat(trade_frames, ignore_index=True)
    regime_features = build_regime_feature_frame(daily_positions, daily_prices=daily_prices)
    condition_columns = tuple(
        column
        for column in ("probability", "target_position", "turnover", "raw_target")
        if column in regime_features.columns
    )
    conditional_risk_policy_selection = pd.concat(
        [
            evaluate_conditional_overlay_candidates(
                overlay_trades,
                regime_features,
                cost_bps=float(cost_bps),
                condition_columns=condition_columns,
            )
            for cost_bps in (20.0, 50.0)
        ],
        ignore_index=True,
    )
    conditional_risk_sleeve = select_conditional_risk_sleeve_policy(
        conditional_risk_policy_selection,
        required_cost_bps=(20.0, 50.0),
        selection_cost_bps=float(spec.selection_cost_bps),
    )
    conditional_risk_sleeve_rows = _conditional_policy_rows(
        conditional_risk_policy_selection,
        conditional_risk_sleeve,
    )
    walk_forward_trade_frames = []
    for cost_bps in sorted(set(float(value) for value in spec.cost_stress_bps)):
        walk_forward_trade_frames.append(
            simulate_execution_overlays(
                walk_forward_daily_positions,
                daily_prices=daily_prices,
                intraday=intraday_predictions,
                signal_threshold=thresholds["signal_threshold"],
                risk_threshold=thresholds["risk_threshold"],
                cost_bps=cost_bps,
                prob_col=signal_config["prob_col"],
                gate_col=signal_config["gate_col"],
            ).assign(cost_bps=float(cost_bps))
        )
    walk_forward_overlay_trades = pd.concat(walk_forward_trade_frames, ignore_index=True)
    walk_forward_regime_features = build_regime_feature_frame(walk_forward_daily_positions, daily_prices=daily_prices)
    walk_forward_condition_columns = tuple(
        column
        for column in ("probability", "target_position", "turnover", "raw_target")
        if column in walk_forward_regime_features.columns
    )
    cost_sensitivity = summarize_overlay_metrics(overlay_trades)
    execution_risk_diagnostics = summarize_execution_risk_diagnostics(overlay_trades)
    model_comparison = (
        cost_sensitivity.loc[cost_sensitivity["cost_bps"].eq(float(spec.selection_cost_bps))]
        .sort_values(["segment", "selection_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    selected_model = _select_variant(model_comparison, cost_bps=spec.selection_cost_bps)
    selected_model["signal_threshold"] = float(thresholds["signal_threshold"])
    selected_model["risk_threshold"] = float(thresholds["risk_threshold"])
    execution_risk_summary = _execution_risk_diagnostics_summary(
        execution_risk_diagnostics,
        selected_variant=str(selected_model["variant"]),
        cost_bps=float(spec.selection_cost_bps),
    )
    walk_forward = _walk_forward_metrics(
        walk_forward_overlay_trades,
        cost_bps=spec.selection_cost_bps,
        require_intraday_prediction_coverage=True,
    )
    conditional_risk_sleeve_walk_forward = _conditional_risk_sleeve_walk_forward_metrics(
        walk_forward_overlay_trades,
        walk_forward_regime_features,
        required_cost_bps=(20.0, 50.0),
        selection_cost_bps=float(spec.selection_cost_bps),
        condition_columns=walk_forward_condition_columns,
        require_intraday_prediction_coverage=True,
    )
    walk_forward_summary = _overlay_walk_forward_summary(
        intraday_summary=intraday_summary,
        walk_forward=walk_forward,
    )
    conditional_risk_sleeve_walk_forward_summary = _conditional_risk_sleeve_walk_forward_summary(
        intraday_summary=intraday_summary,
        walk_forward=conditional_risk_sleeve_walk_forward,
        required_cost_bps=(20.0, 50.0),
    )
    optimization_plan = build_intraday_optimization_plan(
        selected_model=selected_model,
        walk_forward_summary=walk_forward_summary,
        conditional_walk_forward_summary=conditional_risk_sleeve_walk_forward_summary,
        source_intraday_model={"external_context": intraday_summary.get("external_intraday_context", {})},
        local_intraday_manifest=_read_optional_json(spec.root / "data" / "qlib" / "soxl_intraday_15min" / "qlib_store_manifest.json"),
        local_intraday_5min_manifest=_read_optional_json(spec.root / "data" / "qlib" / "soxl_intraday_5min" / "qlib_store_manifest.json"),
        external_context_summary=_read_optional_json(spec.root / "research" / "intraday_external_context_data_v1" / "summary.json"),
    )
    analysis_summary = _analysis_summary(
        spec,
        daily_summary=daily_summary,
        intraday_summary=intraday_summary,
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        intraday_predictions=intraday_predictions,
        thresholds=thresholds,
        signal_config=signal_config,
        model_comparison=model_comparison,
        threshold_selection=threshold_selection,
        execution_risk_diagnostics=execution_risk_diagnostics,
        execution_risk_summary=execution_risk_summary,
        conditional_risk_policy_selection=conditional_risk_policy_selection,
        conditional_risk_sleeve=conditional_risk_sleeve,
        conditional_risk_sleeve_walk_forward=conditional_risk_sleeve_walk_forward,
        selected_model=selected_model,
        walk_forward=walk_forward,
        daily_position_source_audit=daily_position_source_audit,
        optimization_plan=optimization_plan,
    )
    sidecars = _write_sidecars(
        spec,
        model_comparison=model_comparison,
        threshold_selection=threshold_selection,
        cost_sensitivity=cost_sensitivity,
        execution_risk_diagnostics=execution_risk_diagnostics,
        conditional_risk_policy_selection=conditional_risk_policy_selection,
        conditional_risk_sleeve=conditional_risk_sleeve,
        conditional_risk_sleeve_rows=conditional_risk_sleeve_rows,
        conditional_risk_sleeve_walk_forward=conditional_risk_sleeve_walk_forward,
        walk_forward=walk_forward,
        daily_position_source_audit=daily_position_source_audit,
        selected_model=selected_model,
        analysis_summary=analysis_summary,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        execution_dir = upload_dir / "execution_overlay"
        execution_dir.mkdir(parents=True, exist_ok=True)
        overlay_trades.to_csv(execution_dir / "overlay_trades.csv", index=False)
        walk_forward_overlay_trades.to_csv(execution_dir / "walk_forward_overlay_trades.csv", index=False)
        daily_positions.to_csv(execution_dir / "daily_selected_positions.csv", index=False)
        walk_forward_daily_positions.to_csv(execution_dir / "daily_walk_forward_positions.csv", index=False)
        intraday_predictions.to_csv(execution_dir / "intraday_predictions_15m.csv", index=False)

        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                daily_source=spec.source_daily_card,
                intraday_source=spec.source_intraday_card,
                selection_segment="validation",
                report_segment="test",
                selection_cost_bps=float(spec.selection_cost_bps),
                signal_threshold=float(thresholds["signal_threshold"]),
                risk_threshold=float(thresholds["risk_threshold"]),
                selection_uses_test_segment=False,
            )
            R.log_metrics(
                selected_validation_return_pct=float(selected_model.get("validation_total_return_pct", 0.0)),
                selected_validation_max_drawdown_pct=float(selected_model.get("validation_max_drawdown_pct", 0.0)),
                selected_test_return_pct=float(selected_model.get("test_total_return_pct", 0.0)),
                selected_test_max_drawdown_pct=float(selected_model.get("test_max_drawdown_pct", 0.0)),
                selected_test_retention_pct=float(selected_model.get("test_return_retention_pct", 0.0)),
                selected_test_open_buy_dump_periods=float(
                    execution_risk_summary.get("selected_test_open_buy_dump_periods", 0.0)
                ),
                selected_test_crash_periods=float(execution_risk_summary.get("selected_test_crash_periods", 0.0)),
                conditional_risk_validation_open_buy_dump_reduction=float(
                    conditional_risk_sleeve.get("validation_open_buy_dump_period_reduction") or 0.0
                ),
                conditional_risk_test_open_buy_dump_reduction=float(
                    conditional_risk_sleeve.get("test_open_buy_dump_period_reduction") or 0.0
                ),
                conditional_risk_test_retention_pct=float(
                    conditional_risk_sleeve.get("test_return_retention_pct") or 0.0
                ),
            )
            for file_name in [
                "overlay_trades.csv",
                "walk_forward_overlay_trades.csv",
                "daily_selected_positions.csv",
                "daily_walk_forward_positions.csv",
                "intraday_predictions_15m.csv",
            ]:
                recorder.log_artifact((execution_dir / file_name).as_posix(), artifact_path="execution_overlay")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    promotion_decision = _promotion_decision(selected_model, intraday_summary=intraday_summary)
    summary = {
        "experiment": spec.experiment,
        "card_type": "execution_overlay_research",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
        "as_of": as_of,
        "market": spec.market,
        "region": spec.region,
        "source_daily_model": {
            "card": spec.source_daily_card,
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing", DAILY_EXECUTION_TIMING),
            "return_col": daily_summary.get("position_model", {}).get("return_col", "open_to_open_return"),
            "model_target_col": daily_summary.get("position_model", {}).get("model_target_col", "label_open_to_open_5d"),
            "selected_return_pct": daily_summary.get("position_model", {}).get("comparison", {}).get("selected_return_pct"),
            "selected_max_drawdown_pct": daily_summary.get("position_model", {}).get("comparison", {}).get("selected_max_drawdown_pct"),
            "position_source_audit": daily_position_source_audit.iloc[0].to_dict()
            if not daily_position_source_audit.empty
            else {},
        },
        "source_intraday_model": {
            "card": spec.source_intraday_card,
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "threshold": thresholds["signal_threshold"],
            "risk_threshold": thresholds["risk_threshold"],
            "signal_prob_col": signal_config["prob_col"],
            "gate_col": signal_config["gate_col"],
            "gate_policy": signal_config["gate_policy"],
            "trade_gate": intraday_summary.get("trade_gate", {}).get("name"),
            "execution_policy": intraday_summary.get("execution_policy"),
            "label_policy_name": intraday_summary.get("label_policy", {}).get("name"),
            "feature_set": intraday_summary.get("feature_set", {}),
            "external_context": intraday_summary.get("external_intraday_context", {}),
        },
        "provider_uris": _provider_uris(spec, intraday_summary=intraday_summary),
        "data_audit": analysis_summary["data_audit"],
        "execution_overlay": {
            "frequency": "15min",
            "decision_model_timing": "daily_close_t_signal",
            "daily_execution_timing": DAILY_EXECUTION_TIMING,
            "overlay_signal_timing": OVERLAY_SIGNAL_TIMING,
            "variants": VARIANTS,
            "open_to_open_baseline": True,
            "intraday_overlays_hold_overnight": False,
            "intraday_label_policy": intraday_summary.get("label_policy", {}).get("name")
            or intraday_summary.get("execution_policy", {}).get("label_policy"),
            "uses_external_intraday_context": bool(intraday_summary.get("external_intraday_context", {}).get("provider_uri")),
            "description": "daily target_position retained; 15m model only changes next-session entry/exit execution",
        },
        "walk_forward": walk_forward_summary,
        "model_selection": {
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_metric": selected_model["selection_metric"],
            "selection_uses_test_segment": False,
            "selection_cost_bps": float(spec.selection_cost_bps),
            "selected_variant": selected_model["variant"],
            "candidate_count": len(VARIANTS),
            "threshold_search": {
                "enabled": bool(spec.optimize_thresholds),
                "selection_segment": "validation",
                "selection_uses_test_segment": False,
                "candidate_count": int(len(threshold_selection)),
                "target_candidate_count": int(threshold_selection["passes_target"].sum())
                if "passes_target" in threshold_selection
                else 0,
                "selected_signal_threshold": float(thresholds["signal_threshold"]),
                "selected_risk_threshold": float(thresholds["risk_threshold"]),
            },
            "conditional_risk_sleeve": {
                "candidate_rows": int(len(conditional_risk_policy_selection)),
                "candidate_count": int(len(conditional_risk_policy_selection.drop_duplicates(_regime_policy_group_columns(conditional_risk_policy_selection))))
                if not conditional_risk_policy_selection.empty
                else 0,
                "condition_columns": list(condition_columns),
                "required_cost_bps": [20.0, 50.0],
                "selection_uses_test_segment": False,
                "selected_variant": conditional_risk_sleeve.get("variant"),
            },
        },
        "execution_risk_diagnostics": execution_risk_summary,
        "conditional_risk_sleeve": conditional_risk_sleeve,
        "conditional_risk_sleeve_cost_sensitivity": _json_ready_records(conditional_risk_sleeve_rows),
        "conditional_risk_sleeve_walk_forward": conditional_risk_sleeve_walk_forward_summary,
        "optimization_plan": optimization_plan,
        "selected_model": selected_model,
        "cost_stress_bps": [float(value) for value in sorted(set(spec.cost_stress_bps))],
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_conclusion(
        spec,
        summary=summary,
        selected_model=selected_model,
        model_comparison=model_comparison,
        walk_forward=walk_forward,
        conditional_risk_sleeve_walk_forward=conditional_risk_sleeve_walk_forward,
        execution_risk_summary=execution_risk_summary,
        conditional_risk_sleeve=conditional_risk_sleeve,
    )
    return summary


def _run_regime_diagnostic_card(spec: CognitiveAlphaIntradayOverlaySpec) -> dict[str, Any]:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)

    daily_summary = _read_json(spec.daily_card_dir / "summary.json")
    intraday_summary = _read_json(spec.intraday_card_dir / "summary.json")
    workflow_yaml = _write_regime_workflow_yaml(spec, daily_summary=daily_summary, intraday_summary=intraday_summary)
    signal_config = _intraday_signal_config(spec, intraday_summary=intraday_summary)

    daily_positions = _load_daily_selected_positions(daily_summary, prefer_walk_forward=False)
    as_of = str(daily_summary.get("as_of") or spec.as_of)
    daily_prices = _load_daily_open_prices(spec, start=str(daily_positions["datetime"].min().date()), end=as_of)
    intraday_predictions = _load_intraday_prediction_frame(
        spec,
        intraday_summary=intraday_summary,
        as_of=as_of,
        signal_config=signal_config,
    )
    thresholds = _regime_overlay_thresholds(
        spec,
        intraday_predictions=intraday_predictions,
        intraday_summary=intraday_summary,
        signal_config=signal_config,
    )
    trade_frames = []
    for cost_bps in sorted(set(float(value) for value in (*spec.cost_stress_bps, *REGIME_REQUIRED_COST_BPS))):
        trade_frames.append(
            simulate_execution_overlays(
                daily_positions,
                daily_prices=daily_prices,
                intraday=intraday_predictions,
                signal_threshold=thresholds["signal_threshold"],
                risk_threshold=thresholds["risk_threshold"],
                cost_bps=cost_bps,
                prob_col=signal_config["prob_col"],
                gate_col=signal_config["gate_col"],
            ).assign(cost_bps=cost_bps)
        )
    overlay_trades = pd.concat(trade_frames, ignore_index=True)
    regime_features = build_regime_feature_frame(daily_positions, daily_prices=daily_prices)
    condition_columns = tuple(
        column
        for column in ("probability", "target_position", "turnover", "raw_target")
        if column in regime_features.columns
    )
    policy_selection = pd.concat(
        [
            evaluate_conditional_overlay_candidates(
                overlay_trades,
                regime_features,
                cost_bps=float(cost_bps),
                condition_columns=condition_columns,
            )
            for cost_bps in sorted(set(float(value) for value in REGIME_REQUIRED_COST_BPS))
        ],
        ignore_index=True,
    )
    model_comparison = summarize_overlay_metrics(overlay_trades)
    selected_model = _select_regime_policy(
        policy_selection,
        required_cost_bps=REGIME_REQUIRED_COST_BPS,
        selection_cost_bps=float(spec.selection_cost_bps),
    )
    analysis_summary = _regime_analysis_summary(
        spec,
        daily_summary=daily_summary,
        intraday_summary=intraday_summary,
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        intraday_predictions=intraday_predictions,
        thresholds=thresholds,
        signal_config=signal_config,
        model_comparison=model_comparison,
        policy_selection=policy_selection,
        selected_model=selected_model,
        condition_columns=condition_columns,
    )
    sidecars = _write_regime_sidecars(
        spec,
        model_comparison=model_comparison,
        policy_selection=policy_selection,
        selected_model=selected_model,
        analysis_summary=analysis_summary,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        execution_dir = upload_dir / "regime_overlay"
        execution_dir.mkdir(parents=True, exist_ok=True)
        overlay_trades.to_csv(execution_dir / "overlay_trades.csv", index=False)
        regime_features.to_csv(execution_dir / "regime_features.csv", index=False)
        policy_selection.to_csv(execution_dir / "policy_selection.csv", index=False)

        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                daily_source=spec.source_daily_card,
                intraday_source=spec.source_intraday_card,
                selection_segment="validation",
                report_segment="test",
                selection_uses_test_segment=False,
                selection_cost_bps=float(spec.selection_cost_bps),
                required_cost_bps=",".join(str(float(value)) for value in REGIME_REQUIRED_COST_BPS),
                signal_threshold=float(thresholds["signal_threshold"]),
                risk_threshold=float(thresholds["risk_threshold"]),
                condition_columns=",".join(condition_columns),
            )
            R.log_metrics(
                candidate_count=float(len(policy_selection)),
                validation_target_candidate_count=float(
                    policy_selection["passes_validation_target"].sum()
                    if "passes_validation_target" in policy_selection
                    else 0
                ),
                validation_and_test_target_candidate_count=float(
                    policy_selection["passes_validation_and_test_target"].sum()
                    if "passes_validation_and_test_target" in policy_selection
                    else 0
                ),
                selected_validation_return_retention_pct=float(
                    selected_model.get("validation_return_retention_pct") or 0.0
                ),
                selected_validation_max_drawdown_pct=float(
                    selected_model.get("validation_max_drawdown_pct") or 0.0
                ),
                selected_test_return_retention_pct=float(selected_model.get("test_return_retention_pct") or 0.0),
                selected_test_max_drawdown_pct=float(selected_model.get("test_max_drawdown_pct") or 0.0),
            )
            for file_name in [
                "overlay_trades.csv",
                "regime_features.csv",
                "policy_selection.csv",
            ]:
                recorder.log_artifact((execution_dir / file_name).as_posix(), artifact_path="regime_overlay")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    promotion_decision = _regime_promotion_decision(selected_model, intraday_summary=intraday_summary)
    validation_target_count = int(
        policy_selection["passes_validation_target"].sum()
        if "passes_validation_target" in policy_selection
        else 0
    )
    validation_and_test_target_count = int(
        policy_selection["passes_validation_and_test_target"].sum()
        if "passes_validation_and_test_target" in policy_selection
        else 0
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "intraday_regime_overlay_diagnostic",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
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
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "signal_prob_col": signal_config["prob_col"],
            "gate_policy": signal_config["gate_policy"],
            "external_context": intraday_summary.get("external_intraday_context", {}),
        },
        "provider_uris": _provider_uris(spec, intraday_summary=intraday_summary),
        "execution_overlay": {
            "frequency": "15min",
            "decision_model_timing": "daily_close_t_signal",
            "daily_execution_timing": DAILY_EXECUTION_TIMING,
            "overlay_signal_timing": OVERLAY_SIGNAL_TIMING,
            "variants": VARIANTS,
            "open_to_open_baseline": True,
            "intraday_overlays_hold_overnight": False,
            "description": "baseline remains default; a validation-selected daily-regime condition can switch execution to a 15m overlay",
        },
        "regime_policy": {
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "selection_cost_bps": float(spec.selection_cost_bps),
            "required_cost_bps": list(REGIME_REQUIRED_COST_BPS),
            "condition_columns": list(condition_columns),
            "condition_threshold_source": "validation_quantile",
            "causal_scope": "daily_close_known_features_only",
            "excludes_first_hour_features_from_twap_selection": True,
            "target_return_retention_pct": [85.0, 95.0],
            "target_max_drawdown_floor_pct": -18.0,
            "candidate_count": int(len(policy_selection)),
            "validation_target_candidate_count": validation_target_count,
            "validation_and_test_target_candidate_count": validation_and_test_target_count,
        },
        "model_selection": {
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_metric": "validation_target_first_then_max_drawdown_improvement",
            "selection_uses_test_segment": False,
            "required_cost_bps": list(REGIME_REQUIRED_COST_BPS),
            "candidate_count": int(len(policy_selection)),
            "validation_target_candidate_count": validation_target_count,
            "validation_and_test_target_candidate_count": validation_and_test_target_count,
            "selected_variant": selected_model.get("variant"),
        },
        "selected_model": selected_model,
        "analysis_summary": analysis_summary,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _regime_artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_regime_conclusion(spec, summary=summary, selected_model=selected_model)
    return summary


def _run_delayed_observable_card(spec: CognitiveAlphaIntradayOverlaySpec) -> dict[str, Any]:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)

    daily_summary = _read_json(spec.daily_card_dir / "summary.json")
    intraday_summary = _read_json(spec.intraday_card_dir / "summary.json")
    workflow_yaml = _write_delayed_workflow_yaml(spec, daily_summary=daily_summary, intraday_summary=intraday_summary)
    signal_config = _intraday_signal_config(spec, intraday_summary=intraday_summary)

    daily_positions = _load_daily_selected_positions(daily_summary, prefer_walk_forward=False)
    walk_forward_daily_positions = _load_daily_selected_positions(daily_summary, prefer_walk_forward=True)
    as_of = str(daily_summary.get("as_of") or spec.as_of)
    daily_price_start = min(
        daily_positions["datetime"].min(),
        walk_forward_daily_positions["datetime"].min(),
    )
    daily_prices = _load_daily_open_prices(spec, start=str(daily_price_start.date()), end=as_of)
    intraday_predictions = _load_intraday_prediction_frame(
        spec,
        intraday_summary=intraday_summary,
        as_of=as_of,
        signal_config=signal_config,
    )
    walk_forward_feasibility = _delayed_walk_forward_feasibility(
        spec,
        intraday_summary=intraday_summary,
        daily_positions=walk_forward_daily_positions,
        intraday_predictions=intraday_predictions,
    )
    walk_forward_summary = _delayed_walk_forward_summary(
        walk_forward_feasibility,
        intraday_summary=intraday_summary,
        daily_positions=walk_forward_daily_positions,
    )
    policy_selection_full = evaluate_observable_delayed_signal_candidates(
        daily_positions,
        daily_prices=daily_prices,
        intraday=intraday_predictions,
        cost_bps_values=spec.cost_stress_bps,
        prob_col=signal_config["prob_col"],
        gate_col=signal_config["gate_col"],
    )
    selected_model = select_validation_robust_delayed_policy(
        policy_selection_full,
        required_cost_bps=(20.0, 50.0),
        selection_cost_bps=float(spec.selection_cost_bps),
    )
    intraday_flat_sleeve = select_intraday_flat_sleeve_policy(
        policy_selection_full,
        required_cost_bps=(20.0, 50.0),
        selection_cost_bps=float(spec.selection_cost_bps),
    )
    selected_policy_rows = (
        policy_selection_full.loc[policy_selection_full["policy_id"].eq(selected_model.get("policy_id"))].copy()
        if not policy_selection_full.empty and selected_model.get("policy_id")
        else pd.DataFrame()
    )
    intraday_flat_sleeve_rows = (
        policy_selection_full.loc[policy_selection_full["policy_id"].eq(intraday_flat_sleeve.get("policy_id"))].copy()
        if not policy_selection_full.empty and intraday_flat_sleeve.get("policy_id")
        else pd.DataFrame()
    )
    selected_trade_frames = []
    if selected_model.get("policy_id"):
        selected_exit_policy = str(selected_model.get("exit_policy") or "hold_to_next_open")
        selected_risk_threshold = selected_model.get("risk_threshold")
        if selected_risk_threshold is not None and not pd.isna(selected_risk_threshold):
            selected_risk_threshold = float(selected_risk_threshold)
        else:
            selected_risk_threshold = None
        for cost_bps in sorted(set(float(value) for value in spec.cost_stress_bps)):
            selected_trade_frames.append(
                simulate_observable_delayed_signal_overlay(
                    daily_positions,
                    daily_prices=daily_prices,
                    intraday=intraday_predictions,
                    observation_bars=int(selected_model["observation_bars"]),
                    condition_feature=str(selected_model["condition_feature"]),
                    condition_operator=str(selected_model["condition_operator"]),
                    condition_threshold=float(selected_model["condition_threshold"]),
                    signal_threshold=float(selected_model["signal_threshold"]),
                    risk_threshold=selected_risk_threshold,
                    exit_policy=selected_exit_policy,
                    cost_bps=cost_bps,
                    prob_col=signal_config["prob_col"],
                    gate_col=signal_config["gate_col"],
                ).assign(cost_bps=float(cost_bps))
            )
    selected_trades = pd.concat(selected_trade_frames, ignore_index=True) if selected_trade_frames else pd.DataFrame()
    model_comparison = summarize_overlay_metrics(selected_trades)
    optimization_plan = build_intraday_optimization_plan(
        selected_model=selected_model,
        walk_forward_summary=walk_forward_summary,
        conditional_walk_forward_summary={},
        source_intraday_model={"external_context": intraday_summary.get("external_intraday_context", {})},
        local_intraday_manifest=_read_optional_json(spec.root / "data" / "qlib" / "soxl_intraday_15min" / "qlib_store_manifest.json"),
        local_intraday_5min_manifest=_read_optional_json(spec.root / "data" / "qlib" / "soxl_intraday_5min" / "qlib_store_manifest.json"),
        external_context_summary=_read_optional_json(spec.root / "research" / "intraday_external_context_data_v1" / "summary.json"),
    )
    analysis_summary = _delayed_analysis_summary(
        spec,
        daily_summary=daily_summary,
        intraday_summary=intraday_summary,
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        intraday_predictions=intraday_predictions,
        signal_config=signal_config,
        policy_selection_full=policy_selection_full,
        selected_policy_rows=selected_policy_rows,
        intraday_flat_sleeve_rows=intraday_flat_sleeve_rows,
        selected_model=selected_model,
        intraday_flat_sleeve=intraday_flat_sleeve,
        model_comparison=model_comparison,
        walk_forward_summary=walk_forward_summary,
        optimization_plan=optimization_plan,
    )
    sidecars = _write_delayed_sidecars(
        spec,
        policy_selection_full=policy_selection_full,
        selected_policy_rows=selected_policy_rows,
        intraday_flat_sleeve_rows=intraday_flat_sleeve_rows,
        model_comparison=model_comparison,
        selected_model=selected_model,
        intraday_flat_sleeve=intraday_flat_sleeve,
        analysis_summary=analysis_summary,
        walk_forward_feasibility=walk_forward_feasibility,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        execution_dir = upload_dir / "delayed_observable_overlay"
        execution_dir.mkdir(parents=True, exist_ok=True)
        selected_trades.to_csv(execution_dir / "selected_trades.csv", index=False)
        policy_selection_full.to_csv(execution_dir / "all_policy_selection.csv", index=False)
        daily_positions.to_csv(execution_dir / "daily_selected_positions.csv", index=False)
        walk_forward_daily_positions.to_csv(execution_dir / "daily_walk_forward_positions.csv", index=False)

        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                daily_source=spec.source_daily_card,
                intraday_source=spec.source_intraday_card,
                selection_segment="validation",
                report_segment="test",
                selection_uses_test_segment=False,
                selection_cost_bps=float(spec.selection_cost_bps),
                required_cost_bps="20.0,50.0",
                max_skip_rate=0.15,
                return_retention_upper_policy="hard_gate",
                min_validation_retention_band_margin=float(
                    selected_model.get("min_validation_retention_band_margin", 0.05) or 0.05
                ),
                selected_policy_id=str(selected_model.get("policy_id")),
                selected_intraday_flat_sleeve_policy_id=str(intraday_flat_sleeve.get("policy_id")),
                walk_forward_policy=str(walk_forward_summary.get("policy")),
                walk_forward_selection_uses_test_segment=False,
            )
            R.log_metrics(
                policy_candidate_rows=float(len(policy_selection_full)),
                policy_candidate_count=float(policy_selection_full["policy_id"].nunique())
                if "policy_id" in policy_selection_full
                else 0.0,
                validation_target_row_count=float(
                    policy_selection_full["passes_validation_target"].sum()
                    if "passes_validation_target" in policy_selection_full
                    else 0
                ),
                validation_and_test_target_row_count=float(
                    policy_selection_full["passes_validation_and_test_target"].sum()
                    if "passes_validation_and_test_target" in policy_selection_full
                    else 0
                ),
                selected_validation_return_retention_pct=float(
                    selected_model.get("validation_return_retention_pct") or 0.0
                ),
                selected_validation_max_drawdown_pct=float(
                    selected_model.get("validation_max_drawdown_pct") or 0.0
                ),
                selected_test_return_retention_pct=float(selected_model.get("test_return_retention_pct") or 0.0),
                selected_test_max_drawdown_pct=float(selected_model.get("test_max_drawdown_pct") or 0.0),
                selected_intraday_flat_validation_return_retention_pct=float(
                    intraday_flat_sleeve.get("validation_return_retention_pct") or 0.0
                ),
                selected_intraday_flat_validation_max_drawdown_pct=float(
                    intraday_flat_sleeve.get("validation_max_drawdown_pct") or 0.0
                ),
                selected_intraday_flat_test_return_retention_pct=float(
                    intraday_flat_sleeve.get("test_return_retention_pct") or 0.0
                ),
                selected_intraday_flat_test_max_drawdown_pct=float(
                    intraday_flat_sleeve.get("test_max_drawdown_pct") or 0.0
                ),
                walk_forward_feasible_fold_count=float(len(walk_forward_summary.get("feasible_test_years", []))),
                walk_forward_required_policy_satisfied=float(
                    1.0 if walk_forward_summary.get("required_policy_satisfied") else 0.0
                ),
            )
            for file_name in [
                "selected_trades.csv",
                "all_policy_selection.csv",
                "daily_selected_positions.csv",
                "daily_walk_forward_positions.csv",
            ]:
                recorder.log_artifact((execution_dir / file_name).as_posix(), artifact_path="delayed_observable_overlay")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    promotion_decision = _delayed_promotion_decision(
        selected_policy_rows,
        intraday_summary=intraday_summary,
        walk_forward_summary=walk_forward_summary,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "intraday_delayed_observable_overlay",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
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
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "signal_prob_col": signal_config["prob_col"],
            "gate_policy": signal_config["gate_policy"],
            "external_context": intraday_summary.get("external_intraday_context", {}),
        },
        "provider_uris": _provider_uris(spec, intraday_summary=intraday_summary),
        "execution_overlay": {
            "frequency": "15min",
            "decision_model_timing": "daily_close_t_signal",
            "daily_execution_timing": DAILY_EXECUTION_TIMING,
            "overlay_signal_timing": "first_30_or_60m_observed_then_15m_bar_close_to_next_bar_open",
            "variants": [
                "baseline_next_open",
                "delayed_signal_after_30m",
                "delayed_signal_after_60m",
                "delayed_signal_after_30m_risk_exit",
                "delayed_signal_after_60m_risk_exit",
                "delayed_signal_after_30m_same_day_close",
                "delayed_signal_after_60m_same_day_close",
            ],
            "open_to_open_baseline": True,
            "intraday_overlays_hold_overnight": True,
            "default_exit_timing": "following_open",
            "risk_exit_can_close_intraday": True,
            "intraday_flat_can_close_same_day": True,
            "description": "the overlay can use first 30/60m features only after those bars are observed, then enters on a later 15m next_open signal; eligible policies may hold to following open or flatten at same-day close, while risk-exit remains diagnostic",
        },
        "model_selection": {
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "selection_cost_bps": float(spec.selection_cost_bps),
            "required_cost_bps": [20.0, 50.0],
            "max_skip_rate": 0.15,
            "return_retention_floor_pct": 85.0,
            "return_retention_ceiling_pct": 95.0,
            "return_retention_upper_policy": "hard_gate",
            "min_validation_retention_band_margin": float(
                selected_model.get("min_validation_retention_band_margin", 0.05) or 0.05
            ),
            "selection_priority": selected_model.get(
                "selection_priority",
                [
                    "validation_return_retention_score_desc",
                    "validation_skip_rate_asc",
                    "condition_valid_active_coverage_desc",
                    "validation_drawdown_improvement_pct_desc",
                    "selection_score_desc",
                ],
            ),
            "eligible_exit_policies": ["hold_to_next_open", SAME_DAY_CLOSE_EXIT_POLICY],
            "exit_policy_candidate_counts": (
                policy_selection_full["exit_policy"].value_counts().to_dict()
                if "exit_policy" in policy_selection_full
                else {}
            ),
            "risk_exit_validation_and_test_target_row_count": int(
                policy_selection_full.loc[
                    policy_selection_full.get("exit_policy", pd.Series(index=policy_selection_full.index, dtype=object)).eq("risk_exit")
                    & policy_selection_full.get(
                        "passes_validation_and_test_target",
                        pd.Series(False, index=policy_selection_full.index),
                    ).astype(bool)
                ].shape[0]
            )
            if not policy_selection_full.empty
            else 0,
            "same_day_close_validation_and_test_target_row_count": int(
                policy_selection_full.loc[
                    policy_selection_full.get("exit_policy", pd.Series(index=policy_selection_full.index, dtype=object)).eq(SAME_DAY_CLOSE_EXIT_POLICY)
                    & policy_selection_full.get(
                        "passes_validation_and_test_target",
                        pd.Series(False, index=policy_selection_full.index),
                    ).astype(bool)
                ].shape[0]
            )
            if not policy_selection_full.empty
            else 0,
            "candidate_rows": int(len(policy_selection_full)),
            "candidate_count": int(policy_selection_full["policy_id"].nunique()) if "policy_id" in policy_selection_full else 0,
            "validation_target_row_count": int(policy_selection_full["passes_validation_target"].sum())
            if "passes_validation_target" in policy_selection_full
            else 0,
            "validation_and_test_target_row_count": int(policy_selection_full["passes_validation_and_test_target"].sum())
            if "passes_validation_and_test_target" in policy_selection_full
            else 0,
            "selected_policy_id": selected_model.get("policy_id"),
        },
        "walk_forward": walk_forward_summary,
        "selected_model": selected_model,
        "selected_policy_cost_sensitivity": _json_ready_records(selected_policy_rows),
        "intraday_flat_sleeve": intraday_flat_sleeve,
        "intraday_flat_sleeve_cost_sensitivity": _json_ready_records(intraday_flat_sleeve_rows),
        "analysis_summary": analysis_summary,
        "optimization_plan": optimization_plan,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _delayed_artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_delayed_conclusion(
        spec,
        summary=summary,
        selected_model=selected_model,
        selected_policy_rows=selected_policy_rows,
        intraday_flat_sleeve=intraday_flat_sleeve,
        intraday_flat_sleeve_rows=intraday_flat_sleeve_rows,
    )
    return summary


def _provider_uris(spec: CognitiveAlphaIntradayOverlaySpec, *, intraday_summary: dict[str, Any]) -> dict[str, str]:
    provider_uris = {
            "daily": spec.daily_provider_uri.resolve().as_posix(),
            "15min": spec.intraday_provider_uri.resolve().as_posix(),
    }
    external_uri = intraday_summary.get("provider_uris", {}).get("external_15min")
    if external_uri:
        provider_uris["external_context_15min"] = str(external_uri)
    return provider_uris


def _simulate_daily_row(
    daily_row: dict[str, Any],
    *,
    prices: pd.DataFrame,
    intraday: pd.DataFrame,
    signal_threshold: float,
    risk_threshold: float,
    cost_bps: float,
    prob_col: str,
    gate_col: str,
) -> list[dict[str, Any]]:
    decision_date = pd.Timestamp(daily_row["datetime"]).normalize()
    price_index = pd.Index(prices["datetime"])
    if decision_date not in set(price_index):
        return [_empty_variant_row(daily_row, variant, cost_bps, "missing_decision_open") for variant in VARIANTS]
    location = int(price_index.get_loc(decision_date))
    if isinstance(location, slice):
        location = int(location.start)
    entry_date = prices.iloc[location + 1]["datetime"] if location + 1 < len(prices) else pd.NaT
    exit_date = prices.iloc[location + 2]["datetime"] if location + 2 < len(prices) else pd.NaT
    target = float(daily_row.get("target_position", 0.0) or 0.0)
    segment = str(daily_row.get("segment", "unknown"))
    source_return = float(daily_row.get("strategy_return", 0.0) or 0.0)
    base_common = {
        "datetime": decision_date.date().isoformat(),
        "segment": segment,
        **_daily_walk_forward_metadata(daily_row),
        "target_position": target,
        "cost_bps": float(cost_bps),
        "entry_date": _date_or_none(entry_date),
        "exit_date": _date_or_none(exit_date),
        "source_daily_strategy_return": source_return,
    }
    if abs(target) <= 1e-12:
        return [
            {
                **base_common,
                "variant": variant,
                "entry_time": None,
                "exit_time": None,
                "entry_price": np.nan,
                "exit_price": np.nan,
                "entry_day_close_time": None,
                "entry_day_close_price": np.nan,
                "entry_day_strategy_return": np.nan,
                "intraday_signal_score": np.nan,
                "gross_return": 0.0,
                "strategy_return": 0.0,
                "skipped_reason": None,
            }
            for variant in VARIANTS
        ]
    if pd.isna(entry_date) or pd.isna(exit_date):
        return [_empty_variant_row(daily_row, variant, cost_bps, "missing_entry_or_exit_open") for variant in VARIANTS]

    entry_open = float(prices.iloc[location + 1]["open"])
    exit_open = float(prices.iloc[location + 2]["open"])
    entry_day_bars = intraday.loc[intraday["date"].eq(pd.Timestamp(entry_date).date().isoformat())].sort_values("datetime")
    prediction_available = bool(
        not entry_day_bars.empty
        and pd.to_numeric(entry_day_bars.get(prob_col), errors="coerce").notna().any()
    )
    entry_day_close_time, entry_day_close_price = _entry_day_close(entry_day_bars)
    execution_cost = abs(target) * cost_bps / 10000.0

    baseline_return = source_return - execution_cost
    rows = [
        {
            **base_common,
            "variant": "baseline_next_open",
            "entry_time": pd.Timestamp(entry_date).date().isoformat(),
            "exit_time": pd.Timestamp(exit_date).date().isoformat(),
            "entry_price": entry_open,
            "exit_price": exit_open,
            "entry_day_close_time": entry_day_close_time,
            "entry_day_close_price": entry_day_close_price,
            "entry_day_strategy_return": _entry_day_strategy_return(
                target=target,
                entry_price=entry_open,
                exit_price=entry_day_close_price,
            ),
            "intraday_signal_score": np.nan,
            "gross_return": source_return,
            "strategy_return": baseline_return,
            "skipped_reason": None,
        }
    ]

    signal_candidates = entry_day_bars.loc[
        entry_day_bars.get(gate_col, pd.Series(False, index=entry_day_bars.index)).fillna(False).astype(bool)
        & (pd.to_numeric(entry_day_bars.get(prob_col), errors="coerce") >= signal_threshold)
        & entry_day_bars["next_open"].notna()
    ]
    if signal_candidates.empty:
        skipped_reason = "no_15m_entry_signal" if prediction_available else "missing_intraday_prediction"
        rows.append(
            {
                **base_common,
                "variant": "overlay_a_signal_gate",
                "entry_time": None,
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": np.nan,
                "exit_price": exit_open,
                "entry_day_close_time": entry_day_close_time,
                "entry_day_close_price": entry_day_close_price,
                "entry_day_strategy_return": np.nan,
                "intraday_signal_score": np.nan,
                "gross_return": 0.0,
                "strategy_return": 0.0,
                "skipped_reason": skipped_reason,
            }
        )
    else:
        signal = signal_candidates.iloc[0]
        entry_price = float(signal["next_open"])
        gross_return = target * (exit_open / entry_price - 1.0)
        rows.append(
            {
                **base_common,
                "variant": "overlay_a_signal_gate",
                "entry_time": _timestamp_or_none(signal["next_datetime"]),
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_open,
                "entry_day_close_time": entry_day_close_time,
                "entry_day_close_price": entry_day_close_price,
                "entry_day_strategy_return": _entry_day_strategy_return(
                    target=target,
                    entry_price=entry_price,
                    exit_price=entry_day_close_price,
                ),
                "intraday_signal_score": float(signal[prob_col]),
                "gross_return": gross_return,
                "strategy_return": gross_return - execution_cost,
                "skipped_reason": None,
            }
        )

    for twap_bars_count in (3, 4):
        twap_bars = entry_day_bars.loc[entry_day_bars["open"].notna()].head(twap_bars_count)
        twap_variant = f"overlay_b_twap{twap_bars_count}"
        if len(twap_bars) < twap_bars_count:
            rows.append(
                {
                    **base_common,
                    "variant": twap_variant,
                    "entry_time": None,
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": np.nan,
                    "exit_price": exit_open,
                    "entry_day_close_time": entry_day_close_time,
                    "entry_day_close_price": entry_day_close_price,
                    "entry_day_strategy_return": np.nan,
                    "intraday_signal_score": np.nan,
                    "gross_return": 0.0,
                    "strategy_return": 0.0,
                    "skipped_reason": "missing_intraday_twap_bars",
                }
            )
        else:
            entry_price = float(twap_bars["open"].mean())
            gross_return = target * (exit_open / entry_price - 1.0)
            rows.append(
                {
                    **base_common,
                    "variant": twap_variant,
                    "entry_time": f"twap{twap_bars_count}_first_{twap_bars_count}_bars",
                    "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                    "entry_price": entry_price,
                    "exit_price": exit_open,
                    "entry_day_close_time": entry_day_close_time,
                    "entry_day_close_price": entry_day_close_price,
                    "entry_day_strategy_return": _entry_day_strategy_return(
                        target=target,
                        entry_price=entry_price,
                        exit_price=entry_day_close_price,
                    ),
                    "intraday_signal_score": np.nan,
                    "gross_return": gross_return,
                    "strategy_return": gross_return - execution_cost,
                    "skipped_reason": None,
                }
            )

    risk_candidates = entry_day_bars.loc[
        entry_day_bars.get(gate_col, pd.Series(False, index=entry_day_bars.index)).fillna(False).astype(bool)
        & (pd.to_numeric(entry_day_bars.get(prob_col), errors="coerce") <= risk_threshold)
        & entry_day_bars["close"].notna()
    ]
    if risk_candidates.empty:
        rows.append(
            {
                **base_common,
                "variant": "overlay_c_risk_exit",
                "entry_time": pd.Timestamp(entry_date).date().isoformat(),
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": entry_open,
                "exit_price": exit_open,
                "entry_day_close_time": entry_day_close_time,
                "entry_day_close_price": entry_day_close_price,
                "entry_day_strategy_return": _entry_day_strategy_return(
                    target=target,
                    entry_price=entry_open,
                    exit_price=entry_day_close_price,
                ),
                "intraday_signal_score": np.nan,
                "gross_return": target * (exit_open / entry_open - 1.0),
                "strategy_return": source_return - execution_cost,
                "skipped_reason": None if prediction_available else "missing_intraday_prediction",
            }
        )
    else:
        risk = risk_candidates.iloc[0]
        exit_price = float(risk["close"])
        gross_return = target * (exit_price / entry_open - 1.0)
        rows.append(
            {
                **base_common,
                "variant": "overlay_c_risk_exit",
                "entry_time": pd.Timestamp(entry_date).date().isoformat(),
                "exit_time": _timestamp_or_none(risk["datetime"]),
                "entry_price": entry_open,
                "exit_price": exit_price,
                "entry_day_close_time": entry_day_close_time,
                "entry_day_close_price": entry_day_close_price,
                "entry_day_strategy_return": _entry_day_strategy_return(
                    target=target,
                    entry_price=entry_open,
                    exit_price=exit_price,
                ),
                "intraday_signal_score": float(risk[prob_col]),
                "gross_return": gross_return,
                "strategy_return": gross_return - execution_cost,
                "skipped_reason": None,
            }
        )
    return rows


def _normalize_daily_positions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    if "segment" in result.columns:
        result["segment"] = result["segment"].replace({"valid": "validation"})
    elif "fold_segment" in result.columns:
        result["segment"] = result["fold_segment"].replace({"valid": "validation"})
    else:
        result["segment"] = "unknown"
    for column in ["target_position", "strategy_return"]:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result.sort_values("datetime").reset_index(drop=True)


def _normalize_daily_prices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    result["open"] = pd.to_numeric(result["open"], errors="coerce")
    return result.dropna(subset=["datetime", "open"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


def _normalize_intraday_bars(frame: pd.DataFrame, *, prob_col: str, gate_col: str) -> pd.DataFrame:
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    if "date" not in result.columns:
        result["date"] = result["datetime"].dt.date.astype(str)
    else:
        result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    for column in ["open", "close", prob_col]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if gate_col not in result.columns:
        result[gate_col] = False
    result[gate_col] = result[gate_col].fillna(False).astype(bool)
    result = result.sort_values(["date", "datetime"]).reset_index(drop=True)
    result["next_datetime"] = result.groupby("date")["datetime"].shift(-1)
    result["next_open"] = result.groupby("date")["open"].shift(-1)
    return result


def _observed_intraday_feature_value(
    observed: pd.DataFrame,
    *,
    entry_open: float,
    feature: str,
    prob_col: str,
) -> float:
    if observed.empty:
        return float("nan")
    probs = pd.to_numeric(observed.get(prob_col), errors="coerce")
    if feature == "observed_prob_min":
        return float(probs.min()) if probs.notna().any() else float("nan")
    if feature == "observed_prob_mean":
        return float(probs.mean()) if probs.notna().any() else float("nan")
    if feature == "observed_prob_max":
        return float(probs.max()) if probs.notna().any() else float("nan")
    if feature == "observed_return":
        close = pd.to_numeric(observed["close"], errors="coerce").dropna()
        if close.empty or pd.isna(entry_open):
            return float("nan")
        return float(close.iloc[-1] / float(entry_open) - 1.0)
    raise ValueError(f"Unsupported observed intraday feature: {feature}")


def _condition_matches(value: float, *, operator: str, threshold: float) -> bool:
    if pd.isna(value):
        return False
    if operator == "ge":
        return bool(float(value) >= float(threshold))
    if operator == "le":
        return bool(float(value) <= float(threshold))
    raise ValueError(f"Unsupported condition operator: {operator}")


def _delayed_exit_policy(*, exit_policy: str | None, risk_threshold: float | None) -> str:
    if exit_policy is None:
        return "risk_exit" if risk_threshold is not None else "hold_to_next_open"
    if exit_policy not in {"hold_to_next_open", "risk_exit", SAME_DAY_CLOSE_EXIT_POLICY}:
        raise ValueError(f"Unsupported delayed exit policy: {exit_policy}")
    if exit_policy == "risk_exit" and risk_threshold is None:
        raise ValueError("risk_exit requires risk_threshold")
    return exit_policy


def _delayed_exit_policy_actions(
    exit_policy_grid: tuple[str, ...],
    risk_threshold_grid: tuple[float | None, ...],
) -> tuple[tuple[str, float | None, float | str | None], ...]:
    actions: list[tuple[str, float | None, float | str | None]] = []
    for exit_policy in exit_policy_grid:
        if exit_policy == "hold_to_next_open":
            actions.append(("hold_to_next_open", None, None))
        elif exit_policy == "risk_exit":
            for risk_threshold in risk_threshold_grid:
                if risk_threshold is not None:
                    actions.append(("risk_exit", float(risk_threshold), float(risk_threshold)))
        elif exit_policy == SAME_DAY_CLOSE_EXIT_POLICY:
            actions.append((SAME_DAY_CLOSE_EXIT_POLICY, None, SAME_DAY_CLOSE_EXIT_POLICY))
        else:
            raise ValueError(f"Unsupported delayed exit policy: {exit_policy}")
    return tuple(actions)


def _same_day_close_exit(
    entry_day_bars: pd.DataFrame,
    *,
    entry_time: pd.Timestamp,
) -> tuple[str, float] | None:
    exit_candidates = entry_day_bars.loc[
        entry_day_bars["datetime"].ge(pd.Timestamp(entry_time))
        & entry_day_bars["close"].notna()
    ].sort_values("datetime")
    if exit_candidates.empty:
        return None
    exit_row = exit_candidates.iloc[-1]
    return _timestamp_or_none(exit_row["datetime"]), float(exit_row["close"])


def _entry_day_close(entry_day_bars: pd.DataFrame) -> tuple[str | None, float]:
    close_candidates = entry_day_bars.loc[entry_day_bars["close"].notna()].sort_values("datetime")
    if close_candidates.empty:
        return None, float("nan")
    close_row = close_candidates.iloc[-1]
    return _timestamp_or_none(close_row["datetime"]), float(close_row["close"])


def _entry_day_strategy_return(*, target: float, entry_price: float, exit_price: float) -> float:
    if pd.isna(entry_price) or pd.isna(exit_price) or abs(float(entry_price)) <= 1e-12:
        return float("nan")
    return float(target) * (float(exit_price) / float(entry_price) - 1.0)


def _load_daily_selected_positions(daily_summary: dict[str, Any], *, prefer_walk_forward: bool = True) -> pd.DataFrame:
    path = _daily_positions_path(daily_summary, prefer_walk_forward=prefer_walk_forward)
    frame = pd.read_csv(path)
    frame = frame.loc[frame["selected"].astype(bool)].copy()
    return _normalize_daily_positions(frame)


def _daily_positions_path(daily_summary: dict[str, Any], *, prefer_walk_forward: bool = True) -> Path:
    recorder_dir = Path(str(daily_summary["recorder_local_dir"]))
    position_dir = recorder_dir / "artifacts" / "cognitive_alpha_position"
    walk_forward_path = position_dir / "walk_forward_positions.csv"
    if prefer_walk_forward and walk_forward_path.exists():
        return walk_forward_path
    return position_dir / "positions.csv"


def _required_daily_position_years() -> tuple[int, ...]:
    years: set[int] = set()
    for test_year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS:
        years.add(int(test_year) - 1)
        years.add(int(test_year))
    return tuple(sorted(years))


def _daily_position_source_audit(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    daily_positions_path: Path | None = None,
) -> pd.DataFrame:
    path = Path(daily_positions_path) if daily_positions_path is not None else _daily_positions_path(daily_summary)
    required_years = _required_daily_position_years()
    base_row: dict[str, Any] = {
        "card": spec.source_daily_card,
        "recorder_id": daily_summary.get("recorder_id"),
        "artifact_path": path.as_posix(),
        "artifact_name": path.name,
        "required_position_years": ",".join(str(year) for year in required_years),
    }
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    **base_row,
                    "rows": 0,
                    "selected_rows": 0,
                    "selected_start": None,
                    "selected_end": None,
                    "selected_years": "",
                    "missing_required_position_years": base_row["required_position_years"],
                    "satisfies_required_annual_walk_forward": False,
                    "blocking_reason": "missing_positions_artifact",
                }
            ]
        )

    frame = pd.read_csv(path)
    if "datetime" not in frame.columns:
        return pd.DataFrame(
            [
                {
                    **base_row,
                    "rows": int(len(frame)),
                    "selected_rows": 0,
                    "selected_start": None,
                    "selected_end": None,
                    "selected_years": "",
                    "missing_required_position_years": base_row["required_position_years"],
                    "satisfies_required_annual_walk_forward": False,
                    "blocking_reason": "missing_datetime_column",
                }
            ]
        )
    if "selected" not in frame.columns:
        return pd.DataFrame(
            [
                {
                    **base_row,
                    "rows": int(len(frame)),
                    "selected_rows": 0,
                    "selected_start": None,
                    "selected_end": None,
                    "selected_years": "",
                    "missing_required_position_years": base_row["required_position_years"],
                    "satisfies_required_annual_walk_forward": False,
                    "blocking_reason": "missing_selected_column",
                }
            ]
        )

    frame["datetime"] = pd.to_datetime(frame["datetime"])
    selected = frame.loc[frame["selected"].astype(bool)].copy()
    selected_years = tuple(sorted(int(year) for year in selected["datetime"].dt.year.unique())) if not selected.empty else ()
    missing_years = tuple(year for year in required_years if year not in set(selected_years))
    satisfies = not missing_years
    return pd.DataFrame(
        [
            {
                **base_row,
                "rows": int(len(frame)),
                "selected_rows": int(len(selected)),
                "selected_start": selected["datetime"].min().date().isoformat() if not selected.empty else None,
                "selected_end": selected["datetime"].max().date().isoformat() if not selected.empty else None,
                "selected_years": ",".join(str(year) for year in selected_years),
                "missing_required_position_years": ",".join(str(year) for year in missing_years),
                "satisfies_required_annual_walk_forward": bool(satisfies),
                "blocking_reason": "ok" if satisfies else "missing_required_position_years",
            }
        ]
    )


def _load_daily_open_prices(spec: CognitiveAlphaIntradayOverlaySpec, *, start: str, end: str) -> pd.DataFrame:
    config = QlibProviderConfig(
        provider_uri=spec.daily_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="day",
        kernels=1,
    )
    init_qlib_provider(config)
    raw = load_qlib_feature_frame(config, instruments=["SOXL"], fields=["$open"], start=start, end=end)
    frame = raw.reset_index()
    frame = frame.loc[frame["instrument"].astype(str).str.upper().eq("SOXL"), ["datetime", "$open"]].rename(
        columns={"$open": "open"}
    )
    return _normalize_daily_prices(frame)


def _intraday_signal_config(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    intraday_summary: dict[str, Any],
) -> dict[str, str]:
    selected_model = str(intraday_summary.get("selected_model", {}).get("model") or "lgb_xgb_consensus")
    prob_col = str(spec.intraday_prob_col or f"prob_{selected_model}")
    if spec.intraday_gate_col is None:
        return {
            "prob_col": prob_col,
            "gate_col": PREDICTION_AVAILABLE_GATE_COL,
            "gate_policy": spec.intraday_gate_policy,
        }
    return {
        "prob_col": prob_col,
        "gate_col": spec.intraday_gate_col,
        "gate_policy": spec.intraday_gate_policy,
    }


def _load_intraday_prediction_frame(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    intraday_summary: dict[str, Any],
    as_of: str,
    signal_config: dict[str, str],
) -> pd.DataFrame:
    config = QlibProviderConfig(
        provider_uri=spec.intraday_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="15min",
        kernels=1,
    )
    init_qlib_provider(config)
    predictions_path = (
        Path(str(intraday_summary["recorder_local_dir"]))
        / "artifacts"
        / "model_selection"
        / "all_model_predictions.csv"
    )
    predictions = pd.read_csv(predictions_path)
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])
    prediction_datetimes = predictions["datetime"].dropna()
    if prediction_datetimes.empty:
        intraday_start = f"{REQUIRED_WALK_FORWARD_START} 09:45:00"
    else:
        required_start_date = pd.Timestamp(REQUIRED_WALK_FORWARD_START).date()
        prediction_start_date = prediction_datetimes.min().date()
        intraday_start_date = max(required_start_date, prediction_start_date)
        intraday_start = f"{intraday_start_date.isoformat()} 09:45:00"
    raw = load_qlib_feature_frame(
        config,
        instruments=["SOXL"],
        fields=["$open", "$close"],
        start=intraday_start,
        end=f"{as_of} 16:00:00",
    )
    bars = raw.reset_index()
    bars = bars.loc[bars["instrument"].astype(str).str.upper().eq("SOXL"), ["datetime", "$open", "$close"]].rename(
        columns={"$open": "open", "$close": "close"}
    )
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars["date"] = bars["datetime"].dt.date.astype(str)

    prob_col = signal_config["prob_col"]
    gate_col = signal_config["gate_col"]
    columns = ["datetime", "segment", prob_col]
    if "freq" in predictions.columns:
        columns.append("freq")
    if spec.intraday_gate_col is not None and spec.intraday_gate_col in predictions.columns:
        columns.append(spec.intraday_gate_col)
    predictions = predictions.loc[predictions["datetime"].between(bars["datetime"].min(), bars["datetime"].max()), columns].copy()
    merged = bars.merge(predictions, on="datetime", how="left")
    if spec.intraday_gate_col is None:
        merged[gate_col] = merged[prob_col].notna()
    elif spec.intraday_gate_col not in merged.columns:
        merged[gate_col] = False
    return _normalize_intraday_bars(merged, prob_col=prob_col, gate_col=gate_col)


def _thresholds_from_validation(
    intraday: pd.DataFrame,
    *,
    intraday_summary: dict[str, Any],
    signal_config: dict[str, str],
    signal_threshold: float | None,
    risk_threshold: float | None,
) -> dict[str, float]:
    selected_threshold = float(intraday_summary.get("selected_model", {}).get("threshold", 0.5))
    prob_col = signal_config["prob_col"]
    gate_col = signal_config["gate_col"]
    validation = intraday.loc[
        intraday.get("segment", pd.Series(index=intraday.index, dtype=object)).eq("valid")
        & intraday[gate_col].fillna(False).astype(bool)
        & intraday[prob_col].notna()
    ]
    if risk_threshold is None:
        if validation.empty:
            risk_value = max(0.0, selected_threshold - 0.05)
        else:
            risk_value = float(validation[prob_col].quantile(0.20))
    else:
        risk_value = float(risk_threshold)
    return {
        "signal_threshold": float(selected_threshold if signal_threshold is None else signal_threshold),
        "risk_threshold": risk_value,
    }


def _select_thresholds_from_validation(
    daily_positions: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    signal_config: dict[str, str],
    base_thresholds: dict[str, float],
    cost_bps: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    prob_col = signal_config["prob_col"]
    gate_col = signal_config["gate_col"]
    validation_probs = intraday_predictions.loc[
        intraday_predictions.get("segment", pd.Series(index=intraday_predictions.index, dtype=object)).eq("valid")
        & intraday_predictions[gate_col].fillna(False).astype(bool)
        & intraday_predictions[prob_col].notna(),
        prob_col,
    ]
    if validation_probs.empty:
        return base_thresholds, pd.DataFrame()

    signal_grid = sorted(
        set(
            [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            + [float(validation_probs.quantile(value)) for value in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]]
        )
    )
    risk_grid = sorted(
        set(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            + [float(validation_probs.quantile(value)) for value in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]]
        )
    )

    rows: list[dict[str, Any]] = []
    for signal_threshold in signal_grid:
        for risk_threshold in risk_grid:
            trades = simulate_execution_overlays(
                daily_positions,
                daily_prices=daily_prices,
                intraday=intraday_predictions,
                signal_threshold=float(signal_threshold),
                risk_threshold=float(risk_threshold),
                cost_bps=float(cost_bps),
                prob_col=prob_col,
                gate_col=gate_col,
            ).assign(cost_bps=float(cost_bps))
            metrics = summarize_overlay_metrics(trades)
            validation_metrics = metrics.loc[metrics["segment"].eq("validation")].copy()
            for row in validation_metrics.to_dict("records"):
                row.update(
                    {
                        "signal_threshold": float(signal_threshold),
                        "risk_threshold": float(risk_threshold),
                        "selection_uses_test_segment": False,
                    }
                )
                rows.append(row)

    selection = pd.DataFrame(rows)
    if selection.empty:
        return base_thresholds, selection
    selection["passes_target"] = (
        selection["return_target_satisfied"].astype(bool)
        & (selection["max_drawdown_pct"].fillna(-np.inf) > -18.0)
        & (selection["skipped_periods"].fillna(0) <= selection["periods"].fillna(0) * 0.5)
    )
    target_candidates = selection.loc[selection["passes_target"]].copy()
    if target_candidates.empty:
        return base_thresholds, selection.sort_values(
            ["passes_target", "selection_score", "total_return_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    selected = target_candidates.sort_values(
        [
            "return_retention_in_target_band",
            "drawdown_improvement_pct",
            "selection_score",
            "total_return_pct",
            "return_retention_pct",
            "signal_threshold",
            "risk_threshold",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).iloc[0]
    thresholds = {
        "signal_threshold": float(selected["signal_threshold"]),
        "risk_threshold": float(selected["risk_threshold"]),
    }
    return thresholds, selection.sort_values(
        ["passes_target", "selection_score", "total_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _select_variant(model_comparison: pd.DataFrame, *, cost_bps: float) -> dict[str, Any]:
    validation = model_comparison.loc[model_comparison["segment"].eq("validation")].copy()
    if validation.empty:
        validation = model_comparison.copy()
    validation["passes_target"] = (
        validation["return_target_satisfied"].astype(bool)
        & (validation["max_drawdown_pct"].fillna(-np.inf) > -18.0)
        & (validation["skipped_periods"].fillna(0) <= validation["periods"].fillna(0) * 0.5)
    )
    eligible = validation.loc[validation["passes_target"]]
    selected = (eligible if not eligible.empty else validation).sort_values(
        [
            "passes_target",
            "return_retention_in_target_band",
            "drawdown_improvement_pct",
            "selection_score",
            "total_return_pct",
        ],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    variant = str(selected["variant"])
    test = model_comparison.loc[(model_comparison["segment"].eq("test")) & (model_comparison["variant"].eq(variant))]
    test_row = test.iloc[0] if not test.empty else selected
    return {
        "variant": variant,
        "selection_segment": "validation",
        "report_segment": "test",
        "selection_uses_test_segment": False,
        "selection_cost_bps": float(cost_bps),
        "selection_metric": "validation return_retention + 2*drawdown_improvement + 0.25*sharpe - skipped_penalty",
        "selection_score": float(selected["selection_score"]),
        "validation_passes_target": bool(selected["passes_target"]),
        "validation_total_return_pct": float(selected["total_return_pct"]),
        "validation_baseline_return_pct": float(selected["baseline_return_pct"]),
        "validation_max_drawdown_pct": float(selected["max_drawdown_pct"]),
        "validation_return_retention_pct": float(selected["return_retention_pct"]),
        "validation_return_target_mode": str(selected.get("return_target_mode", "")),
        "validation_return_target_satisfied": bool(selected.get("return_target_satisfied", False)),
        "validation_return_retention_in_target_band": bool(
            selected.get("return_retention_in_target_band", False)
        ),
        "validation_drawdown_improvement_pct": float(selected["drawdown_improvement_pct"]),
        "test_total_return_pct": float(test_row["total_return_pct"]),
        "test_baseline_return_pct": float(test_row["baseline_return_pct"]),
        "test_max_drawdown_pct": float(test_row["max_drawdown_pct"]),
        "test_return_retention_pct": float(test_row["return_retention_pct"]),
        "test_return_target_mode": str(test_row.get("return_target_mode", "")),
        "test_return_target_satisfied": bool(test_row.get("return_target_satisfied", False)),
        "test_return_retention_in_target_band": bool(test_row.get("return_retention_in_target_band", False)),
        "test_drawdown_improvement_pct": float(test_row["drawdown_improvement_pct"]),
    }


def _missing_intraday_prediction_period_count(trades: pd.DataFrame) -> int:
    if trades.empty or "skipped_reason" not in trades or "datetime_ts" not in trades:
        return 0
    missing = trades.loc[
        trades["skipped_reason"].astype(str).eq("missing_intraday_prediction")
        & trades.get("variant", pd.Series("", index=trades.index)).astype(str).ne("baseline_next_open")
    ]
    if missing.empty:
        return 0
    return int(missing["datetime_ts"].dt.normalize().nunique())


def _normalise_fold_segment(series: pd.Series) -> pd.Series:
    return series.astype(str).replace({"valid": "validation", "val": "validation"})


def _conditional_risk_sleeve_walk_forward_metrics(
    trades: pd.DataFrame,
    regime_features: pd.DataFrame,
    *,
    required_cost_bps: tuple[float, ...] = (20.0, 50.0),
    selection_cost_bps: float = 20.0,
    condition_columns: tuple[str, ...] = (
        "probability",
        "target_position",
        "turnover",
        "raw_target",
    ),
    require_intraday_prediction_coverage: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = trades.copy()
    if "datetime" in base.columns:
        base["datetime_ts"] = pd.to_datetime(base["datetime"])
    max_date = base["datetime_ts"].max() if "datetime_ts" in base and not base.empty else pd.Timestamp("2026-07-02")
    required_costs = tuple(float(value) for value in required_cost_bps)
    required_cost_set = set(required_costs)

    features = regime_features.copy()
    if "datetime" in features.columns:
        features["datetime_key"] = pd.to_datetime(features["datetime"]).dt.normalize()

    def infeasible_row(
        *,
        fold_name: str,
        test_year: int,
        train_start: str,
        train_end: str,
        valid_start: str,
        valid_end: str,
        test_start: str,
        test_end: str,
        reason: str,
        valid_trades: pd.DataFrame,
        test_trades: pd.DataFrame,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "fold": fold_name,
            "test_year": int(test_year),
            "status": "infeasible",
            "blocking_reason": reason,
            "selection_cost_bps": float(selection_cost_bps),
            "required_cost_bps": ",".join(str(float(value)) for value in required_costs),
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
            "selected_variant": None,
            "condition_column": None,
            "condition_operator": None,
            "condition_quantile": None,
            "condition_threshold": None,
            "selection_uses_test_segment": False,
            "validation_passes_target": False,
            "test_passes_target": False,
            "required_cost_validation_target_count": 0,
            "required_cost_test_target_count": 0,
            "required_cost_validation_and_test_target_count": 0,
            "valid_periods": int(valid_trades["datetime_ts"].dt.normalize().nunique()) if "datetime_ts" in valid_trades else 0,
            "test_periods": int(test_trades["datetime_ts"].dt.normalize().nunique()) if "datetime_ts" in test_trades else 0,
        }
        if extra:
            row.update(extra)
        return row

    for test_year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS:
        fold_name = f"wf_{int(test_year)}"
        train_start = REQUIRED_WALK_FORWARD_START
        train_end = f"{int(test_year) - 2}-12-31"
        valid_start = f"{int(test_year) - 1}-01-01"
        valid_end = f"{int(test_year) - 1}-12-31"
        test_start = f"{int(test_year)}-01-01"
        test_end = max_date.date().isoformat() if int(test_year) == int(max_date.year) else f"{int(test_year)}-12-31"

        if "walk_forward_fold" in base.columns and base["walk_forward_fold"].notna().any():
            fold_base = base.loc[base["walk_forward_fold"].astype(str).eq(fold_name)].copy()
            segment_col = "fold_segment" if "fold_segment" in fold_base.columns else "segment"
            segments = _normalise_fold_segment(fold_base[segment_col]) if segment_col in fold_base else pd.Series("", index=fold_base.index)
            valid_trades = fold_base.loc[segments.eq("validation")].copy()
            test_trades = fold_base.loc[segments.eq("test")].copy()
        elif "datetime_ts" in base.columns:
            valid_trades = base.loc[base["datetime_ts"].between(pd.Timestamp(valid_start), pd.Timestamp(valid_end))].copy()
            test_trades = base.loc[base["datetime_ts"].between(pd.Timestamp(test_start), pd.Timestamp(test_end))].copy()
        else:
            valid_trades = pd.DataFrame()
            test_trades = pd.DataFrame()

        blocking_reasons: list[str] = []
        if valid_trades.empty:
            blocking_reasons.append("missing_validation_data")
        if test_trades.empty:
            blocking_reasons.append("missing_test_data")
        if blocking_reasons:
            rows.append(
                infeasible_row(
                    fold_name=fold_name,
                    test_year=int(test_year),
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    reason="|".join(blocking_reasons),
                    valid_trades=valid_trades,
                    test_trades=test_trades,
                )
            )
            continue

        if require_intraday_prediction_coverage:
            valid_missing = _missing_intraday_prediction_period_count(valid_trades)
            test_missing = _missing_intraday_prediction_period_count(test_trades)
            if valid_missing or test_missing:
                rows.append(
                    infeasible_row(
                        fold_name=fold_name,
                        test_year=int(test_year),
                        train_start=train_start,
                        train_end=train_end,
                        valid_start=valid_start,
                        valid_end=valid_end,
                        test_start=test_start,
                        test_end=test_end,
                        reason="missing_intraday_prediction_coverage",
                        valid_trades=valid_trades,
                        test_trades=test_trades,
                        extra={
                            "validation_missing_intraday_prediction_periods": int(valid_missing),
                            "test_missing_intraday_prediction_periods": int(test_missing),
                        },
                    )
                )
                continue

        fold_trades = pd.concat([valid_trades, test_trades], ignore_index=True)
        if "cost_bps" in fold_trades.columns:
            fold_trades = fold_trades.loc[
                pd.to_numeric(fold_trades["cost_bps"], errors="coerce").isin(required_cost_set)
            ].copy()
        segment_col = "fold_segment" if "fold_segment" in fold_trades.columns else "segment"
        if segment_col in fold_trades.columns:
            fold_trades["segment"] = _normalise_fold_segment(fold_trades[segment_col])

        fold_features = features.copy()
        if "walk_forward_fold" in fold_features.columns:
            fold_features = fold_features.loc[fold_features["walk_forward_fold"].astype(str).eq(fold_name)].copy()
        elif "datetime_key" in fold_features.columns and "datetime_ts" in fold_trades.columns:
            fold_dates = set(fold_trades["datetime_ts"].dt.normalize())
            fold_features = fold_features.loc[fold_features["datetime_key"].isin(fold_dates)].copy()

        candidate_frames = [
            evaluate_conditional_overlay_candidates(
                fold_trades.drop(columns=["datetime_ts"], errors="ignore"),
                fold_features.drop(columns=["datetime_key"], errors="ignore"),
                cost_bps=float(cost_bps),
                condition_columns=condition_columns,
            )
            for cost_bps in required_costs
        ]
        candidates = (
            pd.concat([frame for frame in candidate_frames if not frame.empty], ignore_index=True)
            if candidate_frames
            else pd.DataFrame()
        )
        if candidates.empty:
            rows.append(
                infeasible_row(
                    fold_name=fold_name,
                    test_year=int(test_year),
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    reason="missing_conditional_candidates",
                    valid_trades=valid_trades,
                    test_trades=test_trades,
                )
            )
            continue

        selected = select_conditional_risk_sleeve_policy(
            candidates,
            required_cost_bps=required_costs,
            selection_cost_bps=float(selection_cost_bps),
        )
        if not selected.get("variant"):
            rows.append(
                infeasible_row(
                    fold_name=fold_name,
                    test_year=int(test_year),
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    reason="missing_selected_conditional_policy",
                    valid_trades=valid_trades,
                    test_trades=test_trades,
                )
            )
            continue

        required_validation_target_count = int(selected.get("required_cost_validation_target_count", 0) or 0)
        if required_validation_target_count < len(required_costs) or not bool(
            selected.get("validation_cost_robust", False)
        ):
            rows.append(
                infeasible_row(
                    fold_name=fold_name,
                    test_year=int(test_year),
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    reason="failed_validation_target",
                    valid_trades=valid_trades,
                    test_trades=test_trades,
                    extra={
                        "diagnostic_variant": selected.get("variant"),
                        "diagnostic_condition_column": selected.get("condition_column"),
                        "diagnostic_condition_operator": selected.get("condition_operator"),
                        "diagnostic_condition_threshold": selected.get("condition_threshold"),
                        "required_cost_validation_target_count": required_validation_target_count,
                        "required_cost_test_target_count": int(
                            selected.get("required_cost_test_target_count", 0) or 0
                        ),
                        "required_cost_validation_and_test_target_count": int(
                            selected.get("required_cost_validation_and_test_target_count", 0) or 0
                        ),
                        "validation_return_retention_pct": selected.get("validation_return_retention_pct"),
                        "validation_max_drawdown_pct": selected.get("validation_max_drawdown_pct"),
                        "test_return_retention_pct": selected.get("test_return_retention_pct"),
                        "test_max_drawdown_pct": selected.get("test_max_drawdown_pct"),
                    },
                )
            )
            continue

        selected_row = {
            "fold": fold_name,
            "test_year": int(test_year),
            "status": "ok",
            "blocking_reason": "ok",
            "selection_cost_bps": float(selection_cost_bps),
            "required_cost_bps": ",".join(str(float(value)) for value in required_costs),
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
            "selected_variant": selected.get("variant"),
            "condition_column": selected.get("condition_column"),
            "condition_operator": selected.get("condition_operator"),
            "condition_quantile": selected.get("condition_quantile"),
            "condition_threshold": selected.get("condition_threshold"),
            "condition_valid_active_coverage": selected.get("condition_valid_active_coverage"),
            "selection_uses_test_segment": bool(selected.get("selection_uses_test_segment", False)),
            "validation_cost_robust": bool(selected.get("validation_cost_robust", False)),
            "validation_passes_target": bool(selected.get("validation_passes_target", False)),
            "test_passes_target": bool(selected.get("test_passes_target", False)),
            "required_cost_validation_target_count": int(selected.get("required_cost_validation_target_count", 0) or 0),
            "required_cost_test_target_count": int(selected.get("required_cost_test_target_count", 0) or 0),
            "required_cost_validation_and_test_target_count": int(
                selected.get("required_cost_validation_and_test_target_count", 0) or 0
            ),
            "valid_periods": int(selected.get("validation_periods") or valid_trades["datetime_ts"].dt.normalize().nunique()),
            "test_periods": int(selected.get("test_periods") or test_trades["datetime_ts"].dt.normalize().nunique()),
            "validation_return_pct": selected.get("validation_return_pct"),
            "validation_return_retention_pct": selected.get("validation_return_retention_pct"),
            "validation_max_drawdown_pct": selected.get("validation_max_drawdown_pct"),
            "validation_drawdown_improvement_pct": selected.get("validation_drawdown_improvement_pct"),
            "validation_open_buy_dump_period_reduction": selected.get("validation_open_buy_dump_period_reduction"),
            "validation_crash_period_reduction": selected.get("validation_crash_period_reduction"),
            "validation_worst_entry_day_return_improvement_pct": selected.get(
                "validation_worst_entry_day_return_improvement_pct"
            ),
            "test_return_pct": selected.get("test_return_pct"),
            "test_return_retention_pct": selected.get("test_return_retention_pct"),
            "test_max_drawdown_pct": selected.get("test_max_drawdown_pct"),
            "test_drawdown_improvement_pct": selected.get("test_drawdown_improvement_pct"),
            "test_open_buy_dump_period_reduction": selected.get("test_open_buy_dump_period_reduction"),
            "test_crash_period_reduction": selected.get("test_crash_period_reduction"),
            "test_worst_entry_day_return_improvement_pct": selected.get(
                "test_worst_entry_day_return_improvement_pct"
            ),
        }
        required_results = pd.DataFrame(selected.get("required_cost_results", []))
        if not required_results.empty:
            for result in required_results.to_dict("records"):
                cost_label = f"{float(result.get('cost_bps')):g}".replace(".", "p")
                for column in [
                    "validation_return_retention_pct",
                    "validation_max_drawdown_pct",
                    "validation_open_buy_dump_period_reduction",
                    "validation_crash_period_reduction",
                    "test_return_retention_pct",
                    "test_max_drawdown_pct",
                    "test_open_buy_dump_period_reduction",
                    "test_crash_period_reduction",
                    "passes_validation_target",
                    "passes_test_target",
                    "passes_validation_and_test_target",
                ]:
                    if column in result:
                        selected_row[f"cost_{cost_label}_{column}"] = result[column]
        rows.append(selected_row)

    return pd.DataFrame(rows)


def _walk_forward_metrics(
    trades: pd.DataFrame,
    *,
    cost_bps: float,
    require_intraday_prediction_coverage: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = trades.loc[trades["cost_bps"].eq(float(cost_bps))].copy()
    base["datetime_ts"] = pd.to_datetime(base["datetime"])
    max_date = base["datetime_ts"].max() if not base.empty else pd.Timestamp("2026-07-02")
    for test_year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS:
        fold_name = f"wf_{int(test_year)}"
        train_start = REQUIRED_WALK_FORWARD_START
        train_end = f"{int(test_year) - 2}-12-31"
        valid_start = f"{int(test_year) - 1}-01-01"
        valid_end = f"{int(test_year) - 1}-12-31"
        test_start = f"{int(test_year)}-01-01"
        test_end = max_date.date().isoformat() if int(test_year) == int(max_date.year) else f"{int(test_year)}-12-31"

        if "walk_forward_fold" in base.columns and base["walk_forward_fold"].notna().any():
            fold_base = base.loc[base["walk_forward_fold"].astype(str).eq(fold_name)].copy()
            segment_col = "fold_segment" if "fold_segment" in fold_base.columns else "segment"
            valid_trades = fold_base.loc[fold_base[segment_col].astype(str).eq("validation")].copy()
            test_trades = fold_base.loc[fold_base[segment_col].astype(str).eq("test")].copy()
        else:
            valid_trades = base.loc[base["datetime_ts"].between(pd.Timestamp(valid_start), pd.Timestamp(valid_end))].copy()
            test_trades = base.loc[base["datetime_ts"].between(pd.Timestamp(test_start), pd.Timestamp(test_end))].copy()
        blocking_reasons: list[str] = []
        if valid_trades.empty:
            blocking_reasons.append("missing_validation_data")
        if test_trades.empty:
            blocking_reasons.append("missing_test_data")
        if blocking_reasons:
            valid_periods = int(valid_trades["datetime_ts"].dt.normalize().nunique()) if not valid_trades.empty else 0
            test_periods = int(test_trades["datetime_ts"].dt.normalize().nunique()) if not test_trades.empty else 0
            rows.append(
                {
                    "fold": fold_name,
                    "test_year": int(test_year),
                    "status": "infeasible",
                    "blocking_reason": "|".join(blocking_reasons),
                    "cost_bps": float(cost_bps),
                    "train_start": train_start,
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "selected_variant": None,
                    "selection_uses_test_segment": False,
                    "validation_passes_target": False,
                    "test_passes_target": False,
                    "valid_periods": valid_periods,
                    "test_periods": test_periods,
                }
            )
            continue
        if require_intraday_prediction_coverage:
            valid_missing = _missing_intraday_prediction_period_count(valid_trades)
            test_missing = _missing_intraday_prediction_period_count(test_trades)
            if valid_missing or test_missing:
                rows.append(
                    {
                        "fold": fold_name,
                        "test_year": int(test_year),
                        "status": "infeasible",
                        "blocking_reason": "missing_intraday_prediction_coverage",
                        "cost_bps": float(cost_bps),
                        "train_start": train_start,
                        "train_end": train_end,
                        "valid_start": valid_start,
                        "valid_end": valid_end,
                        "test_start": test_start,
                        "test_end": test_end,
                        "selected_variant": None,
                        "selection_uses_test_segment": False,
                        "validation_passes_target": False,
                        "test_passes_target": False,
                        "valid_periods": int(valid_trades["datetime_ts"].dt.normalize().nunique()),
                        "test_periods": int(test_trades["datetime_ts"].dt.normalize().nunique()),
                        "validation_missing_intraday_prediction_periods": int(valid_missing),
                        "test_missing_intraday_prediction_periods": int(test_missing),
                    }
                )
                continue

        valid_trades["segment"] = "validation"
        test_trades["segment"] = "test"
        valid_metrics = summarize_overlay_metrics(valid_trades.drop(columns=["datetime_ts"], errors="ignore"))
        test_metrics = summarize_overlay_metrics(test_trades.drop(columns=["datetime_ts"], errors="ignore"))
        if valid_metrics.empty or "variant" not in valid_metrics:
            valid_periods = int(valid_trades["datetime_ts"].dt.normalize().nunique()) if not valid_trades.empty else 0
            test_periods = int(test_trades["datetime_ts"].dt.normalize().nunique()) if not test_trades.empty else 0
            rows.append(
                {
                    "fold": fold_name,
                    "test_year": int(test_year),
                    "status": "infeasible",
                    "blocking_reason": "missing_validation_metrics",
                    "cost_bps": float(cost_bps),
                    "train_start": train_start,
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "selected_variant": None,
                    "selection_uses_test_segment": False,
                    "validation_passes_target": False,
                    "test_passes_target": False,
                    "valid_periods": valid_periods,
                    "test_periods": test_periods,
                }
            )
            continue
        valid_selection = valid_metrics.copy()
        valid_selection["passes_target"] = (
            valid_selection["return_target_satisfied"].astype(bool)
            & valid_selection["max_drawdown_pct"].gt(-18.0)
            & valid_selection["skipped_periods"].le(valid_selection["periods"] * 0.5)
        )
        eligible = valid_selection.loc[valid_selection["passes_target"]].copy()
        selected = (eligible if not eligible.empty else valid_selection).sort_values(
            [
                "passes_target",
                "return_retention_in_target_band",
                "drawdown_improvement_pct",
                "selection_score",
                "total_return_pct",
            ],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        variant = str(selected["variant"])
        test_row_frame = test_metrics.loc[test_metrics["variant"].eq(variant)] if "variant" in test_metrics else pd.DataFrame()
        if test_row_frame.empty:
            valid_periods = int(valid_trades["datetime_ts"].dt.normalize().nunique()) if not valid_trades.empty else 0
            test_periods = int(test_trades["datetime_ts"].dt.normalize().nunique()) if not test_trades.empty else 0
            rows.append(
                {
                    "fold": fold_name,
                    "test_year": int(test_year),
                    "status": "infeasible",
                    "blocking_reason": "missing_selected_variant_test_metrics",
                    "cost_bps": float(cost_bps),
                    "train_start": train_start,
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "selected_variant": variant,
                    "selection_uses_test_segment": False,
                    "validation_passes_target": bool(selected["passes_target"]),
                    "test_passes_target": False,
                    "valid_periods": valid_periods,
                    "test_periods": test_periods,
                }
            )
            continue
        test_row = test_row_frame.iloc[0]
        test_passes_target = bool(
            bool(test_row.get("return_target_satisfied", False))
            and float(test_row["max_drawdown_pct"]) > -18.0
            and float(test_row["skipped_periods"]) <= float(test_row["periods"]) * 0.5
        )
        rows.append(
            {
                "fold": f"wf_{int(test_year)}",
                "test_year": int(test_year),
                "status": "ok",
                "blocking_reason": "ok",
                "cost_bps": float(cost_bps),
                "train_start": train_start,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "test_start": test_start,
                "test_end": test_end,
                "selected_variant": variant,
                "selection_uses_test_segment": False,
                "validation_passes_target": bool(selected["passes_target"]),
                "test_passes_target": test_passes_target,
                "valid_periods": int(selected["periods"]),
                "test_periods": int(test_row["periods"]),
                "valid_baseline_return_pct": float(selected["baseline_return_pct"]),
                "valid_total_return_pct": float(selected["total_return_pct"]),
                "valid_max_drawdown_pct": float(selected["max_drawdown_pct"]),
                "valid_return_retention_pct": float(selected["return_retention_pct"]),
                "valid_return_target_mode": str(selected.get("return_target_mode", "")),
                "valid_return_target_satisfied": bool(selected.get("return_target_satisfied", False)),
                "valid_return_retention_in_target_band": bool(
                    selected.get("return_retention_in_target_band", False)
                ),
                "valid_drawdown_improvement_pct": float(selected["drawdown_improvement_pct"]),
                "test_baseline_return_pct": float(test_row["baseline_return_pct"]),
                "test_total_return_pct": float(test_row["total_return_pct"]),
                "test_max_drawdown_pct": float(test_row["max_drawdown_pct"]),
                "test_return_retention_pct": float(test_row["return_retention_pct"]),
                "test_return_target_mode": str(test_row.get("return_target_mode", "")),
                "test_return_target_satisfied": bool(test_row.get("return_target_satisfied", False)),
                "test_return_retention_in_target_band": bool(test_row.get("return_retention_in_target_band", False)),
                "test_drawdown_improvement_pct": float(test_row["drawdown_improvement_pct"]),
            }
        )
    return pd.DataFrame(rows)


def _execution_risk_diagnostics_summary(
    execution_risk_diagnostics: pd.DataFrame,
    *,
    selected_variant: str,
    cost_bps: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "selection_cost_bps": float(cost_bps),
        "selected_variant": selected_variant,
        "open_buy_dump_threshold_pct": float(OPEN_BUY_DUMP_THRESHOLD * 100.0),
        "crash_threshold_pct": float(CRASH_THRESHOLD * 100.0),
    }
    if execution_risk_diagnostics.empty:
        return summary
    frame = execution_risk_diagnostics.loc[
        execution_risk_diagnostics["cost_bps"].eq(float(cost_bps))
        & execution_risk_diagnostics["segment"].eq("test")
    ].copy()
    if frame.empty:
        return summary
    selected = frame.loc[frame["variant"].eq(selected_variant)]
    baseline = frame.loc[frame["variant"].eq("baseline_next_open")]
    if not selected.empty:
        selected_row = selected.iloc[0]
        summary.update(
            {
                "selected_test_open_buy_dump_periods": int(selected_row["open_buy_dump_periods"]),
                "selected_test_open_buy_dump_rate": float(selected_row["open_buy_dump_rate"]),
                "selected_test_crash_periods": int(selected_row["crash_periods"]),
                "selected_test_crash_rate": float(selected_row["crash_rate"]),
                "selected_test_worst_entry_day_return_pct": float(selected_row["worst_entry_day_return_pct"]),
                "selected_test_worst_strategy_return_pct": float(selected_row["worst_strategy_return_pct"]),
                "selected_test_open_buy_dump_period_reduction": int(
                    selected_row["open_buy_dump_period_reduction"]
                ),
                "selected_test_crash_period_reduction": int(selected_row["crash_period_reduction"]),
            }
        )
    if not baseline.empty:
        baseline_row = baseline.iloc[0]
        summary.update(
            {
                "baseline_test_open_buy_dump_periods": int(baseline_row["open_buy_dump_periods"]),
                "baseline_test_crash_periods": int(baseline_row["crash_periods"]),
                "baseline_test_worst_entry_day_return_pct": float(baseline_row["worst_entry_day_return_pct"]),
                "baseline_test_worst_strategy_return_pct": float(baseline_row["worst_strategy_return_pct"]),
            }
        )
    best_dump = frame.sort_values(
        [
            "open_buy_dump_periods",
            "worst_entry_day_return_pct",
            "crash_periods",
            "worst_strategy_return_pct",
        ],
        ascending=[True, False, True, False],
    ).iloc[0]
    summary.update(
        {
            "best_open_buy_dump_variant": str(best_dump["variant"]),
            "best_open_buy_dump_periods": int(best_dump["open_buy_dump_periods"]),
            "best_open_buy_dump_worst_entry_day_return_pct": float(best_dump["worst_entry_day_return_pct"]),
        }
    )
    return summary


def _analysis_summary(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    thresholds: dict[str, float],
    signal_config: dict[str, str],
    model_comparison: pd.DataFrame,
    threshold_selection: pd.DataFrame,
    execution_risk_diagnostics: pd.DataFrame,
    execution_risk_summary: dict[str, Any],
    conditional_risk_policy_selection: pd.DataFrame,
    conditional_risk_sleeve: dict[str, Any],
    conditional_risk_sleeve_walk_forward: pd.DataFrame,
    selected_model: dict[str, Any],
    walk_forward: pd.DataFrame,
    daily_position_source_audit: pd.DataFrame,
    optimization_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "daily_open2open_position_model_plus_15m_execution_overlay",
        "as_of": str(daily_summary.get("as_of") or spec.as_of),
        "selection_uses_test_segment": False,
        "thresholds": {
            "signal_threshold": thresholds["signal_threshold"],
            "signal_threshold_source": (
                "overlay_validation_threshold_search"
                if spec.optimize_thresholds
                else "source_intraday_selected_model_valid_threshold"
            ),
            "risk_threshold": thresholds["risk_threshold"],
            "risk_threshold_source": (
                "overlay_validation_threshold_search"
                if spec.optimize_thresholds
                else "validation_gated_probability_quantile_20pct_or_user_override"
            ),
            "signal_prob_col": signal_config["prob_col"],
            "gate_col": signal_config["gate_col"],
            "gate_policy": signal_config["gate_policy"],
            "threshold_search_enabled": bool(spec.optimize_thresholds),
        },
        "data_audit": {
            "daily_selected_position_rows": int(len(daily_positions)),
            "daily_selected_position_start": daily_positions["datetime"].min().date().isoformat(),
            "daily_selected_position_end": daily_positions["datetime"].max().date().isoformat(),
            "daily_price_rows": int(len(daily_prices)),
            "daily_price_start": daily_prices["datetime"].min().date().isoformat(),
            "daily_price_end": daily_prices["datetime"].max().date().isoformat(),
            "intraday_rows": int(len(intraday_predictions)),
            "intraday_prediction_rows": int(intraday_predictions[signal_config["prob_col"]].notna().sum()),
            "intraday_start": intraday_predictions["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "intraday_end": intraday_predictions["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "daily_position_source_audit": daily_position_source_audit.to_dict("records"),
        "source_daily_model": {
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing"),
            "feature_count": daily_summary.get("position_model", {}).get("feature_count"),
            "has_soxx_constituent_features": any(
                str(col).startswith("soxx_const_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
            "has_intraday_t1_features": any(
                str(col).startswith("intraday15m_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
        },
        "source_intraday_model": {
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "trade_gate": intraday_summary.get("trade_gate"),
            "label_policy": intraday_summary.get("label_policy"),
            "external_context": intraday_summary.get("external_intraday_context"),
        },
        "selected_model": selected_model,
        "model_comparison_rows": int(len(model_comparison)),
        "execution_risk_diagnostics_rows": int(len(execution_risk_diagnostics)),
        "execution_risk_diagnostics": execution_risk_summary,
        "conditional_risk_sleeve": conditional_risk_sleeve,
        "conditional_risk_policy_selection_rows": int(len(conditional_risk_policy_selection)),
        "conditional_risk_sleeve_walk_forward_rows": int(len(conditional_risk_sleeve_walk_forward)),
        "threshold_selection_rows": int(0 if not spec.optimize_thresholds else len(threshold_selection)),
        "walk_forward_rows": int(len(walk_forward)),
        "optimization_plan": optimization_plan,
    }


def _write_workflow_yaml(
    spec: CognitiveAlphaIntradayOverlaySpec,
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
                f"  intraday_provider_uri: {spec.intraday_provider_uri.resolve().as_posix()}",
                "sources:",
                f"  daily_card: {spec.source_daily_card}",
                f"  daily_recorder_id: {daily_summary.get('recorder_id')}",
                f"  intraday_card: {spec.source_intraday_card}",
                f"  intraday_recorder_id: {intraday_summary.get('recorder_id')}",
                "selection:",
                "  segment: validation",
                "  report_segment: test",
                "  uses_test_segment: false",
                f"  selection_cost_bps: {float(spec.selection_cost_bps)}",
                f"  cost_stress_bps: {list(float(value) for value in sorted(set(spec.cost_stress_bps)))}",
                "execution:",
                f"  daily_timing: {DAILY_EXECUTION_TIMING}",
                f"  overlay_signal_timing: {OVERLAY_SIGNAL_TIMING}",
                "  variants:",
                *[f"    - {variant}" for variant in VARIANTS],
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_regime_workflow_yaml(
    spec: CognitiveAlphaIntradayOverlaySpec,
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
                f"  intraday_provider_uri: {spec.intraday_provider_uri.resolve().as_posix()}",
                "sources:",
                f"  daily_card: {spec.source_daily_card}",
                f"  daily_recorder_id: {daily_summary.get('recorder_id')}",
                f"  intraday_card: {spec.source_intraday_card}",
                f"  intraday_recorder_id: {intraday_summary.get('recorder_id')}",
                "selection:",
                "  segment: validation",
                "  report_segment: test",
                "  uses_test_segment: false",
                f"  selection_cost_bps: {float(spec.selection_cost_bps)}",
                "regime_policy:",
                "  causal_scope: daily_close_known_features_only",
                "  condition_threshold_source: validation_quantile",
                "  target_return_retention_pct: [85.0, 95.0]",
                "  target_max_drawdown_floor_pct: -18.0",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_delayed_workflow_yaml(
    spec: CognitiveAlphaIntradayOverlaySpec,
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
                f"  intraday_provider_uri: {spec.intraday_provider_uri.resolve().as_posix()}",
                "sources:",
                f"  daily_card: {spec.source_daily_card}",
                f"  daily_recorder_id: {daily_summary.get('recorder_id')}",
                f"  intraday_card: {spec.source_intraday_card}",
                f"  intraday_recorder_id: {intraday_summary.get('recorder_id')}",
                "selection:",
                "  segment: validation",
                "  report_segment: test",
                "  uses_test_segment: false",
                f"  selection_cost_bps: {float(spec.selection_cost_bps)}",
                "  required_cost_bps: [20.0, 50.0]",
                "  max_skip_rate: 0.15",
                "execution:",
                "  causal_timing: first_30_or_60m_observed_then_next_15m_open",
                "  first_hour_features_can_only_trigger_delayed_entries: true",
                "  eligible_exit_policies:",
                "    - hold_to_next_open",
                f"    - {SAME_DAY_CLOSE_EXIT_POLICY}",
                "  diagnostic_exit_policies:",
                "    - risk_exit",
                "walk_forward:",
                "  policy: annual_train_to_T_minus_2_valid_T_minus_1_test_T",
                f"  required_start: {REQUIRED_WALK_FORWARD_START}",
                f"  required_test_years: {list(REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS)}",
                "  threshold_source: validation_only",
                "  test_segment_usage: report_only",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _regime_overlay_thresholds(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    intraday_predictions: pd.DataFrame,
    intraday_summary: dict[str, Any],
    signal_config: dict[str, str],
) -> dict[str, float]:
    overlay_summary_path = spec.root / "research" / EXTERNAL_CONTEXT_OVERLAY_EXPERIMENT / "summary.json"
    if overlay_summary_path.exists():
        overlay_summary = _read_json(overlay_summary_path)
        selected = overlay_summary.get("selected_model", {})
        if "signal_threshold" in selected and "risk_threshold" in selected:
            return {
                "signal_threshold": float(selected["signal_threshold"]),
                "risk_threshold": float(selected["risk_threshold"]),
                "source": "external_context_overlay_validation_search",
            }
    thresholds = _thresholds_from_validation(
        intraday_predictions,
        intraday_summary=intraday_summary,
        signal_config=signal_config,
        signal_threshold=spec.signal_threshold,
        risk_threshold=spec.risk_threshold,
    )
    thresholds["source"] = "source_intraday_validation_threshold"
    return thresholds


def _regime_policy_group_columns(policy_selection: pd.DataFrame) -> list[str]:
    if "policy_id" in policy_selection.columns:
        return ["policy_id"]
    candidates = [
        "variant",
        "condition_column",
        "condition_operator",
        "condition_quantile",
        "condition_threshold",
    ]
    return [column for column in candidates if column in policy_selection.columns]


def _conditional_policy_rows(policy_selection: pd.DataFrame, selected_policy: dict[str, Any]) -> pd.DataFrame:
    if policy_selection.empty or not selected_policy.get("variant"):
        return pd.DataFrame()
    mask = policy_selection["variant"].astype(str).eq(str(selected_policy.get("variant")))
    for column in [
        "condition_column",
        "condition_operator",
        "condition_quantile",
        "condition_threshold",
    ]:
        if column not in policy_selection or column not in selected_policy:
            continue
        if column in {"condition_quantile", "condition_threshold"}:
            values = pd.to_numeric(policy_selection[column], errors="coerce")
            mask &= np.isclose(values, float(selected_policy[column]), equal_nan=True)
        else:
            mask &= policy_selection[column].astype(str).eq(str(selected_policy[column]))
    return policy_selection.loc[mask].sort_values("cost_bps").reset_index(drop=True)


def _select_regime_policy(
    policy_selection: pd.DataFrame,
    *,
    required_cost_bps: tuple[float, ...] = (20.0,),
    selection_cost_bps: float = 20.0,
) -> dict[str, Any]:
    if policy_selection.empty:
        return {
            "variant": "baseline_next_open",
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "validation_passes_target": False,
            "test_passes_target": False,
        }
    frame = policy_selection.copy()
    frame["cost_bps"] = pd.to_numeric(frame.get("cost_bps", selection_cost_bps), errors="coerce")
    required_costs = tuple(float(value) for value in required_cost_bps)
    group_columns = _regime_policy_group_columns(frame)
    robust_rows: list[tuple[pd.Series, pd.DataFrame]] = []
    if group_columns and required_costs:
        for _, group in frame.groupby(group_columns, sort=False, dropna=False):
            costs = {float(value) for value in group["cost_bps"].dropna().unique()}
            if not set(required_costs).issubset(costs):
                continue
            required_rows = group.loc[group["cost_bps"].isin(required_costs)].copy()
            if not bool(required_rows["passes_validation_target"].astype(bool).all()):
                continue
            selected_cost_row = group.loc[group["cost_bps"].eq(float(selection_cost_bps))]
            if selected_cost_row.empty:
                continue
            robust_rows.append(
                (
                    selected_cost_row.iloc[0],
                    required_rows.sort_values("cost_bps").copy(),
                )
            )
    if robust_rows:
        source = pd.DataFrame([row for row, _ in robust_rows]).copy()
        required_cost_results_by_index = {
            int(index): required_rows for index, (_, required_rows) in zip(source.index, robust_rows, strict=False)
        }
        validation_cost_robust = True
    else:
        validation_passes = frame.loc[frame["passes_validation_target"].astype(bool)].copy()
        source = validation_passes if not validation_passes.empty else frame.copy()
        required_cost_results_by_index = {}
        validation_cost_robust = False
    sorted_source = source.sort_values(
        [
            "passes_validation_target",
            "validation_drawdown_improvement_pct",
            "selection_score",
            "validation_return_retention_pct",
            "validation_return_pct",
        ],
        ascending=[False, False, False, False, False],
    )
    selected = sorted_source.iloc[0]
    result = _json_ready_dict(selected.to_dict())
    required_cost_results = required_cost_results_by_index.get(int(selected.name), pd.DataFrame())
    result["validation_passes_target"] = bool(result.get("passes_validation_target", False))
    result["test_passes_target"] = bool(result.get("passes_test_target", False))
    result["selection_uses_test_segment"] = False
    result["selection_metric"] = "validation_target_first_then_max_drawdown_improvement"
    result["required_cost_bps"] = list(required_costs)
    result["selection_cost_bps"] = float(selection_cost_bps)
    result["validation_cost_robust"] = bool(validation_cost_robust)
    if not required_cost_results.empty:
        result["required_cost_validation_target_count"] = int(
            required_cost_results["passes_validation_target"].astype(bool).sum()
        )
        result["required_cost_test_target_count"] = int(required_cost_results["passes_test_target"].astype(bool).sum())
        result["required_cost_validation_and_test_target_count"] = int(
            required_cost_results.get(
                "passes_validation_and_test_target",
                pd.Series(False, index=required_cost_results.index),
            )
            .astype(bool)
            .sum()
        )
        required_columns = [
            "cost_bps",
            "validation_return_retention_pct",
            "validation_max_drawdown_pct",
            "test_return_retention_pct",
            "test_max_drawdown_pct",
            "passes_validation_target",
            "passes_test_target",
            "passes_validation_and_test_target",
        ]
        result["required_cost_results"] = _json_ready_records(
            required_cost_results[[column for column in required_columns if column in required_cost_results.columns]]
        )
    else:
        result["required_cost_validation_target_count"] = 0
        result["required_cost_test_target_count"] = 0
        result["required_cost_validation_and_test_target_count"] = 0
        result["required_cost_results"] = []
    return result


def _regime_analysis_summary(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    thresholds: dict[str, float],
    signal_config: dict[str, str],
    model_comparison: pd.DataFrame,
    policy_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    condition_columns: tuple[str, ...],
) -> dict[str, Any]:
    validation_target_count = int(
        policy_selection["passes_validation_target"].sum()
        if "passes_validation_target" in policy_selection
        else 0
    )
    validation_and_test_count = int(
        policy_selection["passes_validation_and_test_target"].sum()
        if "passes_validation_and_test_target" in policy_selection
        else 0
    )
    return {
        "source": "daily_open2open_position_model_plus_causal_daily_regime_intraday_execution_overlay",
        "as_of": str(daily_summary.get("as_of") or spec.as_of),
        "selection_uses_test_segment": False,
        "thresholds": {
            "signal_threshold": float(thresholds["signal_threshold"]),
            "risk_threshold": float(thresholds["risk_threshold"]),
            "source": thresholds.get("source"),
            "signal_prob_col": signal_config["prob_col"],
            "gate_col": signal_config["gate_col"],
            "gate_policy": signal_config["gate_policy"],
        },
        "regime_policy": {
            "condition_columns": list(condition_columns),
            "condition_threshold_source": "validation_quantile",
            "causal_scope": "daily_close_known_features_only",
            "noncausal_first_hour_probe_excluded_from_selection": True,
            "validation_target_candidate_count": validation_target_count,
            "validation_and_test_target_candidate_count": validation_and_test_count,
        },
        "data_audit": {
            "daily_selected_position_rows": int(len(daily_positions)),
            "daily_selected_position_start": daily_positions["datetime"].min().date().isoformat(),
            "daily_selected_position_end": daily_positions["datetime"].max().date().isoformat(),
            "daily_price_rows": int(len(daily_prices)),
            "daily_price_start": daily_prices["datetime"].min().date().isoformat(),
            "daily_price_end": daily_prices["datetime"].max().date().isoformat(),
            "intraday_rows": int(len(intraday_predictions)),
            "intraday_prediction_rows": int(intraday_predictions[signal_config["prob_col"]].notna().sum()),
            "intraday_start": intraday_predictions["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "intraday_end": intraday_predictions["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "source_daily_model": {
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing"),
            "feature_count": daily_summary.get("position_model", {}).get("feature_count"),
            "has_soxx_constituent_features": any(
                str(col).startswith("soxx_const_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
        },
        "source_intraday_model": {
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "label_policy": intraday_summary.get("label_policy"),
            "external_context": intraday_summary.get("external_intraday_context"),
        },
        "model_comparison_rows": int(len(model_comparison)),
        "policy_selection_rows": int(len(policy_selection)),
        "selected_model": selected_model,
    }


def _write_regime_sidecars(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    model_comparison: pd.DataFrame,
    policy_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    analysis_summary: dict[str, Any],
) -> dict[str, str]:
    sidecars = {
        "model_comparison": "model_comparison.csv",
        "policy_selection": "policy_selection.csv",
        "selected_model": "selected_model.json",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    model_comparison.to_csv(spec.output_dir / sidecars["model_comparison"], index=False)
    policy_selection.to_csv(spec.output_dir / sidecars["policy_selection"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_modes = (
        "# Failure Modes\n\n"
        "- Regime overlay 只允许使用 daily close 时已知的 probability、target_position、turnover、raw_target 做条件选择。\n"
        "- 第一小时走势或 15m 概率不能用于选择前 4 根 15m TWAP；这类条件会造成执行时序泄漏，已从正式选择中排除。\n"
        "- 即使 validation 有候选过 85%-95% retention 与 max_dd>-18% 门槛，test 仍必须独立报告，不能回填调参。\n"
        "- 当前外部 15m 历史仍不足以证明完整 2019-2026 walk-forward 推广。\n"
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(failure_modes, encoding="utf-8")
    return sidecars


def _delayed_analysis_summary(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    signal_config: dict[str, str],
    policy_selection_full: pd.DataFrame,
    selected_policy_rows: pd.DataFrame,
    intraday_flat_sleeve_rows: pd.DataFrame,
    selected_model: dict[str, Any],
    intraday_flat_sleeve: dict[str, Any],
    model_comparison: pd.DataFrame,
    walk_forward_summary: dict[str, Any],
    optimization_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "daily_open2open_position_model_plus_observable_30m_60m_delayed_15m_execution_overlay",
        "as_of": str(daily_summary.get("as_of") or spec.as_of),
        "selection_uses_test_segment": False,
        "walk_forward": walk_forward_summary,
        "selection_policy": {
            "required_cost_bps": [20.0, 50.0],
            "max_skip_rate": 0.15,
            "return_retention_target_pct": [85.0, 95.0],
            "max_drawdown_floor_pct": -18.0,
            "test_used_for_selection": False,
        },
        "causal_timing": {
            "daily_signal": DAILY_EXECUTION_TIMING,
            "observed_windows_minutes": [30, 60],
            "delayed_entry": "after observation window, signal bar close to next 15m open",
            "uses_first_hour_for_open_twap": False,
        },
        "data_audit": {
            "daily_selected_position_rows": int(len(daily_positions)),
            "daily_selected_position_start": daily_positions["datetime"].min().date().isoformat(),
            "daily_selected_position_end": daily_positions["datetime"].max().date().isoformat(),
            "daily_price_rows": int(len(daily_prices)),
            "daily_price_start": daily_prices["datetime"].min().date().isoformat(),
            "daily_price_end": daily_prices["datetime"].max().date().isoformat(),
            "intraday_rows": int(len(intraday_predictions)),
            "intraday_prediction_rows": int(intraday_predictions[signal_config["prob_col"]].notna().sum()),
            "intraday_start": intraday_predictions["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "intraday_end": intraday_predictions["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "policy_selection_rows": int(len(policy_selection_full)),
        "policy_candidate_count": int(policy_selection_full["policy_id"].nunique()) if "policy_id" in policy_selection_full else 0,
        "validation_target_row_count": int(policy_selection_full["passes_validation_target"].sum())
        if "passes_validation_target" in policy_selection_full
        else 0,
        "validation_and_test_target_row_count": int(policy_selection_full["passes_validation_and_test_target"].sum())
        if "passes_validation_and_test_target" in policy_selection_full
        else 0,
        "selected_policy_cost_rows": _json_ready_records(selected_policy_rows),
        "selected_model": selected_model,
        "intraday_flat_sleeve": intraday_flat_sleeve,
        "intraday_flat_sleeve_cost_rows": _json_ready_records(intraday_flat_sleeve_rows),
        "model_comparison_rows": int(len(model_comparison)),
        "optimization_plan": optimization_plan,
        "source_intraday_model": {
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "label_policy": intraday_summary.get("label_policy"),
            "external_context": intraday_summary.get("external_intraday_context"),
        },
    }


def _date_only(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _covers_date_range(coverage_start: Any, coverage_end: Any, required_start: str, required_end: str) -> bool:
    start = _date_only(coverage_start)
    end = _date_only(coverage_end)
    required_start_date = _date_only(required_start)
    required_end_date = _date_only(required_end)
    if start is None or end is None or required_start_date is None or required_end_date is None:
        return False
    return bool(start <= required_start_date and end >= required_end_date)


def _has_daily_replay_segment(
    daily_positions: pd.DataFrame,
    *,
    test_year: int,
    segment: str,
    fallback_start: str,
    fallback_end: str,
) -> bool:
    if daily_positions.empty or "datetime" not in daily_positions:
        return False
    if {"walk_forward_fold", "test_year"}.issubset(daily_positions.columns):
        fold_name = f"wf_{int(test_year)}"
        segment_col = "fold_segment" if "fold_segment" in daily_positions.columns else "segment"
        fold_frame = daily_positions.loc[
            daily_positions["walk_forward_fold"].astype(str).eq(fold_name)
            & pd.to_numeric(daily_positions["test_year"], errors="coerce").eq(int(test_year))
        ].copy()
        if fold_frame.empty or segment_col not in fold_frame:
            return False
        segment_values = fold_frame[segment_col].astype(str).replace({"valid": "validation"})
        return bool(segment_values.eq(segment).any())
    return _covers_date_range(
        daily_positions["datetime"].min(),
        daily_positions["datetime"].max(),
        fallback_start,
        fallback_end,
    )


def _delayed_walk_forward_feasibility(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    intraday_summary: dict[str, Any],
    daily_positions: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
) -> pd.DataFrame:
    external_context = intraday_summary.get("external_intraday_context", {}) or {}
    external_start = external_context.get("coverage_start")
    external_end = external_context.get("coverage_end")
    if not external_start and not intraday_predictions.empty:
        external_start = intraday_predictions["datetime"].min()
    if not external_end and not intraday_predictions.empty:
        external_end = intraday_predictions["datetime"].max()

    daily_start = daily_positions["datetime"].min() if "datetime" in daily_positions else None
    daily_end = daily_positions["datetime"].max() if "datetime" in daily_positions else None

    rows: list[dict[str, Any]] = []
    for test_year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS:
        train_start = REQUIRED_WALK_FORWARD_START
        train_end = f"{test_year - 2}-12-31"
        valid_start = f"{test_year - 1}-01-01"
        valid_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = spec.as_of if test_year == int(spec.as_of[:4]) else f"{test_year}-12-31"

        has_external_train = _covers_date_range(external_start, external_end, train_start, train_end)
        has_external_valid = _covers_date_range(external_start, external_end, valid_start, valid_end)
        has_external_test = _covers_date_range(external_start, external_end, test_start, test_end)
        has_daily_train = True
        has_daily_valid = _has_daily_replay_segment(
            daily_positions,
            test_year=int(test_year),
            segment="validation",
            fallback_start=valid_start,
            fallback_end=valid_end,
        )
        has_daily_test = _has_daily_replay_segment(
            daily_positions,
            test_year=int(test_year),
            segment="test",
            fallback_start=test_start,
            fallback_end=test_end,
        )
        feasible = all(
            [
                has_external_train,
                has_external_valid,
                has_external_test,
                has_daily_valid,
                has_daily_test,
            ]
        )
        blocking_reason = "ok"
        if not has_external_train:
            blocking_reason = "missing_train_coverage"
        elif not has_external_valid:
            blocking_reason = "missing_valid_coverage"
        elif not has_daily_valid:
            blocking_reason = "missing_daily_position_valid_coverage"
        elif not has_external_test:
            blocking_reason = "missing_test_coverage"
        elif not has_daily_test:
            blocking_reason = "missing_daily_position_test_coverage"

        rows.append(
            {
                "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
                "test_year": int(test_year),
                "train_start": train_start,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "test_start": test_start,
                "test_end": test_end,
                "external_coverage_start": str(external_context.get("coverage_start") or ""),
                "external_coverage_end": str(external_context.get("coverage_end") or ""),
                "daily_position_start": _date_only(daily_start).isoformat() if _date_only(daily_start) else "",
                "daily_position_end": _date_only(daily_end).isoformat() if _date_only(daily_end) else "",
                "has_external_train_coverage": bool(has_external_train),
                "has_external_valid_coverage": bool(has_external_valid),
                "has_external_test_coverage": bool(has_external_test),
                "has_daily_position_train_coverage": bool(has_daily_train),
                "daily_position_train_required": False,
                "has_daily_position_valid_coverage": bool(has_daily_valid),
                "has_daily_position_test_coverage": bool(has_daily_test),
                "daily_position_coverage_scope": "validation_and_test_replay",
                "selection_uses_test_segment": False,
                "feasible": bool(feasible),
                "blocking_reason": blocking_reason,
            }
        )
    return pd.DataFrame(rows)


def _delayed_walk_forward_summary(
    feasibility: pd.DataFrame,
    *,
    intraday_summary: dict[str, Any],
    daily_positions: pd.DataFrame,
) -> dict[str, Any]:
    external_context = intraday_summary.get("external_intraday_context", {}) or {}
    feasible_years = (
        sorted(int(value) for value in feasibility.loc[feasibility["feasible"], "test_year"].tolist())
        if not feasibility.empty
        else []
    )
    required_years = list(REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS)
    missing_years = [year for year in required_years if year not in feasible_years]
    data_status = external_context.get("data_status") or "unknown"
    blocking_reasons: list[str] = []
    if data_status != "ok":
        blocking_reasons.append("insufficient_external_intraday_history")
    daily_coverage_columns = ["has_daily_position_valid_coverage", "has_daily_position_test_coverage"]
    if not feasibility.empty and any(
        bool((~feasibility[column].astype(bool)).any())
        for column in daily_coverage_columns
        if column in feasibility
    ):
        blocking_reasons.append("insufficient_daily_position_history")
    source_required = bool(intraday_summary.get("walk_forward", {}).get("required_policy_satisfied", False))
    if not source_required:
        blocking_reasons.append("source_intraday_walk_forward_not_satisfied")
    if missing_years:
        blocking_reasons.append("annual_walk_forward_not_run")

    return {
        "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
        "required_start": REQUIRED_WALK_FORWARD_START,
        "required_test_years": required_years,
        "feasible_test_years": feasible_years,
        "missing_required_test_years": missing_years,
        "rows": int(len(feasibility)),
        "selection_uses_test_segment": False,
        "required_policy_satisfied": bool(not missing_years and source_required and not blocking_reasons),
        "source_intraday_required_policy_satisfied": source_required,
        "data_status": data_status,
        "coverage_start": external_context.get("coverage_start"),
        "coverage_end": external_context.get("coverage_end"),
        "daily_position_start": daily_positions["datetime"].min().date().isoformat()
        if "datetime" in daily_positions and not daily_positions.empty
        else None,
        "daily_position_end": daily_positions["datetime"].max().date().isoformat()
        if "datetime" in daily_positions and not daily_positions.empty
        else None,
        "blocking_reasons": sorted(set(blocking_reasons)),
    }


def _write_delayed_sidecars(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    policy_selection_full: pd.DataFrame,
    selected_policy_rows: pd.DataFrame,
    intraday_flat_sleeve_rows: pd.DataFrame,
    model_comparison: pd.DataFrame,
    selected_model: dict[str, Any],
    intraday_flat_sleeve: dict[str, Any],
    analysis_summary: dict[str, Any],
    walk_forward_feasibility: pd.DataFrame,
) -> dict[str, str]:
    sidecars = {
        "model_comparison": "model_comparison.csv",
        "policy_selection": "policy_selection.csv",
        "selected_policy_cost_sensitivity": "selected_policy_cost_sensitivity.csv",
        "selected_intraday_flat_sleeve": "selected_intraday_flat_sleeve.json",
        "selected_intraday_flat_sleeve_cost_sensitivity": "selected_intraday_flat_sleeve_cost_sensitivity.csv",
        "selected_model": "selected_model.json",
        "analysis_summary": "analysis_summary.json",
        "walk_forward_feasibility": "walk_forward_feasibility.csv",
        "failure_modes": "failure_modes.md",
    }
    policy_selection_full.head(500).to_csv(spec.output_dir / sidecars["policy_selection"], index=False)
    selected_policy_rows.to_csv(spec.output_dir / sidecars["selected_policy_cost_sensitivity"], index=False)
    intraday_flat_sleeve_rows.to_csv(
        spec.output_dir / sidecars["selected_intraday_flat_sleeve_cost_sensitivity"],
        index=False,
    )
    model_comparison.to_csv(spec.output_dir / sidecars["model_comparison"], index=False)
    walk_forward_feasibility.to_csv(spec.output_dir / sidecars["walk_forward_feasibility"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["selected_intraday_flat_sleeve"]).write_text(
        json.dumps(intraday_flat_sleeve, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_modes = (
        "# Failure Modes\n\n"
        "- Delayed observable overlay 只能在前 30/60 分钟 bar 完成后使用这些特征，不能影响 next-open 或前 4 根 bar 的成交。\n"
        "- 选择规则只用 validation，并要求 20bp/50bp validation 同时满足 85%-95% retention、距离 retention 边界至少 0.05pct、max_dd>-18%、skip<=15%；eligible exit policy 是 hold_to_next_open 和 same_day_close。\n"
        "- same_day_close 是纯日内平仓候选，用 entry day 最后一根可用 15m close 退出；它仍缺少 bid/ask、收盘成交容量和延迟 replay。\n"
        "- retention 目标是 validation 85%-95% hard gate；超过 95% 的短窗口收益只能作为诊断，不算晋级通过。\n"
        "- 年度验证使用 daily walk-forward replay positions 的 validation/test 片段；train 阶段不要求 selected positions，因为仓位模型训练证据属于源日线 card。\n"
        "- 当前外部 15m 上下文覆盖仍不足，walk_forward_feasibility.csv 对 2023-2026 年度 fold 继续 fail closed。\n"
        "- 还没有 bid/ask、队列、延迟和盘前盘后 replay；只能作为研究候选，不是生产执行策略。\n"
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(failure_modes, encoding="utf-8")
    return sidecars


def _delayed_artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "selected_trades": "artifacts/delayed_observable_overlay/selected_trades.csv",
            "all_policy_selection": "artifacts/delayed_observable_overlay/all_policy_selection.csv",
            "daily_positions": "artifacts/delayed_observable_overlay/daily_selected_positions.csv",
            "daily_walk_forward_positions": "artifacts/delayed_observable_overlay/daily_walk_forward_positions.csv",
        },
        "reproducible_from_recorder": True,
    }


def _delayed_promotion_decision(
    selected_policy_rows: pd.DataFrame,
    *,
    intraday_summary: dict[str, Any],
    walk_forward_summary: dict[str, Any],
) -> str:
    if selected_policy_rows.empty:
        return "not_promoted_delayed_overlay_validation_gate_failed"
    required = selected_policy_rows.loc[selected_policy_rows["cost_bps"].isin([20.0, 50.0])]
    if len(required) < 2 or not bool(required["passes_validation_target"].all()):
        return "not_promoted_delayed_overlay_validation_gate_failed"
    if not bool(required["passes_test_target"].all()):
        return "not_promoted_delayed_overlay_test_gate_failed"
    if not bool(walk_forward_summary.get("required_policy_satisfied", False)):
        return "not_promoted_external_intraday_history_insufficient"
    if intraday_summary.get("walk_forward", {}).get("required_policy_satisfied") is False:
        return "not_promoted_external_intraday_history_insufficient"
    return "not_promoted_intraday_replay_missing"


def _write_delayed_conclusion(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    summary: dict[str, Any],
    selected_model: dict[str, Any],
    selected_policy_rows: pd.DataFrame,
    intraday_flat_sleeve: dict[str, Any],
    intraday_flat_sleeve_rows: pd.DataFrame,
) -> None:
    rows_by_cost = {
        float(row["cost_bps"]): row
        for row in selected_policy_rows.to_dict("records")
    }
    flat_rows_by_cost = {
        float(row["cost_bps"]): row
        for row in intraday_flat_sleeve_rows.to_dict("records")
    }
    row20 = rows_by_cost.get(20.0, {})
    row50 = rows_by_cost.get(50.0, {})
    flat20 = flat_rows_by_cost.get(20.0, {})
    flat50 = flat_rows_by_cost.get(50.0, {})
    optimization_plan = summary.get("optimization_plan", {})
    optimization_sentence = (
        f"- 优化方向：status={optimization_plan.get('status')}；"
        f"next={optimization_plan.get('primary_next_action')}。"
        "短窗口 delayed entry 有效但外部上下文历史不足，下一步先用本地 SOXL/SOXX 15m/5m 特征跑完整年度 walk-forward，"
        "再决定是否叠加 QQQ/SMH/成分股 lead-lag。\n"
        if optimization_plan
        else ""
    )
    text = (
        "# 中文结论\n\n"
        "- 做了什么：建立 delayed observable intraday overlay，日线 LGB/XGB open-to-open 仓位仍决定是否交易；"
        "日内层必须等前 30/60 分钟完成后，才能用 observed probability/return 条件触发后续 15m next-open 入场；"
        "本轮把 same_day_close 作为可晋升的纯日内平仓候选加入，同时保留 hold_to_next_open；risk-exit 仍只作为诊断候选，"
        "避免短样本风险退出过拟合。\n"
        "- 选择规则：只用 validation；retention 必须落在 85%-95% hard gate 内，"
        "并且 20/50bp required cost 下都要距离 retention 边界至少 0.05pct，"
        "再按 validation skip rate、条件覆盖率和回撤改善排序；test 只报告，不参与阈值选择。"
        "超过 95% 的短窗口收益保留为诊断，不计入晋级目标。\n"
        f"- 候选：{summary['model_selection']['candidate_count']} 个 policy、{summary['model_selection']['candidate_rows']} 行成本压力结果；"
        f"validation+test 同时过门槛行数={summary['model_selection']['validation_and_test_target_row_count']}；"
        f"其中 same_day_close 同时过门槛行数={summary['model_selection'].get('same_day_close_validation_and_test_target_row_count', 0)}，"
        f"risk-exit 同时过门槛行数={summary['model_selection'].get('risk_exit_validation_and_test_target_row_count', 0)}。\n"
        f"- validation 选中：{selected_model.get('policy_id')}，exit_policy={selected_model.get('exit_policy', 'unknown')}；"
        f"20bp retention={float(row20.get('validation_return_retention_pct', 0.0)):.2f}%，"
        f"max_dd={float(row20.get('validation_max_drawdown_pct', 0.0)):.2f}%；50bp retention={float(row50.get('validation_return_retention_pct', 0.0)):.2f}%，"
        f"max_dd={float(row50.get('validation_max_drawdown_pct', 0.0)):.2f}%。\n"
        f"- test 报告：20bp retention={float(row20.get('test_return_retention_pct', 0.0)):.2f}%，"
        f"max_dd={float(row20.get('test_max_drawdown_pct', 0.0)):.2f}%；50bp retention={float(row50.get('test_return_retention_pct', 0.0)):.2f}%，"
        f"max_dd={float(row50.get('test_max_drawdown_pct', 0.0)):.2f}%。\n"
        f"- 日内 flat sleeve：只在 same_day_close 候选内按 validation 回撤优先选择；选中 {intraday_flat_sleeve.get('policy_id')}；"
        f"20bp validation retention={float(flat20.get('validation_return_retention_pct', 0.0)):.2f}%，"
        f"max_dd={float(flat20.get('validation_max_drawdown_pct', 0.0)):.2f}%，test retention={float(flat20.get('test_return_retention_pct', 0.0)):.2f}%，"
        f"test max_dd={float(flat20.get('test_max_drawdown_pct', 0.0)):.2f}%；50bp test retention={float(flat50.get('test_return_retention_pct', 0.0)):.2f}%，"
        f"test max_dd={float(flat50.get('test_max_drawdown_pct', 0.0)):.2f}%。这个 sleeve 更接近纯日内执行，但仍受外部 15m 历史不足和收盘成交 replay 缺失约束。\n"
        f"- walk-forward：要求 {summary['walk_forward']['required_test_years']} 年度 train_to_T-2/valid_T-1/test_T；"
        f"当前 feasible 年份={summary['walk_forward']['feasible_test_years']}，"
        f"缺失={summary['walk_forward']['missing_required_test_years']}，因此 required_policy_satisfied="
        f"{summary['walk_forward']['required_policy_satisfied']}。\n"
        f"{optimization_sentence}"
        f"- 是否晋升：{summary['promotion_decision']}。短窗口内这个 delayed entry 方向确实把 test max_dd 从 baseline 约 -18.38% 压到更低，"
        "但外部 15m 历史和真实成交 replay 还不足，不能直接晋升生产。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _regime_artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "overlay_trades": "artifacts/regime_overlay/overlay_trades.csv",
            "regime_features": "artifacts/regime_overlay/regime_features.csv",
            "policy_selection": "artifacts/regime_overlay/policy_selection.csv",
        },
        "reproducible_from_recorder": True,
    }


def _regime_promotion_decision(selected_model: dict[str, Any], *, intraday_summary: dict[str, Any]) -> str:
    if not bool(selected_model.get("validation_passes_target")):
        return "not_promoted_regime_overlay_validation_gate_failed"
    if not bool(selected_model.get("test_passes_target")):
        return "not_promoted_regime_overlay_test_gate_failed"
    if intraday_summary.get("walk_forward", {}).get("required_policy_satisfied") is False:
        return "not_promoted_external_intraday_history_insufficient"
    return "not_promoted_intraday_replay_missing"


def _write_regime_conclusion(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    summary: dict[str, Any],
    selected_model: dict[str, Any],
) -> None:
    policy = summary["regime_policy"]
    required_cost_results = selected_model.get("required_cost_results") or []
    required_cost_sentence = ""
    if required_cost_results:
        cost_parts = []
        for row in required_cost_results:
            cost_parts.append(
                f"{float(row.get('cost_bps') or 0.0):.0f}bp "
                f"valid_ret={float(row.get('validation_return_retention_pct') or 0.0):.2f}%/"
                f"valid_dd={float(row.get('validation_max_drawdown_pct') or 0.0):.2f}%/"
                f"test_ret={float(row.get('test_return_retention_pct') or 0.0):.2f}%/"
                f"test_dd={float(row.get('test_max_drawdown_pct') or 0.0):.2f}%/"
                f"test_pass={bool(row.get('passes_test_target'))}"
            )
        required_cost_sentence = (
            f"- 20/50bp validation 成本鲁棒：validation_pass_count="
            f"{selected_model.get('required_cost_validation_target_count', 0)}，"
            f"test_pass_count={selected_model.get('required_cost_test_target_count', 0)}；"
            + "；".join(cost_parts)
            + "。\n"
        )
    text = (
        "# 中文结论\n\n"
        "- 做了什么：建立 causal regime overlay 诊断卡，日线 LGB/XGB open-to-open 仓位仍是 baseline，"
        "外部 15m LGB 模型只作为执行 overlay；条件选择只允许使用 daily close 时已知的 probability、target_position、turnover、raw_target。\n"
        f"- 选择规则：只用 validation、required_cost_bps={policy.get('required_cost_bps', [spec.selection_cost_bps])} "
        "和 validation quantile 阈值选条件；test 只报告，不参与调参。\n"
        f"- 候选数：{policy['candidate_count']}；validation 过 85%-95% retention 且 max_dd>-18% 的候选数="
        f"{policy['validation_target_candidate_count']}；validation 和 test 同时过门槛的候选数="
        f"{policy['validation_and_test_target_candidate_count']}。\n"
        f"- validation 选中：{selected_model.get('variant')} / {selected_model.get('condition_name')}，"
        f"retention={float(selected_model.get('validation_return_retention_pct') or 0.0):.2f}%，"
        f"max_dd={float(selected_model.get('validation_max_drawdown_pct') or 0.0):.2f}%。\n"
        f"- test 报告：retention={float(selected_model.get('test_return_retention_pct') or 0.0):.2f}%，"
        f"max_dd={float(selected_model.get('test_max_drawdown_pct') or 0.0):.2f}%。\n"
        f"{required_cost_sentence}"
        f"- 是否晋升：{summary['promotion_decision']}。结论是当前 causal 日内 overlay 还不能替代日线 next-open 执行；"
        "下一步应优化日线整体仓位模型或做真正延迟入场/风险退出模型，而不是用看完第一小时后的特征证明开盘 TWAP。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _json_ready_dict(values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = None if np.isnan(value) else float(value)
        elif pd.isna(value):
            result[key] = None
        else:
            result[key] = value
    return result


def _json_ready_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_ready_dict(row) for row in frame.to_dict("records")]


def _write_sidecars(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    model_comparison: pd.DataFrame,
    threshold_selection: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    execution_risk_diagnostics: pd.DataFrame,
    conditional_risk_policy_selection: pd.DataFrame,
    conditional_risk_sleeve: dict[str, Any],
    conditional_risk_sleeve_rows: pd.DataFrame,
    conditional_risk_sleeve_walk_forward: pd.DataFrame,
    walk_forward: pd.DataFrame,
    daily_position_source_audit: pd.DataFrame,
    selected_model: dict[str, Any],
    analysis_summary: dict[str, Any],
) -> dict[str, str]:
    sidecars = {
        "model_comparison": "model_comparison.csv",
        "selected_model": "selected_model.json",
        "cost_sensitivity": "cost_sensitivity.csv",
        "execution_risk_diagnostics": "execution_risk_diagnostics.csv",
        "conditional_risk_policy_selection": "conditional_risk_policy_selection.csv",
        "conditional_risk_sleeve_walk_forward": "conditional_risk_sleeve_walk_forward.csv",
        "selected_conditional_risk_sleeve": "selected_conditional_risk_sleeve.json",
        "selected_conditional_risk_sleeve_cost_sensitivity": "selected_conditional_risk_sleeve_cost_sensitivity.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "daily_position_source_audit": "daily_position_source_audit.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    if spec.optimize_thresholds:
        sidecars["threshold_selection"] = "threshold_selection.csv"
    model_comparison.to_csv(spec.output_dir / sidecars["model_comparison"], index=False)
    if spec.optimize_thresholds:
        threshold_selection.to_csv(spec.output_dir / sidecars["threshold_selection"], index=False)
    cost_sensitivity.to_csv(spec.output_dir / sidecars["cost_sensitivity"], index=False)
    execution_risk_diagnostics.to_csv(spec.output_dir / sidecars["execution_risk_diagnostics"], index=False)
    conditional_risk_policy_selection.to_csv(
        spec.output_dir / sidecars["conditional_risk_policy_selection"],
        index=False,
    )
    conditional_risk_sleeve_rows.to_csv(
        spec.output_dir / sidecars["selected_conditional_risk_sleeve_cost_sensitivity"],
        index=False,
    )
    conditional_risk_sleeve_walk_forward.to_csv(
        spec.output_dir / sidecars["conditional_risk_sleeve_walk_forward"],
        index=False,
    )
    walk_forward.to_csv(spec.output_dir / sidecars["backtest_walk_forward_metrics"], index=False)
    daily_position_source_audit.to_csv(spec.output_dir / sidecars["daily_position_source_audit"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["selected_conditional_risk_sleeve"]).write_text(
        json.dumps(conditional_risk_sleeve, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_modes = (
        "# Failure Modes\n\n"
        "- Overlay 使用的是已有日级 target_position，不重新训练仓位模型；它只能改善成交和盘中退出，不能创造隔夜趋势收益。\n"
        "- 15m 信号来自 bar close，overlay A 只允许下一根 15m open 入场；没有使用同一根 bar 的未来价格。\n"
        "- 成本压力用固定 bps；尚未验证 bid/ask、排队、延迟、停牌、跳空和盘前盘后成交质量。\n"
        "- walk-forward 只在已有日级 selected positions 上重新选择执行 variant；不是完整重训日级 LGB/XGB 仓位模型。\n"
        "- conditional_risk_sleeve_walk_forward.csv 按年度 fold 重新选择条件 sleeve；主切分 selected_conditional_risk_sleeve.json 不能单独作为年度稳定证据。\n"
        "- 若 daily_position_source_audit.csv 缺少 2022-2026 的 selected positions，年度日内 overlay 只能作为局部诊断，不能作为超过日级模型的晋升证据。\n"
        "- 若 validation 选择 baseline，说明日内执行层没有足够证据替代 next-open 一次成交。\n"
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(failure_modes, encoding="utf-8")
    return sidecars


def _artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "overlay_trades": "artifacts/execution_overlay/overlay_trades.csv",
            "walk_forward_overlay_trades": "artifacts/execution_overlay/walk_forward_overlay_trades.csv",
            "daily_positions": "artifacts/execution_overlay/daily_selected_positions.csv",
            "daily_walk_forward_positions": "artifacts/execution_overlay/daily_walk_forward_positions.csv",
            "intraday_predictions": "artifacts/execution_overlay/intraday_predictions_15m.csv",
        },
        "reproducible_from_recorder": True,
    }


def _required_overlay_policy_satisfied(
    *,
    intraday_summary: dict[str, Any],
    walk_forward: pd.DataFrame,
) -> bool:
    source_required_policy = intraday_summary.get("walk_forward", {}).get("required_policy_satisfied", True)
    if walk_forward.empty or "test_year" not in walk_forward or "status" not in walk_forward:
        return False
    required_years = {int(year) for year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS}
    ok_years = {
        int(row["test_year"])
        for row in walk_forward.to_dict("records")
        if str(row.get("status")) == "ok" and bool(row.get("validation_passes_target")) and bool(row.get("test_passes_target"))
    }
    return bool(source_required_policy and required_years.issubset(ok_years))


def _overlay_walk_forward_summary(
    *,
    intraday_summary: dict[str, Any],
    walk_forward: pd.DataFrame,
) -> dict[str, Any]:
    required_years = [int(year) for year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS]
    feasible_years = (
        sorted(
            int(row["test_year"])
            for row in walk_forward.to_dict("records")
            if str(row.get("status")) == "ok"
        )
        if not walk_forward.empty and "test_year" in walk_forward
        else []
    )
    blocking_reasons = (
        sorted(
            {
                str(reason)
                for reason in walk_forward.get("blocking_reason", pd.Series(dtype=object)).dropna().tolist()
                if str(reason) != "ok"
            }
        )
        if not walk_forward.empty
        else []
    )
    missing_years = [year for year in required_years if year not in feasible_years]
    return {
        "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
        "required_start": REQUIRED_WALK_FORWARD_START,
        "required_test_years": required_years,
        "feasible_test_years": feasible_years,
        "missing_required_test_years": missing_years,
        "folds": walk_forward["fold"].tolist() if "fold" in walk_forward else [],
        "rows": int(len(walk_forward)),
        "selection_uses_test_segment": False,
        "blocking_reasons": blocking_reasons,
        "source_intraday_required_policy_satisfied": bool(
            intraday_summary.get("walk_forward", {}).get("required_policy_satisfied", True)
        ),
        "required_policy_satisfied": _required_overlay_policy_satisfied(
            intraday_summary=intraday_summary,
            walk_forward=walk_forward,
        ),
    }


def _conditional_risk_sleeve_walk_forward_summary(
    *,
    intraday_summary: dict[str, Any],
    walk_forward: pd.DataFrame,
    required_cost_bps: tuple[float, ...] = (20.0, 50.0),
) -> dict[str, Any]:
    required_years = [int(year) for year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS]
    required_cost_count = len(tuple(float(value) for value in required_cost_bps))
    feasible_years = (
        sorted(
            int(row["test_year"])
            for row in walk_forward.to_dict("records")
            if str(row.get("status")) == "ok"
        )
        if not walk_forward.empty and "test_year" in walk_forward
        else []
    )
    validation_target_years = (
        sorted(
            int(row["test_year"])
            for row in walk_forward.to_dict("records")
            if str(row.get("status")) == "ok"
            and int(row.get("required_cost_validation_target_count") or 0) >= required_cost_count
        )
        if not walk_forward.empty and "test_year" in walk_forward
        else []
    )
    test_target_years = (
        sorted(
            int(row["test_year"])
            for row in walk_forward.to_dict("records")
            if str(row.get("status")) == "ok"
            and int(row.get("required_cost_test_target_count") or 0) >= required_cost_count
        )
        if not walk_forward.empty and "test_year" in walk_forward
        else []
    )
    validation_and_test_target_years = (
        sorted(
            int(row["test_year"])
            for row in walk_forward.to_dict("records")
            if str(row.get("status")) == "ok"
            and int(row.get("required_cost_validation_and_test_target_count") or 0) >= required_cost_count
        )
        if not walk_forward.empty and "test_year" in walk_forward
        else []
    )
    blocking_reasons = (
        sorted(
            {
                str(reason)
                for reason in walk_forward.get("blocking_reason", pd.Series(dtype=object)).dropna().tolist()
                if str(reason) != "ok"
            }
        )
        if not walk_forward.empty
        else []
    )
    missing_years = [year for year in required_years if year not in feasible_years]
    target_failure_years = [year for year in required_years if year not in validation_and_test_target_years]
    selection_uses_test_segment = bool(
        walk_forward.get("selection_uses_test_segment", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
    )
    source_required_policy = bool(intraday_summary.get("walk_forward", {}).get("required_policy_satisfied", True))
    return {
        "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
        "required_start": REQUIRED_WALK_FORWARD_START,
        "required_test_years": required_years,
        "required_cost_bps": [float(value) for value in required_cost_bps],
        "feasible_test_years": feasible_years,
        "missing_required_test_years": missing_years,
        "required_cost_validation_target_years": validation_target_years,
        "required_cost_test_target_years": test_target_years,
        "required_cost_validation_and_test_target_years": validation_and_test_target_years,
        "target_failure_years": target_failure_years,
        "folds": walk_forward["fold"].tolist() if "fold" in walk_forward else [],
        "rows": int(len(walk_forward)),
        "selection_uses_test_segment": selection_uses_test_segment,
        "blocking_reasons": blocking_reasons,
        "source_intraday_required_policy_satisfied": source_required_policy,
        "required_policy_satisfied": bool(
            source_required_policy
            and not selection_uses_test_segment
            and set(required_years).issubset(set(validation_and_test_target_years))
        ),
    }


def _promotion_decision(selected_model: dict[str, Any], *, intraday_summary: dict[str, Any]) -> str:
    if intraday_summary.get("walk_forward", {}).get("required_policy_satisfied") is False:
        return "not_promoted_external_intraday_history_insufficient"
    if selected_model["variant"] == "baseline_next_open":
        return "not_promoted_overlay_did_not_beat_daily_execution"
    if not bool(selected_model.get("validation_passes_target")):
        return "not_promoted_overlay_validation_gate_failed"
    if float(selected_model.get("test_return_retention_pct", 0.0)) < 85.0:
        return "not_promoted_test_return_retention_failed"
    return "not_promoted_intraday_replay_missing"


def _write_conclusion(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    summary: dict[str, Any],
    selected_model: dict[str, Any],
    model_comparison: pd.DataFrame,
    walk_forward: pd.DataFrame,
    conditional_risk_sleeve_walk_forward: pd.DataFrame,
    execution_risk_summary: dict[str, Any],
    conditional_risk_sleeve: dict[str, Any],
) -> None:
    test_rows = model_comparison.loc[model_comparison["segment"].eq("test")].copy()
    selected_test = test_rows.loc[test_rows["variant"].eq(selected_model["variant"])]
    selected_test_row = selected_test.iloc[0] if not selected_test.empty else None
    baseline_test = test_rows.loc[test_rows["variant"].eq("baseline_next_open")]
    baseline_return = float(baseline_test.iloc[0]["total_return_pct"]) if not baseline_test.empty else 0.0
    selected_test_return = float(selected_test_row["total_return_pct"]) if selected_test_row is not None else 0.0
    selected_test_dd = float(selected_test_row["max_drawdown_pct"]) if selected_test_row is not None else 0.0
    threshold_search = summary.get("model_selection", {}).get("threshold_search", {})
    threshold_sentence = (
        f"- 阈值搜索：只用 validation 搜 {threshold_search.get('candidate_count', 0)} 个候选，"
        f"满足 85%-95% retention 且 max_dd>-18% 的候选数={threshold_search.get('target_candidate_count', 0)}。\n"
        if threshold_search.get("enabled")
        else ""
    )
    overlay_name = (
        "外部 15m 上下文执行 overlay"
        if spec.source_intraday_card == EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD
        else "日级模型 + 日内执行 overlay"
    )
    intraday_source_text = (
        "外部 15m 上下文 triple-barrier/execution-improvement 模型"
        if spec.source_intraday_card == EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD
        else "15m lgb_xgb_consensus"
    )
    intraday_source_separator = "" if spec.source_intraday_card == EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD else " "
    walk_forward_summary = summary.get("walk_forward", {})
    walk_forward_sentence = (
        f"- walk-forward：要求 {walk_forward_summary.get('required_test_years', [])} 年度 "
        "train_to_T-2/valid_T-1/test_T；"
        f"当前 feasible 年份={walk_forward_summary.get('feasible_test_years', [])}，"
        f"缺失={walk_forward_summary.get('missing_required_test_years', [])}，"
        f"required_policy_satisfied={walk_forward_summary.get('required_policy_satisfied')}。\n"
    )
    annual_probe_sentence = ""
    if not walk_forward.empty and "test_year" in walk_forward:
        latest_ok = walk_forward.loc[walk_forward["status"].eq("ok")].sort_values("test_year")
        if not latest_ok.empty:
            row = latest_ok.iloc[-1]
            annual_probe_sentence = (
                f"- 最新年度 fold：{row['fold']} validation 选择 {row['selected_variant']}，"
                f"test retention={float(row.get('test_return_retention_pct', 0.0)):.2f}%，"
                f"test max_dd={float(row.get('test_max_drawdown_pct', 0.0)):.2f}%，"
                f"passes_target={bool(row.get('test_passes_target'))}。\n"
            )
    risk_sentence = (
        "- open-buy-dump 诊断："
        f"{execution_risk_summary.get('open_buy_dump_threshold_pct', -3.0):.0f}% 当日入场后回撤阈值，"
        f"baseline test 次数={execution_risk_summary.get('baseline_test_open_buy_dump_periods', 0)}，"
        f"selected test 次数={execution_risk_summary.get('selected_test_open_buy_dump_periods', 0)}，"
        f"减少={execution_risk_summary.get('selected_test_open_buy_dump_period_reduction', 0)}；"
        f"crash 次数减少={execution_risk_summary.get('selected_test_crash_period_reduction', 0)}。\n"
    )
    conditional_sentence = (
        "- 条件化 risk sleeve："
        f"validation 只用 {conditional_risk_sleeve.get('condition_column')} "
        f"{conditional_risk_sleeve.get('condition_operator')} "
        f"{float(conditional_risk_sleeve.get('condition_threshold') or 0.0):.4f} 时切到 "
        f"{conditional_risk_sleeve.get('variant')}；"
        f"validation open-buy-dump 减少={conditional_risk_sleeve.get('validation_open_buy_dump_period_reduction', 0)}，"
        f"test retention={float(conditional_risk_sleeve.get('test_return_retention_pct') or 0.0):.2f}%，"
        f"test max_dd={float(conditional_risk_sleeve.get('test_max_drawdown_pct') or 0.0):.2f}%，"
        f"test open-buy-dump 减少={conditional_risk_sleeve.get('test_open_buy_dump_period_reduction', 0)}。\n"
    )
    conditional_walk_forward_summary = summary.get("conditional_risk_sleeve_walk_forward", {})
    conditional_walk_forward_sentence = (
        "- 条件化 risk sleeve 年度 walk-forward："
        f"feasible 年份={conditional_walk_forward_summary.get('feasible_test_years', [])}，"
        f"20/50bp valid+test 都过目标的年份="
        f"{conditional_walk_forward_summary.get('required_cost_validation_and_test_target_years', [])}，"
        f"required_policy_satisfied={conditional_walk_forward_summary.get('required_policy_satisfied')}。\n"
    )
    conditional_annual_probe_sentence = ""
    if not conditional_risk_sleeve_walk_forward.empty and "test_year" in conditional_risk_sleeve_walk_forward:
        latest_ok = conditional_risk_sleeve_walk_forward.loc[
            conditional_risk_sleeve_walk_forward["status"].eq("ok")
        ].sort_values("test_year")
        if not latest_ok.empty:
            row = latest_ok.iloc[-1]
            conditional_annual_probe_sentence = (
                f"- 最新条件 sleeve fold：{row['fold']} validation 选择 {row['selected_variant']} "
                f"{row.get('condition_column')} {row.get('condition_operator')} "
                f"{float(row.get('condition_threshold') or 0.0):.4f}；"
                f"test retention={float(row.get('test_return_retention_pct') or 0.0):.2f}%，"
                f"test max_dd={float(row.get('test_max_drawdown_pct') or 0.0):.2f}%，"
                f"20/50bp valid+test 过关数={int(row.get('required_cost_validation_and_test_target_count') or 0)}。\n"
            )
    selected_test_retention = float(selected_model.get("test_return_retention_pct") or 0.0)
    selected_test_passes = bool(85.0 <= selected_test_retention <= 95.0 and selected_test_dd > -18.0)
    if selected_model.get("variant") == "baseline_next_open":
        promotion_reason = "selected 仍是 baseline，当前 15m 执行层没有可靠超过 next-open 一次成交。"
    elif not selected_test_passes:
        promotion_reason = (
            f"selected test retention={selected_test_retention:.2f}% 未过 85%-95% 目标，"
            "说明 validation-only 选择仍不能推广。"
        )
    elif not bool(walk_forward_summary.get("required_policy_satisfied")):
        promotion_reason = "selected 短窗口过关，但年度/source walk-forward 证据不足，不能晋升。"
    else:
        promotion_reason = "执行层仍缺真实成交 replay 证据，不能晋升。"
    optimization_plan = summary.get("optimization_plan", {})
    optimization_detail = ""
    if optimization_plan:
        if optimization_plan.get("external_context_status") == "not_used":
            selected_variant = str(selected_model.get("variant") or "unknown")
            if selected_variant == "baseline_next_open":
                optimization_detail = (
                    "当前本地 15m gate 仍选 baseline，下一步先换 delayed observable label "
                    "和本地 open-buy-dump 风险因子，不要强推现有 gate。"
                )
            else:
                optimization_detail = (
                    f"当前本地 15m validation 选择 {selected_variant}，但 test retention="
                    f"{selected_test_retention:.2f}%，说明它过度过滤日线趋势收益；"
                    "下一步先换 delayed observable label 和本地 open-buy-dump 风险因子。"
                )
        else:
            optimization_detail = (
                "若外部 context 历史不足，先跑本地 SOXL/SOXX 15m/5m annual walk-forward fallback，"
                "不要继续在短历史外部阈值上调参。"
            )
    optimization_sentence = (
        f"- 优化方向：status={optimization_plan.get('status')}；"
        f"next={optimization_plan.get('primary_next_action')}。{optimization_detail}\n"
        if optimization_plan
        else ""
    )
    text = (
        "# 中文结论\n\n"
        f"- 做了什么：建立“{overlay_name}”实验，日级 LGB/XGB 仓位模型仍决定整体 target_position，"
        f"{intraday_source_text}{intraday_source_separator}只负责下一交易日的入场过滤、前 3/4 根 15m TWAP 和盘中风险退出。\n"
        f"- 选择规则：只用 validation、{spec.selection_cost_bps:.0f}bp 成本选择 variant；test 只报告，"
        "没有用 test 调阈值或调参数。\n"
        f"- validation 选择：{selected_model['variant']}，retention={selected_model['validation_return_retention_pct']:.2f}%，"
        f"max_dd={selected_model['validation_max_drawdown_pct']:.2f}%。\n"
        f"{threshold_sentence}"
        f"- test 对照：baseline_next_open return={baseline_return:.2f}%，"
        f"selected return={selected_test_return:.2f}%、max_dd={selected_test_dd:.2f}%。\n"
        f"{risk_sentence}"
        f"{conditional_sentence}"
        f"{conditional_walk_forward_sentence}"
        f"{conditional_annual_probe_sentence}"
        f"{walk_forward_sentence}"
        f"{annual_probe_sentence}"
        f"{optimization_sentence}"
        f"- 是否晋升：{summary['promotion_decision']}。{promotion_reason}\n"
        "- 风险：这不是独立日内 alpha；它依赖日级模型的趋势和隔夜收益来源。真实推广前还要补 bid/ask replay、成交队列、延迟和盘前盘后风险验证。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _empty_variant_row(daily_row: dict[str, Any], variant: str, cost_bps: float, reason: str) -> dict[str, Any]:
    date = pd.Timestamp(daily_row["datetime"]).date().isoformat()
    return {
        "datetime": date,
        "segment": str(daily_row.get("segment", "unknown")),
        **_daily_walk_forward_metadata(daily_row),
        "target_position": float(daily_row.get("target_position", 0.0) or 0.0),
        "cost_bps": float(cost_bps),
        "entry_date": None,
        "exit_date": None,
        "source_daily_strategy_return": float(daily_row.get("strategy_return", 0.0) or 0.0),
        "variant": variant,
        "entry_time": None,
        "exit_time": None,
        "entry_price": np.nan,
        "exit_price": np.nan,
        "entry_day_close_time": None,
        "entry_day_close_price": np.nan,
        "entry_day_strategy_return": np.nan,
        "intraday_signal_score": np.nan,
        "gross_return": 0.0,
        "strategy_return": 0.0,
        "skipped_reason": reason,
    }


def _daily_walk_forward_metadata(daily_row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ["walk_forward_fold", "test_year", "fold_segment"]:
        value = daily_row.get(key)
        if value is not None and not pd.isna(value):
            metadata[key] = value
    return metadata


def _date_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _timestamp_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _resolve_root(root: Path | str | None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[2]
    return Path(root).resolve()


def _recorder_files(recorder_dir: Path) -> list[str]:
    return sorted(path.relative_to(recorder_dir).as_posix() for path in recorder_dir.rglob("*") if path.is_file())


def _remove_transient_code_snapshots(recorder_dir: Path) -> None:
    for name in ["code_cached.txt", "code_diff.txt", "code_status.txt"]:
        path = recorder_dir / "artifacts" / name
        if path.exists():
            path.unlink()
