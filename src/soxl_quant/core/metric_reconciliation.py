from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .constants import ACCOUNT


def _trading_dates(index: pd.Index | pd.MultiIndex | Sequence[object]) -> pd.DatetimeIndex:
    raw_index = pd.Index(index)
    if isinstance(raw_index, pd.MultiIndex):
        level_name = "datetime" if "datetime" in raw_index.names else raw_index.names[0]
        values = raw_index.get_level_values(level_name)
    else:
        values = raw_index
    dates = pd.DatetimeIndex(pd.to_datetime(values)).dropna().unique().sort_values()
    if dates.empty:
        raise ValueError("empty trading calendar")
    return dates


def _date_label(value: pd.Timestamp) -> str:
    return str(pd.Timestamp(value).date())


def build_warm_start_replay_segment(
    index: pd.Index | pd.MultiIndex | Sequence[object],
    *,
    eval_seg: tuple[str, str],
    lookback: int,
) -> tuple[str, str]:
    """Return an official replay segment with prior trading rows for warmup.

    The returned segment starts ``lookback`` trading rows before the requested
    evaluation start, clamped to available data, and ends at the last available
    trading row within ``eval_seg``. Metrics should still be sliced back to
    ``eval_seg`` after the qlib replay completes.
    """

    dates = _trading_dates(index)
    eval_start = pd.Timestamp(eval_seg[0])
    eval_end = pd.Timestamp(eval_seg[1])
    eval_dates = dates[(dates >= eval_start) & (dates <= eval_end)]
    if eval_dates.empty:
        raise ValueError(f"empty evaluation window for segment {eval_seg}")
    eval_start_pos = int(dates.get_loc(eval_dates[0]))
    replay_start_pos = max(0, eval_start_pos - max(int(lookback), 0))
    return (_date_label(dates[replay_start_pos]), _date_label(eval_dates[-1]))


def _series_with_datetime_index(series: pd.Series) -> pd.Series:
    out = series.copy()
    if isinstance(out.index, pd.MultiIndex):
        level_name = "datetime" if "datetime" in out.index.names else out.index.names[0]
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index.get_level_values(level_name)))
    else:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    return out[~out.index.duplicated(keep="last")].sort_index()


def _frame_with_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex):
        level_name = "datetime" if "datetime" in out.index.names else out.index.names[0]
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index.get_level_values(level_name)))
    else:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    return out[~out.index.duplicated(keep="last")].sort_index()


def slice_warm_start_daily_path(
    *,
    portfolio_df: pd.DataFrame,
    price_series: pd.Series,
    eval_seg: tuple[str, str],
    position_history: Sequence[float] | None = None,
    trade_signals: Sequence[tuple] | None = None,
    account_base: float = ACCOUNT,
) -> pd.DataFrame:
    """Slice a warm-start official replay to eval dates and rebase NAVs.

    qlib should be allowed to replay a warmup segment so the policy has enough
    history to issue orders. This helper makes the metric window explicit by
    cutting back to ``eval_seg`` and rebasing both strategy and buy-and-hold NAV
    from the first evaluation date.
    """

    portfolio = _frame_with_datetime_index(portfolio_df)
    if "account" not in portfolio.columns:
        raise ValueError("portfolio_df missing required 'account' column")
    account = portfolio["account"].astype(float)
    eval_account = account.loc[pd.Timestamp(eval_seg[0]) : pd.Timestamp(eval_seg[1])].dropna()
    if eval_account.empty:
        raise ValueError(f"empty evaluation account window for segment {eval_seg}")
    start_account = float(eval_account.iloc[0])
    if not math.isfinite(start_account) or start_account == 0.0:
        raise ValueError("invalid evaluation start account")

    prices = _series_with_datetime_index(price_series).astype(float)
    aligned_prices = prices.reindex(account.index).ffill().bfill()
    eval_prices = aligned_prices.reindex(eval_account.index).ffill().bfill()
    if eval_prices.isna().any():
        raise ValueError(f"price series does not cover evaluation segment {eval_seg}")
    start_price = float(eval_prices.iloc[0])
    if not math.isfinite(start_price) or start_price == 0.0:
        raise ValueError("invalid evaluation start price")

    strategy_nav = float(account_base) * (eval_account / start_account)
    buyhold_nav = float(account_base) * (eval_prices / start_price)

    if position_history:
        pos_values = [float(value) for value in position_history]
        pos_index = account.index[: len(pos_values)]
        position = pd.Series(pos_values, index=pos_index).reindex(eval_account.index).ffill().fillna(0.0)
    else:
        position = pd.Series(0.0, index=eval_account.index)

    action_map: dict[str, str] = {}
    for signal in trade_signals or []:
        if len(signal) < 4:
            continue
        dt, _price, _pos, signal_type = signal[:4]
        action_map[_date_label(pd.Timestamp(dt))] = str(signal_type).upper()

    daily = pd.DataFrame(
        {
            "datetime": [_date_label(dt) for dt in eval_account.index],
            "strategy_nav": strategy_nav.astype(float).values,
            "price": eval_prices.astype(float).values,
            "buyhold_nav": buyhold_nav.astype(float).values,
            "position": position.astype(float).values,
        }
    )
    daily["excess"] = daily["strategy_nav"] - daily["buyhold_nav"]
    daily["excess_pct"] = daily["excess"] / daily["buyhold_nav"] * 100.0
    daily["action"] = [action_map.get(dt, "HOLD") for dt in daily["datetime"]]
    return daily


def _window_prices(price_series: pd.Series, seg: tuple[str, str]) -> pd.Series:
    prices = price_series.sort_index().astype(float).loc[seg[0] : seg[1]].dropna()
    if prices.empty:
        raise ValueError(f"empty price window for segment {seg}")
    return prices


def _return_pct(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    if not math.isfinite(start) or start == 0.0:
        raise ValueError("invalid benchmark start price")
    return float((end / start - 1.0) * 100.0)


def compute_relative_metric_variants(
    *,
    strategy_return: float,
    price_series: pd.Series,
    requested_seg: tuple[str, str],
    effective_seg: tuple[str, str] | None = None,
    lookback: int = 0,
    target_relative_return: float = 30.0,
    material_threshold_pp: float = 5.0,
) -> dict[str, Any]:
    """Compare calendar and tradable-window relative-return metrics.

    Isolated qlib fold backtests often pass a slice beginning at the evaluation
    start date. The strategy then cannot issue orders for the first ``lookback``
    rows, while a calendar buy-and-hold benchmark can still include that warmup
    period. This helper keeps both benchmark views explicit so notebooks and
    promotion gates can label warmup-driven relative-return flips instead of
    silently mixing metric definitions.
    """

    effective = effective_seg or requested_seg
    calendar_prices = _window_prices(price_series, effective)
    requested_prices = _window_prices(price_series, requested_seg)
    end_date = calendar_prices.index[-1]
    requested_until_effective_end = requested_prices.loc[:end_date]
    if requested_until_effective_end.empty:
        requested_until_effective_end = calendar_prices

    first_tradable_idx = min(max(int(lookback), 0), len(requested_until_effective_end) - 1)
    first_tradable_date = requested_until_effective_end.index[first_tradable_idx]
    tradable_prices = requested_until_effective_end.loc[first_tradable_date:end_date]
    if len(tradable_prices) < 2:
        tradable_prices = calendar_prices
        first_tradable_date = tradable_prices.index[0]
        first_tradable_idx = 0

    calendar_bnh = _return_pct(calendar_prices)
    tradable_bnh = _return_pct(tradable_prices)
    calendar_relative = float(strategy_return) - calendar_bnh
    tradable_relative = float(strategy_return) - tradable_bnh
    calendar_pass = calendar_relative >= float(target_relative_return)
    tradable_pass = tradable_relative >= float(target_relative_return)
    gap = tradable_relative - calendar_relative
    gate_flip = bool(calendar_pass != tradable_pass)

    if gate_flip:
        reason = "warmup_benchmark_alignment_gate_flip"
    elif abs(gap) >= float(material_threshold_pp):
        reason = "material_warmup_benchmark_alignment_gap"
    else:
        reason = "metric_variants_aligned"

    return {
        "strategy_return": float(strategy_return),
        "calendar_buy_and_hold_return": float(calendar_bnh),
        "tradable_buy_and_hold_return": float(tradable_bnh),
        "calendar_relative_return": float(calendar_relative),
        "tradable_relative_return": float(tradable_relative),
        "relative_metric_gap": float(gap),
        "target_relative_return": float(target_relative_return),
        "calendar_gate": bool(calendar_pass),
        "tradable_gate": bool(tradable_pass),
        "gate_flip": gate_flip,
        "mismatch_reason": reason,
        "requested_start": str(pd.Timestamp(requested_prices.index[0]).date()),
        "requested_end": str(pd.Timestamp(requested_prices.index[-1]).date()),
        "effective_start": str(pd.Timestamp(calendar_prices.index[0]).date()),
        "effective_end": str(pd.Timestamp(calendar_prices.index[-1]).date()),
        "first_tradable_date": str(pd.Timestamp(first_tradable_date).date()),
        "warmup_days": int(first_tradable_idx),
    }


def _first_present(row: pd.Series, names: Sequence[str], default: object = None) -> object:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _source_gate(row: pd.Series, relative_return: float, max_dd: float, target_relative: float, target_max_dd: float) -> bool:
    value = _first_present(row, ("source_bc_gate", "source_gate", "source_gate_pass"), default=None)
    if value is not None:
        return _as_bool(value)
    return bool(relative_return >= target_relative and max_dd <= target_max_dd)


def _official_gate(row: pd.Series, relative_return: float, max_dd: float, target_relative: float, target_max_dd: float) -> bool:
    value = _first_present(row, ("official_gate", "window_gate_pass", "official_gate_pass"), default=None)
    if value is not None:
        return _as_bool(value)
    return bool(relative_return >= target_relative and max_dd <= target_max_dd)


def _primary_alignment_mismatch(
    *,
    source_optimistic_gate_flip: bool,
    source_gate_pass: bool,
    official_gate_pass: bool,
    material_relative: bool,
    material_maxdd: bool,
) -> str:
    if source_optimistic_gate_flip:
        if material_relative and material_maxdd:
            return "relative_and_drawdown_gate_flip"
        if material_relative:
            return "relative_return_overstated_gate_flip"
        if material_maxdd:
            return "max_drawdown_understated_gate_flip"
        return "threshold_edge_gate_flip"
    if source_gate_pass == official_gate_pass:
        if material_relative and material_maxdd:
            return "material_relative_and_drawdown_gap"
        if material_relative:
            return "material_relative_gap"
        if material_maxdd:
            return "material_drawdown_gap"
        return "aligned"
    if official_gate_pass and not source_gate_pass:
        return "source_backtest_pessimistic_gate_flip"
    return "official_gate_failed_without_source_pass"


def audit_source_vs_official_metrics(
    frame: pd.DataFrame,
    *,
    target_relative_return: float = 30.0,
    target_max_dd: float = 15.0,
    material_relative_delta_pp: float = 5.0,
    material_maxdd_delta_pp: float = 2.0,
) -> pd.DataFrame:
    """Classify ordinary fold-backtest metrics against official replay metrics.

    This is intentionally a metric classifier, not an experiment runner. Runtime
    notebooks can use it to keep promotion decisions tied to official replay
    evidence and to label source-backtest false positives before launching more
    PPO or retaining actors.
    """

    columns = [
        "candidate",
        "fold",
        "source_relative_return",
        "official_relative_return",
        "source_max_dd",
        "official_max_dd",
        "source_gate_pass",
        "official_gate_pass",
        "source_optimistic_gate_flip",
        "relative_delta_official_minus_source",
        "relative_overstatement_pp",
        "maxdd_delta_official_minus_source",
        "maxdd_understatement_pp",
        "material_relative_overstatement",
        "material_maxdd_understatement",
        "primary_mismatch",
        "required_action",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        series = pd.Series(row)
        candidate = str(_first_present(series, ("candidate", "profile"), default=""))
        fold = str(_first_present(series, ("fold", "window"), default=""))
        source_relative = _as_float(
            _first_present(series, ("source_bc_relative_return", "source_relative_return", "source_rel"), default=0.0)
        )
        official_relative = _as_float(
            _first_present(
                series,
                ("official_warm_relative_return", "official_relative_return", "relative_return"),
                default=0.0,
            )
        )
        source_max_dd = _as_float(_first_present(series, ("source_bc_max_dd", "source_max_dd"), default=0.0))
        official_max_dd = _as_float(
            _first_present(series, ("official_warm_max_dd", "official_max_dd", "max_dd"), default=0.0)
        )
        source_gate_pass = _source_gate(
            series,
            source_relative,
            source_max_dd,
            float(target_relative_return),
            float(target_max_dd),
        )
        official_gate_pass = _official_gate(
            series,
            official_relative,
            official_max_dd,
            float(target_relative_return),
            float(target_max_dd),
        )
        relative_delta = official_relative - source_relative
        relative_overstatement = max(0.0, source_relative - official_relative)
        maxdd_delta = official_max_dd - source_max_dd
        maxdd_understatement = max(0.0, official_max_dd - source_max_dd)
        material_relative = relative_overstatement >= float(material_relative_delta_pp)
        material_maxdd = maxdd_understatement >= float(material_maxdd_delta_pp)
        source_optimistic_gate_flip = bool(source_gate_pass and not official_gate_pass)
        primary_mismatch = _primary_alignment_mismatch(
            source_optimistic_gate_flip=source_optimistic_gate_flip,
            source_gate_pass=source_gate_pass,
            official_gate_pass=official_gate_pass,
            material_relative=material_relative,
            material_maxdd=material_maxdd,
        )
        rows.append(
            {
                "candidate": candidate,
                "fold": fold,
                "source_relative_return": round(float(source_relative), 6),
                "official_relative_return": round(float(official_relative), 6),
                "source_max_dd": round(float(source_max_dd), 6),
                "official_max_dd": round(float(official_max_dd), 6),
                "source_gate_pass": bool(source_gate_pass),
                "official_gate_pass": bool(official_gate_pass),
                "source_optimistic_gate_flip": source_optimistic_gate_flip,
                "relative_delta_official_minus_source": round(float(relative_delta), 6),
                "relative_overstatement_pp": round(float(relative_overstatement), 6),
                "maxdd_delta_official_minus_source": round(float(maxdd_delta), 6),
                "maxdd_understatement_pp": round(float(maxdd_understatement), 6),
                "material_relative_overstatement": bool(material_relative),
                "material_maxdd_understatement": bool(material_maxdd),
                "primary_mismatch": primary_mismatch,
                "required_action": (
                    "block_training_until_official_replay_metric_alignment"
                    if source_optimistic_gate_flip
                    else "use_official_replay_metrics_for_selection"
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["source_optimistic_gate_flip", "relative_overstatement_pp", "maxdd_understatement_pp", "candidate", "fold"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)[columns]


def summarize_source_vs_official_audit(audit: pd.DataFrame) -> dict[str, object]:
    """Summarize source-vs-official metric alignment for promotion governance."""

    if audit.empty:
        return {
            "decision": "metric_alignment_no_rows",
            "source_optimistic_gate_flip_count": 0,
            "official_gate_pass_count": 0,
            "next_mechanism": "collect_source_and_official_replay_metrics",
        }

    source_optimistic = audit["source_optimistic_gate_flip"].astype(bool)
    official_gate = audit["official_gate_pass"].astype(bool)
    if bool(source_optimistic.any()):
        decision = "source_backtest_optimistic_gate_flip_no_training"
        next_mechanism = "replace_training_selection_gate_with_official_replay_metrics"
    elif bool(official_gate.all()):
        decision = "metric_alignment_official_gate_passed"
        next_mechanism = "continue_full_walkforward_or_retention_review"
    else:
        decision = "metric_alignment_official_gate_failed"
        next_mechanism = "drop_candidate_or_redesign_before_training"

    worst = audit.iloc[0]
    return {
        "decision": decision,
        "rows": int(len(audit)),
        "source_optimistic_gate_flip_count": int(source_optimistic.sum()),
        "official_gate_pass_count": int(official_gate.sum()),
        "max_relative_overstatement_pp": float(audit["relative_overstatement_pp"].max()),
        "max_maxdd_understatement_pp": float(audit["maxdd_understatement_pp"].max()),
        "worst_candidate": str(worst["candidate"]),
        "worst_fold": str(worst["fold"]),
        "worst_primary_mismatch": str(worst["primary_mismatch"]),
        "next_mechanism": next_mechanism,
    }


__all__ = [
    "audit_source_vs_official_metrics",
    "build_warm_start_replay_segment",
    "compute_relative_metric_variants",
    "slice_warm_start_daily_path",
    "summarize_source_vs_official_audit",
]
