"""Frozen risk-mechanism explanation for the uniform layer-attribution replays.

Read-only analysis of decision-time risk fields per case (``layer::segment::cost``).
Reports when the risk controller stopped / stayed in emergency / bound its caps, the
SOXX and strategy behaviour during those phases, and a strictly trade-date-aligned
paired log-wealth difference versus the same segment/cost ``raw_allocator_only`` path.

No fitting, no parameter changes, no causal claim: this is an observable ordered-path
diagnostic. Missing / incomplete / non-aligned cases are ``needs_data`` (never dropped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECISION_FIELDS = (
    "drawdown_before", "risk_state_before", "risk_clock_before", "risk_state_after", "risk_clock_after",
    "current_beta_cap", "raw_beta", "pretrade_beta", "desired_beta", "risk_q",
    "stop_entered", "emergency_active", "trailing_loss_active",
    "clock_cap_binding", "emergency_cap_binding", "trailing_cap_binding",
    "max_soxl_binding", "min_cash_binding",
)
CORE_FIELDS = ("strategy_return", "soxx_return", "ending_cash_weight", "signal_date", "trade_date")
BINDING_FIELDS = ("clock_cap_binding", "emergency_cap_binding", "trailing_cap_binding",
                  "max_soxl_binding", "min_cash_binding")
PHASE_FIELDS = ("stop_entered", "emergency_active", "trailing_loss_active")
BOOL_FIELDS = (*PHASE_FIELDS, *BINDING_FIELDS)
NUMERIC_STATE_FIELDS = ("drawdown_before", "risk_state_before", "risk_clock_before", "risk_state_after",
                        "risk_clock_after", "raw_beta", "pretrade_beta", "desired_beta")
OPTIONAL_STATE_FIELDS = ("current_beta_cap", "risk_q")
RAW_LAYER = "raw_allocator_only"
_BOOL_VALUES = {True, False, 0, 1, 0.0, 1.0}


def _split(key: str):
    parts = key.split("::")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], int(float(parts[2]))


def _is_bool_like(series: pd.Series) -> bool:
    if series.dtype == object or series.isna().any():
        return False
    return set(pd.unique(series)) <= _BOOL_VALUES


def _validate(frame: pd.DataFrame, layer: str) -> str | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "empty_or_not_frame"
    missing = [c for c in (*CORE_FIELDS, *DECISION_FIELDS) if c not in frame.columns]
    if missing:
        return f"missing_columns:{','.join(missing)}"
    for column in ("strategy_return", "soxx_return"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            return f"nonfinite_{column}"
        if not (values > -1.0).all():
            return f"{column}_le_minus_one"
    cash = pd.to_numeric(frame["ending_cash_weight"], errors="coerce").to_numpy(float)
    if not np.isfinite(cash).all():
        return "nonfinite_ending_cash_weight"
    if not ((cash >= 0.0) & (cash <= 1.0)).all():
        return "ending_cash_weight_out_of_range"
    trade = pd.to_datetime(frame["trade_date"], errors="coerce")
    if trade.isna().any():
        return "non_datetime_trade_date"
    if trade.duplicated().any():
        return "duplicate_trade_date"
    if not trade.is_monotonic_increasing:
        return "trade_date_not_ordered"
    for column in BOOL_FIELDS:
        if not _is_bool_like(frame[column]):
            return f"nonbool_flag:{column}"
    for column in NUMERIC_STATE_FIELDS:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(float)).all():
            return f"nonfinite:{column}"
    for column in OPTIONAL_STATE_FIELDS:
        finite = np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(float)).all()
        if layer == RAW_LAYER:
            continue  # current_beta_cap / risk_q may be None for the raw layer
        if not finite:
            return f"nonfinite:{column}"
    if layer == RAW_LAYER:
        for column in BOOL_FIELDS:
            if frame[column].astype(bool).any():
                return f"raw_flag_true:{column}"
    return None


def _longest_run(values: pd.Series) -> int:
    best = run = 0
    for flag in values.astype(bool).to_numpy():
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def _emergency_entries(values: pd.Series) -> int:
    """Number of False->True entries into emergency (consecutive emergency is one episode)."""
    flags = values.astype(bool).to_numpy()
    if flags.size == 0:
        return 0
    return int(flags[0]) + int(np.sum((~flags[:-1]) & flags[1:]))


def _sumlog(frame: pd.DataFrame, mask: np.ndarray, column: str) -> float:
    values = frame.loc[mask, column].to_numpy(float)
    return float(np.sum(np.log1p(values)))


def _phase_stats(frame: pd.DataFrame) -> dict:
    out = {}
    for phase in PHASE_FIELDS:
        mask = frame[phase].astype(bool)
        n = int(mask.sum())
        entry = {"days": n, "share": float(n / len(frame))}
        if n:
            section = frame[mask]
            up = mask & (frame["soxx_return"].to_numpy(float) > 0)
            down = mask & (frame["soxx_return"].to_numpy(float) < 0)
            entry.update({
                "mean_strategy_return": float(section["strategy_return"].mean()),
                "mean_soxx_return": float(section["soxx_return"].mean()),
                "mean_ending_cash_weight": float(section["ending_cash_weight"].mean()),
                "mean_pretrade_beta": _nanmean_or_none(section["pretrade_beta"].to_numpy(float)),
                "mean_desired_beta": _nanmean_or_none(section["desired_beta"].to_numpy(float)),
                "soxx_up_days": int(up.sum()), "soxx_down_days": int(down.sum()),
                "soxx_up_sumlog_strategy": _sumlog(frame, up, "strategy_return"),
                "soxx_up_sumlog_soxx": _sumlog(frame, up, "soxx_return"),
                "soxx_down_sumlog_strategy": _sumlog(frame, down, "strategy_return"),
                "soxx_down_sumlog_soxx": _sumlog(frame, down, "soxx_return"),
            })
        out[phase] = entry
    return out


def _nanmean_or_none(values) -> float | None:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else None


def _beta(frame: pd.DataFrame) -> float | None:
    for column in ("pretrade_beta", "desired_beta", "raw_beta"):
        if column in frame and np.isfinite(frame[column].to_numpy(float)).any():
            return float(np.nanmean(frame[column].to_numpy(float)))
    return None


def _paired(frame: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Strict trade-date-aligned paired log-wealth difference vs the raw path.

    Grouped days (by risk_state_before / phase true-false) are non-continuous conditional
    subsamples of the same observational path; their log contributions are path
    decompositions, not tradable wealth or a counterfactual.
    """
    left = frame[["trade_date", "strategy_return"]].copy()
    right = raw[["trade_date", "strategy_return"]].copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    right["trade_date"] = pd.to_datetime(right["trade_date"])
    if len(left) != len(right) or not left["trade_date"].reset_index(drop=True).equals(
            right["trade_date"].reset_index(drop=True)):
        return {"status": "needs_data", "reason": "trade_date alignment with raw path is not exact"}
    lr = left["strategy_return"].to_numpy(float)
    rr = right["strategy_return"].to_numpy(float)
    diff = np.log1p(lr) - np.log1p(rr)
    by_year = {int(y): float(np.sum(g)) for y, g in
               pd.Series(diff, index=left["trade_date"]).groupby(lambda d: d.year)}
    by_state = {str(k): float(np.sum(diff[frame["risk_state_before"].to_numpy() == k]))
                for k in pd.unique(frame["risk_state_before"])}
    by_phase = {}
    for phase in PHASE_FIELDS:
        mask = frame[phase].astype(bool).to_numpy()
        by_phase[phase] = {"true": float(np.sum(diff[mask])), "false": float(np.sum(diff[~mask]))}
    phase_soxx = {}
    soxx = frame["soxx_return"].to_numpy(float)
    for phase in PHASE_FIELDS:
        pmask = frame[phase].astype(bool).to_numpy()
        phase_soxx[phase] = {}
        for name, dmask in (("soxx_up", pmask & (soxx > 0)), ("soxx_down", pmask & (soxx < 0))):
            phase_soxx[phase][name] = {
                "sumlog_strategy": _sumlog(frame, dmask, "strategy_return"),
                "sumlog_soxx": _sumlog(frame, dmask, "soxx_return"),
                "paired_raw_sumlog_strategy": _sumlog(raw, dmask, "strategy_return"),
            }
    return {"status": "ok", "paired_log_wealth_diff": float(np.sum(diff)),
            "by_year": by_year, "by_risk_state_before": by_state, "by_phase": by_phase,
            "phase_soxx": phase_soxx,
            "note": "observational path decomposition; grouped days are non-continuous "
                    "conditional subsamples, not tradable wealth or a counterfactual"}


def analyze_risk_mechanism(replays: dict[str, pd.DataFrame]) -> list[dict]:
    """Return one grouped small table (dict) per ``layer::segment::cost`` case."""
    by_key = {}
    for key, frame in replays.items():
        parsed = _split(key)
        if parsed is None:
            continue
        by_key[key] = (parsed, frame)

    results = []
    for key, ((layer, segment, cost), frame) in sorted(by_key.items(), key=lambda kv: kv[0]):
        reason = _validate(frame, layer)
        if reason is not None:
            results.append({"case": key, "layer": layer, "segment": segment, "cost_bps": cost,
                            "status": "needs_data", "reason": reason})
            continue
        stats = {
            "case": key, "layer": layer, "segment": segment, "cost_bps": cost, "status": "ok",
            "sessions": int(len(frame)),
            "stop_events": int(frame["stop_entered"].astype(bool).sum()),
            "emergency_entries": _emergency_entries(frame["emergency_active"]),
            "emergency_longest_consecutive_days": _longest_run(frame["emergency_active"]),
            "state_share": {str(k): float(v) for k, v in
                            frame["risk_state_before"].value_counts(normalize=True, dropna=False).items()},
            "binding_share": {f: float(frame[f].astype(bool).mean()) for f in BINDING_FIELDS},
            "mean_drawdown_before": _nanmean_or_none(frame["drawdown_before"].to_numpy(float)),
            "mean_beta": _beta(frame),
            "phases": _phase_stats(frame),
            "mean_strategy_return": float(frame["strategy_return"].mean()),
            "mean_soxx_return": float(frame["soxx_return"].mean()),
            "mean_ending_cash_weight": float(frame["ending_cash_weight"].mean()),
        }
        raw_key = f"{RAW_LAYER}::{segment}::{cost}"
        if layer == RAW_LAYER:
            stats["paired_vs_raw"] = {"status": "self_raw_reference"}
        elif raw_key in by_key and _validate(by_key[raw_key][1], RAW_LAYER) is None:
            stats["paired_vs_raw"] = _paired(frame, by_key[raw_key][1])
        else:
            stats["paired_vs_raw"] = {"status": "needs_data", "reason": "raw reference missing or invalid"}
        results.append(stats)
    return results
