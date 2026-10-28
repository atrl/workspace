"""Same-state Tiered local values for complementary Stage1 opening actions."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


def _candidate_target(base: np.ndarray, *, binding: bool, cap: float) -> tuple[str, np.ndarray] | None:
    soxl, soxx, cash = map(float, base)
    if binding:
        delta_beta = min(.10, 3 * max(0., cash - .15), 3 * max(0., .30 - soxl))
        if delta_beta <= 1e-10:
            return None
        target = np.array([soxl + delta_beta / 3, soxx, cash - delta_beta / 3])
        if 3 * target[0] + target[1] > cap + .10 + 1e-8:
            raise ValueError("blocked: binding cash deployment exceeds frozen cap extension")
        name = "binding_cash_to_soxl"
    else:
        delta = min(max(0., soxl), max(0., (cash - .15) / 2), max(0., (1 - soxx) / 3))
        if delta <= 1e-10:
            return None
        target = np.array([soxl - delta, soxx + 3 * delta, cash - 2 * delta])
        if abs(3 * target[0] + target[1] - (3 * soxl + soxx)) > 1e-8:
            raise ValueError("blocked: nonbinding rotation changes nominal beta")
        name = "nonbinding_soxl_cash_to_soxx"
    if not np.isfinite(target).all() or target.min() < -1e-8 or abs(target.sum() - 1) > 1e-8:
        raise ValueError("blocked: candidate target leaves funded simplex")
    if target[0] > .30 + 1e-8 or target[2] < .15 - 1e-8:
        raise ValueError("blocked: candidate target exceeds frozen structural limits")
    return name, target


def _branch_value(q, cash, *, row, next_row, target, reset, fees: TieredFees) -> tuple[float, float, np.ndarray, float]:
    local = deepcopy(fees)
    opened = settle_tiered_event(q, cash, desired=target,
        adjusted_prices=row[["soxl_open", "soxx_open"]],
        raw_prices=row[["raw_soxl_open", "raw_soxx_open"]],
        stamp=row.trade_date + pd.Timedelta(hours=9, minutes=47), fees=local)
    next_prices = next_row[["soxl_open", "soxx_open"]].to_numpy(float)
    before_reset = float(opened["quantities"] @ next_prices + opened["cash"])
    restored = settle_tiered_event(opened["quantities"], opened["cash"], desired=reset,
        adjusted_prices=next_prices, raw_prices=next_row[["raw_soxl_open", "raw_soxx_open"]],
        stamp=next_row.trade_date + pd.Timedelta(hours=9, minutes=47), fees=local)
    after_reset = float(restored["quantities"] @ next_prices + restored["cash"])
    return before_reset, after_reset, opened["quantities"], float(opened["cash"])


def build_labels(frame: pd.DataFrame, book: dict, *, tariff: dict, initial_cash: float) -> tuple[pd.DataFrame, dict]:
    frame = frame.reset_index(drop=True)
    authority = book["authority"].reset_index(drop=True)
    targets = book["targets"].reset_index(drop=True)
    orders = book["orders"].sort_values("stamp")
    if (len(frame) != len(authority) or len(targets) != len(frame)
            or not frame.trade_date.equals(authority.trade_date)
            or not targets.trade_date.equals(frame.trade_date)
            or authority.recovery_action.any()):
        raise ValueError("blocked: matched no-recovery training account differs")
    by_day = {pd.Timestamp(day): part for day, part in orders.groupby("trade_date", sort=True)}
    fees = TieredFees(tariff, slippage_bps=0)
    q, cash = np.zeros(2), float(initial_cash)
    records = []
    maximum_reference_error = 0.
    for index, row in frame.iterrows():
        actual = authority.iloc[index]
        base = targets.iloc[index][["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float)
        raw_beta = 3 * float(row.soxl_weight) + float(row.soxx_weight)
        cap = float(actual.current_beta_cap)
        binding = raw_beta > cap + 1e-9
        candidate = _candidate_target(base, binding=binding, cap=cap) if bool(actual.execute_open_target) else None
        record = {"signal_date": row.signal_date, "trade_date": row.trade_date,
            "decision_at": row.trade_date + pd.Timedelta(hours=9, minutes=46),
            "label_available_at": pd.NaT, "binding": bool(binding),
            "execute": bool(actual.execute_open_target),
            "route": candidate[0] if candidate else "unsupported",
            "eligible": False, "target_supported": candidate is not None,
            "funded_action_changed": False,
            "local_advantage": np.nan, "candidate_target_soxl": np.nan,
            "candidate_target_soxx": np.nan, "candidate_target_cash": np.nan,
            "pretrade_soxl_weight": np.nan, "pretrade_soxx_weight": np.nan,
            "pretrade_cash_weight": np.nan, "known_drawdown": float(targets.iloc[index].known_drawdown),
            "current_beta_cap": cap, "raw_beta": raw_beta,
            "raw_soxl_weight": float(row.soxl_weight), "raw_soxx_weight": float(row.soxx_weight),
            "raw_cash_weight": float(row.cash_weight), "alpha_probability": float(row.alpha_probability),
            "tail_probability": float(row.tail_probability)}
        known = row[["known_soxl", "known_soxx"]].to_numpy(float)
        nav = float(q @ known + cash)
        record.update(dict(zip(("pretrade_soxl_weight", "pretrade_soxx_weight", "pretrade_cash_weight"),
                               np.r_[q * known, cash] / nav)))
        if candidate is not None and index + 1 < len(frame):
            next_row = frame.iloc[index + 1]
            reset = targets.iloc[index + 1][["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float)
            base_before, base_after, base_q, base_cash = _branch_value(q, cash, row=row, next_row=next_row,
                target=base, reset=reset, fees=fees)
            cand_before, cand_after, cand_q, cand_cash = _branch_value(q, cash, row=row, next_row=next_row,
                target=candidate[1], reset=reset, fees=fees)
            error = abs(base_before - float(actual.native_expected_nav))
            maximum_reference_error = max(maximum_reference_error, error)
            if error > 1e-7:
                raise ValueError("blocked: local baseline does not reproduce own-state funded account")
            if not np.isfinite(cand_before + cand_after + base_after):
                raise ValueError("blocked: nonfinite local branch value")
            changed = bool(np.max(np.abs(base_q - cand_q)) > 1e-8 or abs(base_cash - cand_cash) > 1e-7)
            record["funded_action_changed"] = changed
            record["eligible"] = changed
            if changed:
                record["local_advantage"] = (cand_after - base_after) / nav
                record["label_available_at"] = next_row.trade_date + pd.Timedelta(hours=9, minutes=49)
            record.update(dict(zip(("candidate_target_soxl", "candidate_target_soxx", "candidate_target_cash"), candidate[1])))
        records.append(record)
        q = actual[["ending_soxl_quantity", "ending_soxx_quantity"]].to_numpy(float)
        cash = float(actual.ending_cash)
        for order in by_day.get(pd.Timestamp(row.trade_date), pd.DataFrame()).itertuples(index=False):
            fees.commit(int(order.raw_shares), stamp=order.stamp, cap_applied=bool(order.cap_applied))
    out = pd.DataFrame(records)
    if out.loc[out.eligible, "label_available_at"].le(out.loc[out.eligible, "decision_at"]).any():
        raise ValueError("blocked: local label known before its outcome")
    audit = {"sessions": len(out), "maximum_reference_nav_error_usd": maximum_reference_error,
        "eligible": int(out.eligible.sum()), "mature": int(out.local_advantage.notna().sum()),
        "routes": {route: int((out.route == route).sum()) for route in ("binding_cash_to_soxl", "nonbinding_soxl_cash_to_soxx")}}
    return out, audit
