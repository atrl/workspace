"""Own-state, fee-net minute switch labels on dated Stage1 development accounts."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.event_guard_clock import guard_slot
from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


def _features(day: pd.Timestamp, minute: pd.DataFrame, *, q: np.ndarray, cash: float,
              known_drawdown: float, cap: float, alpha: float, tail: float,
              close_at: pd.Timestamp) -> pd.DataFrame:
    values = minute[["SOXL_close", "SOXX_close"]].to_numpy(float)
    log_price = np.log(values)
    changes = np.vstack([np.zeros((1, 2)), np.diff(log_price, axis=0)])
    relative = changes[:, 0] - 3 * changes[:, 1]
    rows = []
    for index, stamp in enumerate(minute.index):
        slot = guard_slot(stamp, session_open=day + pd.Timedelta(hours=9, minutes=30),
                          session_close=close_at,
                          original_open_fill=day + pd.Timedelta(hours=9, minutes=47))
        if slot is None or index < 15:
            continue
        nav = float(q @ values[index] + cash)
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("blocked: event-time known account NAV invalid")
        trailing = relative[index - 14:index + 1]
        downside = np.minimum(trailing, 0.)
        rows.append({"trade_date": day, "bar_end_at": stamp,
            "decision_at": slot.decision_at, "fill_at": slot.fill_at,
            "soxl_weight": q[0] * values[index, 0] / nav,
            "soxx_weight": q[1] * values[index, 1] / nav,
            "cash_weight": cash / nav, "known_drawdown": known_drawdown,
            "current_beta_cap": cap, "alpha_probability": alpha,
            "tail_probability": tail,
            "time_to_close_minutes": (close_at - slot.decision_at).total_seconds() / 60,
            "soxl_ret5": log_price[index, 0] - log_price[index - 5, 0],
            "soxx_ret5": log_price[index, 1] - log_price[index - 5, 1],
            "soxl_ret15": log_price[index, 0] - log_price[index - 15, 0],
            "soxx_ret15": log_price[index, 1] - log_price[index - 15, 1],
            "relative_downside_share15": float(np.square(downside).sum()
                / (np.square(trailing).sum() + 1e-12))})
    return pd.DataFrame(rows)


def build_labels(frame: pd.DataFrame, book: dict, quotes: pd.DataFrame, *, tariff: dict,
                 session_closes: dict[pd.Timestamp, pd.Timestamp],
                 initial_cash: float) -> tuple[pd.DataFrame, dict]:
    authority = book["authority"].reset_index(drop=True)
    targets = book["targets"].reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    if (len(frame) != len(authority) or not frame.trade_date.equals(authority.trade_date)
            or not targets.trade_date.equals(frame.trade_date)):
        raise ValueError("blocked: event label parent identity differs")
    orders = book["orders"].sort_values("stamp")
    by_day = {pd.Timestamp(day): part for day, part in orders.groupby("trade_date", sort=True)}
    fees = TieredFees(tariff, slippage_bps=0)
    all_rows, eligible_days, masked = [], set(), []
    max_parent_error = 0.
    for index, row in frame.iterrows():
        day = pd.Timestamp(row.trade_date)
        actual = authority.iloc[index]
        asset = quotes.loc[quotes.index.normalize() == day]
        valid = bool(asset[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all())
        position = book["native_positions"].get(day + pd.Timedelta(hours=9, minutes=47))
        if position is None:
            raise ValueError("blocked: dated account has no after-opening native position")
        q = np.array([position.get_stock_amount(s) if position.check_stock(s) else 0.
                      for s in ("SOXL", "SOXX")], dtype=float)
        cash = float(position.get_cash())
        raw_sellable = q[0] * float(row.soxl_open) / float(row.raw_soxl_open)
        if index + 1 == len(frame) or actual.recovery_action or not valid or raw_sellable < 1.:
            masked.append({"trade_date": str(day.date()), "terminal": index + 1 == len(frame),
                "recovery": bool(actual.recovery_action), "source_valid": valid,
                "soxl_sellable": bool(raw_sellable >= 1.)})
        else:
            nxt = frame.iloc[index + 1]
            reset = targets.iloc[index + 1][["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float)
            next_price = nxt[["soxl_open", "soxx_open"]].to_numpy(float)
            next_raw = nxt[["raw_soxl_open", "raw_soxx_open"]].to_numpy(float)
            next_stamp = pd.Timestamp(nxt.trade_date) + pd.Timedelta(hours=9, minutes=47)
            base_fee = deepcopy(fees)
            baseline = settle_tiered_event(q, cash, desired=reset,
                adjusted_prices=next_price, raw_prices=next_raw, stamp=next_stamp, fees=base_fee)
            base_after = float(baseline["quantities"] @ next_price + baseline["cash"])
            baseline_before = float(q @ next_price + cash)
            error = abs(baseline_before - float(actual.native_expected_nav))
            max_parent_error = max(max_parent_error, error)
            if error > 1e-7:
                raise ValueError("blocked: event label baseline misses dated account next-open NAV")
            features = _features(day, asset, q=q, cash=cash,
                known_drawdown=float(targets.iloc[index].known_drawdown),
                cap=float(actual.current_beta_cap),
                alpha=float(row.alpha_probability), tail=float(row.tail_probability),
                close_at=session_closes[day])
            if not features.empty:
                eligible_days.add(day)
            for event in features.itertuples(index=False):
                stamp = pd.Timestamp(event.fill_at)
                adjusted = asset.loc[stamp + pd.Timedelta(minutes=1),
                    ["SOXL_open", "SOXX_open"]].to_numpy(float)
                raw = asset.loc[stamp + pd.Timedelta(minutes=1),
                    ["SOXL_raw_open", "SOXX_raw_open"]].to_numpy(float)
                current_nav = float(q @ adjusted + cash)
                desired = np.array([0., (q @ adjusted) / current_nav, cash / current_nav])
                branch_fee = deepcopy(fees)
                traded = settle_tiered_event(q, cash, desired=desired,
                    adjusted_prices=adjusted, raw_prices=raw, stamp=stamp, fees=branch_fee)
                sold = any(item["symbol"] == "SOXL" and item["sell"] for item in traded["orders"])
                bought = any(item["symbol"] == "SOXX" and not item["sell"] for item in traded["orders"])
                if not (sold and bought):
                    all_rows.append({**event._asdict(), "local_advantage": np.nan,
                        "label_available_at": pd.NaT, "funded_change": False})
                    continue
                restored = settle_tiered_event(traded["quantities"], traded["cash"],
                    desired=reset, adjusted_prices=next_price, raw_prices=next_raw,
                    stamp=next_stamp, fees=branch_fee)
                after = float(restored["quantities"] @ next_price + restored["cash"])
                all_rows.append({**event._asdict(),
                    "local_advantage": (after - base_after) / current_nav,
                    "label_available_at": next_stamp + pd.Timedelta(minutes=2),
                    "funded_change": True})
        for order in by_day.get(day, pd.DataFrame()).itertuples(index=False):
            fees.commit(int(order.raw_shares), stamp=order.stamp,
                        cap_applied=bool(order.cap_applied))
    labels = pd.DataFrame(all_rows)
    if labels.empty or labels.label_available_at.dropna().le(labels.decision_at.loc[labels.label_available_at.notna()]).any():
        raise ValueError("blocked: event labels absent or available before outcomes")
    masked_frame = pd.DataFrame(masked)
    audit = {"calendar_days": len(frame), "candidate_days": len(eligible_days),
        "rows": len(labels), "mature_funded_rows": int(labels.funded_change.sum()),
        "parent_nav_max_error_usd": max_parent_error,
        "masked_days": {"terminal": int(masked_frame.terminal.sum()),
            "recovery": int(masked_frame.recovery.sum()),
            "source_invalid": int((~masked_frame.source_valid).sum()),
            "no_sellable_soxl": int((~masked_frame.soxl_sellable).sum())}}
    return labels, audit
