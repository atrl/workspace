"""Minute-timed, same-state Tiered labels for a bounded beta-neutral allocation."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.event_guard_clock import guard_slot
from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


def build_labels(frame, book, quotes, *, closes, tariff, initial_cash, action):
    frame = frame.reset_index(drop=True)
    authority = book["authority"].reset_index(drop=True)
    targets = book["targets"].reset_index(drop=True)
    if (len(frame) != len(authority) or len(targets) != len(frame)
            or not frame.trade_date.equals(authority.trade_date)
            or not targets.trade_date.equals(frame.trade_date)):
        raise ValueError("blocked: dated parent calendar differs")
    orders = book["orders"].sort_values("stamp")
    by_day = {pd.Timestamp(day): part for day, part in orders.groupby("trade_date", sort=True)}
    fees = TieredFees(tariff, slippage_bps=0)
    records, masks = [], []
    max_parent_error = 0.0
    total_slots = 0
    for index, row in frame.iterrows():
        day = pd.Timestamp(row.trade_date)
        actual = authority.iloc[index]
        current_orders = by_day.get(day, pd.DataFrame())
        asset = quotes.loc[quotes.index.normalize() == day]
        source_valid = (not asset.empty and
                        bool(asset[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all()))
        status = ("terminal" if index + 1 == len(frame) else
                  "recovery" if bool(actual.recovery_action) else
                  "source_invalid" if not source_valid else "eligible")
        if status == "eligible":
            opened = day + pd.Timedelta(hours=9, minutes=47)
            pos = book["native_positions"].get(opened)
            if pos is None:
                raise ValueError("blocked: missing own after-opening position")
            q = np.array([pos.get_stock_amount(s) if pos.check_stock(s) else 0.
                          for s in ("SOXL", "SOXX")], dtype=float)
            cash = float(pos.get_cash())
            next_row = frame.iloc[index + 1]
            next_stamp = pd.Timestamp(next_row.trade_date) + pd.Timedelta(hours=9, minutes=47)
            next_prices = next_row[["soxl_open", "soxx_open"]].to_numpy(float)
            next_raw = next_row[["raw_soxl_open", "raw_soxx_open"]].to_numpy(float)
            reset = targets.iloc[index + 1][["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(float)
            base_before = float(q @ next_prices + cash)
            error = abs(base_before - float(actual.native_expected_nav))
            max_parent_error = max(max_parent_error, error)
            if error > 1e-7:
                raise ValueError("blocked: parent next-open NAV differs")
            branch_fees = deepcopy(fees)
            for order in current_orders.itertuples(index=False):
                branch_fees.commit(int(order.raw_shares), stamp=order.stamp,
                                   cap_applied=bool(order.cap_applied))
            baseline = settle_tiered_event(q, cash, desired=reset,
                adjusted_prices=next_prices, raw_prices=next_raw,
                stamp=next_stamp, fees=deepcopy(branch_fees))
            base_after = float(baseline["quantities"] @ next_prices + baseline["cash"])
            close = closes[day]
            valid = ((asset.index >= opened + pd.Timedelta(minutes=1)) &
                     (asset.index + pd.Timedelta(minutes=2) < close))
            event = asset.loc[valid]
            total_slots += len(event)
            for bar in event.itertuples():
                slot = guard_slot(bar.Index, session_open=day + pd.Timedelta(hours=9, minutes=30),
                                  session_close=close, original_open_fill=opened)
                if slot is None:
                    raise ValueError("blocked: minute event clock differs")
                decision_prices = np.array([bar.SOXL_close, bar.SOXX_close], float)
                nav = float(q @ decision_prices + cash)
                if not np.isfinite(nav) or nav <= 0:
                    raise ValueError("blocked: minute decision NAV invalid")
                weights = np.r_[q * decision_prices, cash] / nav
                delta = min(float(action["max_soxl_weight_reduction"]),
                            float(weights[0]),
                            max(0., (float(weights[2]) - action["minimum_cash_weight_after_trade"]) / 2),
                            max(0., (1. - float(weights[1])) / 3))
                rec = {"trade_date": day, "decision_at": slot.decision_at,
                       "fill_at": slot.fill_at, "bar_end_at": slot.bar_end_at,
                       "local_advantage": np.nan, "label_available_at": pd.NaT,
                       "funded_change": False, "reason": "no_target_room",
                       "decision_soxl_weight": float(weights[0]),
                       "decision_soxx_weight": float(weights[1]),
                       "decision_cash_weight": float(weights[2]),
                       "current_beta_cap": float(actual.current_beta_cap),
                       "delta_soxl_weight": max(0., delta)}
                if delta > 1e-10:
                    desired = weights + np.array([-delta, 3 * delta, -2 * delta])
                    if (abs(desired.sum() - 1) > 1e-8 or desired.min() < -1e-8 or
                            abs(3 * desired[0] + desired[1] - (3 * weights[0] + weights[1])) > 1e-8):
                        raise ValueError("blocked: beta-neutral target invalid")
                    quote = asset.loc[slot.fill_at + pd.Timedelta(minutes=1)]
                    fill_prices = quote[["SOXL_open", "SOXX_open"]].to_numpy(float)
                    fill_raw = quote[["SOXL_raw_open", "SOXX_raw_open"]].to_numpy(float)
                    local_fees = deepcopy(branch_fees)
                    traded = settle_tiered_event(q, cash, desired=desired,
                        adjusted_prices=fill_prices, raw_prices=fill_raw,
                        stamp=slot.fill_at, fees=local_fees)
                    sold = any(item["symbol"] == "SOXL" and item["sell"] for item in traded["orders"])
                    bought = any(item["symbol"] == "SOXX" and not item["sell"] for item in traded["orders"])
                    post_nav = float(traded["quantities"] @ fill_prices + traded["cash"])
                    post_cash = float(traded["cash"] / post_nav)
                    parent_beta = float((3 * q[0] * fill_prices[0] + q[1] * fill_prices[1]) /
                                        (q @ fill_prices + cash))
                    post_beta = float((3 * traded["quantities"][0] * fill_prices[0]
                                       + traded["quantities"][1] * fill_prices[1]) / post_nav)
                    if not (sold and bought):
                        rec["reason"] = "no_whole_share_pair"
                    elif post_cash < action["minimum_cash_weight_after_trade"] - 1e-8:
                        rec["reason"] = "cash_floor_after_fee"
                    elif post_beta > parent_beta + 1e-3:
                        rec["reason"] = "rounding_increased_beta"
                    else:
                        restored = settle_tiered_event(traded["quantities"], traded["cash"],
                            desired=reset, adjusted_prices=next_prices, raw_prices=next_raw,
                            stamp=next_stamp, fees=local_fees)
                        cand_after = float(restored["quantities"] @ next_prices + restored["cash"])
                        rec.update(local_advantage=(cand_after - base_after) / nav,
                                   label_available_at=next_stamp + pd.Timedelta(minutes=2),
                                   funded_change=True, reason="funded")
                records.append(rec)
        masks.append({"trade_date": day, "status": status})
        for order in current_orders.itertuples(index=False):
            fees.commit(int(order.raw_shares), stamp=order.stamp,
                        cap_applied=bool(order.cap_applied))
    labels = pd.DataFrame(records)
    if total_slots != len(labels):
        raise ValueError("blocked: missing minute candidate rows")
    if not labels.empty and labels.loc[labels.funded_change, "label_available_at"].le(
            labels.loc[labels.funded_change, "decision_at"]).any():
        raise ValueError("blocked: label matures before decision")
    mask = pd.DataFrame(masks)
    audit = {"sessions": len(frame), "event_slots": total_slots,
             "status_counts": mask.status.value_counts().to_dict(),
             "reason_counts": labels.reason.value_counts().to_dict(),
             "funded_rows": int(labels.funded_change.sum()),
             "funded_days": int(labels.loc[labels.funded_change, "trade_date"].nunique()),
             "parent_nav_max_error_usd": max_parent_error}
    return labels, mask, audit
