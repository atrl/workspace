"""Same-state, equal-beta SOXL/SOXX/HOLD minute value discovery labels."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.event_guard_clock import guard_slot
from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


def build_labels(frame, book, quotes, *, closes, tariff, delta_beta: float = .10):
    frame = frame.reset_index(drop=True)
    authority = book["authority"].reset_index(drop=True)
    if len(frame) != len(authority) or not frame.trade_date.equals(authority.trade_date):
        raise ValueError("blocked: dated parent calendar differs")
    orders = book["orders"].sort_values("stamp")
    by_day = {pd.Timestamp(day): part for day, part in orders.groupby("trade_date", sort=True)}
    fees = TieredFees(tariff, slippage_bps=0)
    records, masks = [], []
    maximum_parent_error = 0.
    event_slots = 0
    for index, row in frame.iterrows():
        day = pd.Timestamp(row.trade_date)
        actual = authority.iloc[index]
        current_orders = by_day.get(day, pd.DataFrame())
        asset = quotes.loc[quotes.index.normalize() == day]
        source_valid = not asset.empty and bool(asset[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all().all())
        status = ("terminal" if index + 1 == len(frame) else
                  "original_recovery" if bool(actual.recovery_action) else
                  "source_invalid" if not source_valid else "eligible")
        if status == "eligible":
            opened = day + pd.Timedelta(hours=9, minutes=47)
            position = book["native_positions"].get(opened)
            if position is None:
                raise ValueError("blocked: after-open parent position absent")
            q = np.array([position.get_stock_amount(s) if position.check_stock(s) else 0.
                          for s in ("SOXL", "SOXX")], float)
            cash = float(position.get_cash())
            next_row = frame.iloc[index + 1]
            next_stamp = pd.Timestamp(next_row.trade_date) + pd.Timedelta(hours=9, minutes=47)
            next_prices = next_row[["soxl_open", "soxx_open"]].to_numpy(float)
            hold_nav = float(q @ next_prices + cash)
            error = abs(hold_nav - float(actual.native_expected_nav))
            maximum_parent_error = max(maximum_parent_error, error)
            if error > 1e-7:
                raise ValueError("blocked: HOLD next-open parent NAV differs")
            branch_fees = deepcopy(fees)
            for order in current_orders.itertuples(index=False):
                if order.slot == "open":
                    branch_fees.commit(int(order.raw_shares), stamp=order.stamp,
                                       cap_applied=bool(order.cap_applied))
            close = closes[day]
            eligible = asset.loc[(asset.index >= opened + pd.Timedelta(minutes=1)) &
                                 (asset.index + pd.Timedelta(minutes=2) < close)]
            event_slots += len(eligible)
            for bar in eligible.itertuples():
                slot = guard_slot(bar.Index, session_open=day + pd.Timedelta(hours=9, minutes=30),
                                  session_close=close, original_open_fill=opened)
                if slot is None:
                    raise ValueError("blocked: event clock differs")
                mark = np.array([bar.SOXL_close, bar.SOXX_close], float)
                nav = float(q @ mark + cash)
                if not np.isfinite(nav) or nav <= 0:
                    raise ValueError("blocked: decision NAV invalid")
                weights = np.r_[q * mark, cash] / nav
                rec = {"trade_date": day, "bar_end_at": slot.bar_end_at,
                       "decision_at": slot.decision_at, "fill_at": slot.fill_at,
                       "label_available_at": next_stamp + pd.Timedelta(minutes=2),
                       "decision_nav": nav, "decision_cash_weight": float(weights[2]),
                       "hold_next_open_nav": hold_nav,
                       "soxl_value": np.nan, "soxx_value": np.nan,
                       "soxl_funded": False, "soxx_funded": False,
                       "relative_value_soxl_minus_soxx": np.nan}
                if weights[2] >= delta_beta - 1e-10:
                    fillbar = asset.loc[slot.fill_at + pd.Timedelta(minutes=1)]
                    adjusted = np.array([fillbar.SOXL_open, fillbar.SOXX_open], float)
                    raw = np.array([fillbar.SOXL_raw_open, fillbar.SOXX_raw_open], float)
                    targets = {"SOXL": weights + np.array([delta_beta / 3, 0., -delta_beta / 3]),
                               "SOXX": weights + np.array([0., delta_beta, -delta_beta])}
                    for symbol, target in targets.items():
                        if target.min() < -1e-10 or abs(target.sum() - 1.) > 1e-8:
                            continue
                        trade = settle_tiered_event(q, cash, desired=target,
                            adjusted_prices=adjusted, raw_prices=raw, stamp=slot.fill_at,
                            fees=deepcopy(branch_fees))
                        after = float(trade["quantities"] @ adjusted + trade["cash"])
                        orders_now = trade["orders"]
                        funded = (after > 0 and trade["cash"] >= -1e-8 and
                                  (trade["quantities"] >= 0).all() and len(orders_now) == 1 and
                                  orders_now[0]["symbol"] == symbol and not orders_now[0]["sell"])
                        if funded:
                            ending = float(trade["quantities"] @ next_prices + trade["cash"])
                            rec[symbol.lower() + "_value"] = (ending - hold_nav) / nav
                            rec[symbol.lower() + "_funded"] = True
                    if rec["soxl_funded"] and rec["soxx_funded"]:
                        rec["relative_value_soxl_minus_soxx"] = rec["soxl_value"] - rec["soxx_value"]
                records.append(rec)
        masks.append({"trade_date": day, "status": status})
        for order in current_orders.itertuples(index=False):
            fees.commit(int(order.raw_shares), stamp=order.stamp,
                        cap_applied=bool(order.cap_applied))
    labels = pd.DataFrame(records)
    if len(labels) != event_slots:
        raise ValueError("blocked: event slots missing")
    if labels.loc[labels.soxl_funded | labels.soxx_funded, "label_available_at"].le(
            labels.loc[labels.soxl_funded | labels.soxx_funded, "decision_at"]).any():
        raise ValueError("blocked: label matures before decision")
    mask = pd.DataFrame(masks)
    paired = labels.loc[labels.soxl_funded & labels.soxx_funded]
    audit = {"sessions": len(frame), "event_slots": event_slots,
             "status_counts": mask.status.value_counts().to_dict(),
             "soxl_funded_rows": int(labels.soxl_funded.sum()),
             "soxx_funded_rows": int(labels.soxx_funded.sum()),
             "paired_rows": len(paired),
             "paired_days": int(paired.trade_date.nunique()),
             "parent_nav_max_error_usd": maximum_parent_error}
    return labels, mask, audit
