"""Fixed-share source gate and one full five-session path-label batch.

All completed-minute decision intents remain visible. Future fill and path
coverage are label status only. Source scan runs before any economic value.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd

from src.analysis.stage1_capfree_lock_batch import _fee_snapshots, _opening_peaks, _position
from src.analysis.stage1_capfree_lock_vectorized import compile_lock_price_plane, value_lock_event
from src.backtest.event_guard_clock import guard_slot
from src.backtest.stage1_fixed_share_event import fixed_buy_intent, settle_fixed_buy


def _day_intents(day, asset, q0, cash0, q1, cash1, opening_peak, fee_pair):
    first = day + pd.Timedelta(hours=9, minutes=48)
    recovery_clock = day + pd.Timedelta(hours=10, minutes=32)
    valid = asset[["SOXL_price_valid", "SOXX_price_valid"]].eq(1).all(axis=1)
    invalid_prefix = (~valid.loc[valid.index >= first]).cumsum()
    early = asset.loc[(asset.index >= first) & (asset.index <= recovery_clock)]
    early_prices = early[["SOXL_close", "SOXX_close"]].to_numpy(float)
    early_nav = early_prices @ q0 + cash0
    early_peaks = np.maximum(float(opening_peak), np.maximum.accumulate(early_nav))
    early_lookup = dict(zip(early.index, early_peaks))
    recovery_peak = float(early_peaks[-1]) if len(early_peaks) else float(opening_peak)
    late = asset.loc[asset.index > recovery_clock]
    late_nav = late[["SOXL_close", "SOXX_close"]].to_numpy(float) @ q1 + cash1
    late_peaks = np.maximum(recovery_peak, np.maximum.accumulate(late_nav))
    late_lookup = dict(zip(late.index, late_peaks))
    for stamp in asset.index:
        if stamp < first:
            continue
        slot = guard_slot(stamp, session_open=day + pd.Timedelta(hours=9, minutes=30),
            session_close=asset.index.max(), original_open_fill=day + pd.Timedelta(hours=9, minutes=47))
        if slot is None or not (slot.fill_at < recovery_clock or stamp > recovery_clock):
            continue
        if not bool(valid.loc[stamp]) or int(invalid_prefix.loc[stamp]) > 0:
            continue
        is_late = stamp > recovery_clock
        q, cash = (q1, cash1) if is_late else (q0, cash0)
        mark = asset.loc[stamp]
        price = mark[["SOXL_close", "SOXX_close"]].to_numpy(float)
        nav = float(q @ price + cash)
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("blocked: decision NAV invalid")
        if cash / nav < .10 - 1e-10:
            continue
        peak = float(late_lookup[stamp] if is_late else early_lookup[stamp])
        initial_peak = recovery_peak if is_late else float(opening_peak)
        yield {"stamp":stamp,"slot":slot,"q":q,"cash":cash,"nav":nav,
               "mark":mark,"peak":peak,"initial_peak":initial_peak,
               "fees":fee_pair[1 if is_late else 0],
               "start_at":day + pd.Timedelta(hours=10, minutes=33) if is_late else first,
               "phase":"after_1032" if is_late else "before_1032",
               "valid":valid}


def _context(frame, book, quotes, perday_support, tariff):
    frame = frame.reset_index(drop=True)
    path = book["independent_replay"].reset_index(drop=True)
    support = perday_support.reset_index(drop=True)
    if (len(frame) != len(path) or len(frame) != len(support) or
            not frame.trade_date.equals(path.trade_date) or
            not frame.trade_date.equals(pd.to_datetime(support.trade_date))):
        raise ValueError("blocked: dated source/support calendars differ")
    by_day = {pd.Timestamp(day):part for day,part in quotes.groupby(quotes.index.normalize(),sort=True)}
    if set(frame.trade_date)-set(by_day):
        raise ValueError("blocked: Qlib minute parent day absent")
    fee_book = _fee_snapshots(frame, book["orders"], tariff)
    peaks, peak_error = _opening_peaks(frame, book)
    return frame,path,support,by_day,fee_book,peaks,peak_error


def _day_positions(book, day):
    p0 = book["native_positions"].get(day + pd.Timedelta(hours=9, minutes=47))
    p1 = book["native_positions"].get(day + pd.Timedelta(hours=10, minutes=32))
    if p0 is None or p1 is None:
        raise ValueError("blocked: dated parent position absent")
    return (*_position(p0), *_position(p1))


def _path_ready(source, asset, intent):
    if not bool(source.five_session_mature) or not bool(source.future_four_full_source_valid):
        return False
    valid = intent["valid"]
    fill_stamp = intent["slot"].fill_at + pd.Timedelta(minutes=1)
    return (fill_stamp in asset.index and bool(valid.loc[fill_stamp]) and
            bool(valid.loc[valid.index >= fill_stamp].all()))


def _arms(intent, asset):
    fill_stamp = intent["slot"].fill_at + pd.Timedelta(minutes=1)
    bar = asset.loc[fill_stamp]
    adjusted = bar[["SOXL_open", "SOXX_open"]].to_numpy(float)
    raw = bar[["SOXL_raw_open", "SOXX_raw_open"]].to_numpy(float)
    output = {}
    for symbol in ("SOXL", "SOXX"):
        spec = fixed_buy_intent(symbol=symbol, decision_nav=intent["nav"],
            raw_close=float(intent["mark"][symbol + "_raw_close"]), cash=intent["cash"],
            fees=intent["fees"], decision_at=intent["slot"].decision_at)
        output[symbol] = (spec, adjusted, raw)
    return output


def scan_fixed_share_source(frame, book, quotes, perday_support, *, tariff):
    frame,path,support,by_day,fee_book,peaks,peak_error = _context(
        frame,book,quotes,perday_support,tariff)
    rows=[]
    for i,row in frame.iterrows():
        day=pd.Timestamp(row.trade_date);asset=by_day[day];source=support.iloc[i]
        q0,cash0,q1,cash1=_day_positions(book,day)
        counts={"intent_rows":0,"mature_path_rows":0,"paired_funded":0,
                "soxl_funded":0,"soxx_funded":0,
                "decision_no_whole_share":0,"decision_cash_insufficient":0,
                "insufficient_fill_cash":0}
        for intent in _day_intents(day,asset,q0,cash0,q1,cash1,peaks[i],fee_book[day]):
            counts["intent_rows"]+=1
            if not _path_ready(source,asset,intent):
                continue
            counts["mature_path_rows"]+=1
            arms=_arms(intent,asset)
            funded=[]
            for symbol in ("SOXL","SOXX"):
                spec,adjusted,raw=arms[symbol]
                result=settle_fixed_buy(intent["q"],intent["cash"],intent=spec,
                    adjusted_fill_prices=adjusted,raw_fill_prices=raw,
                    fill_at=intent["slot"].fill_at,fees=deepcopy(intent["fees"]))
                status=result["status"]
                counts[status]=counts.get(status,0)+int(status!="funded")
                yes=status=="funded" and len(result["orders"])==1 and result["orders"][0]["symbol"]==symbol
                funded.append(yes)
                counts[symbol.lower()+"_funded"]+=int(yes)
            counts["paired_funded"]+=int(all(funded))
        if (counts["intent_rows"]!=int(source.cash_intents) or
                counts["mature_path_rows"]!=int(source.mature_path_source_intents)):
            raise ValueError("blocked: fixed-share source clock/maturity differs")
        rows.append({"trade_date":day,"year":int(day.year),**counts})
    perday=pd.DataFrame(rows)
    annual=[]
    for year,part in perday.groupby("year"):
        annual.append({"year":int(year),"days":len(part),
            "paired_funded_days":int(part.paired_funded.gt(0).sum()),
            "paired_funded_rows":int(part.paired_funded.sum()),
            "mature_path_rows":int(part.mature_path_rows.sum()),
            "decision_no_whole_share":int(part.decision_no_whole_share.sum()),
            "decision_cash_insufficient":int(part.decision_cash_insufficient.sum()),
            "insufficient_fill_cash":int(part.insufficient_fill_cash.sum())})
    return perday,{"annual":annual,"opening_peak_max_error":peak_error,
                   "intent_rows":int(perday.intent_rows.sum()),
                   "mature_path_rows":int(perday.mature_path_rows.sum()),
                   "paired_funded_rows":int(perday.paired_funded.sum())}


def build_fixed_share_lock_labels(frame, book, quotes, perday_support,
                                  fixed_support, *, tariff,
                                  maximum_intent_rows=400000,
                                  maximum_seconds=900):
    frame,path,support,by_day,fee_book,peaks,peak_error = _context(
        frame,book,quotes,perday_support,tariff)
    fixed_support=fixed_support.reset_index(drop=True)
    if len(fixed_support)!=len(frame) or not frame.trade_date.equals(pd.to_datetime(fixed_support.trade_date)):
        raise ValueError("blocked: fixed-share source calendar differs")
    started=perf_counter();records=[]
    for i,row in frame.iterrows():
        day=pd.Timestamp(row.trade_date);asset=by_day[day];source=support.iloc[i]
        q0,cash0,q1,cash1=_day_positions(book,day)
        plane=(compile_lock_price_plane(frame.iloc[i:i+5],
            quotes.loc[quotes.index.normalize().isin(frame.trade_date.iloc[i:i+5])])
               if int(source.mature_path_source_intents)>0 else None)
        day_counts={"intent_rows":0,"mature_path_rows":0,"paired_funded":0}
        for intent in _day_intents(day,asset,q0,cash0,q1,cash1,peaks[i],fee_book[day]):
            day_counts["intent_rows"]+=1
            if len(records)>=maximum_intent_rows:
                raise ValueError("blocked: fixed-share row budget exceeded")
            slot=intent["slot"];mature=bool(source.five_session_mature)
            label_at=(pd.Timestamp(frame.iloc[i+4].exit_date)+pd.Timedelta(hours=9,minutes=49)) if mature else pd.NaT
            record={"trade_date":day,"bar_end_at":intent["stamp"],"decision_at":slot.decision_at,
                "fill_at":slot.fill_at,"label_available_at":label_at,
                "decision_nav":intent["nav"],"decision_cash_weight":intent["cash"]/intent["nav"],
                "decision_own_drawdown":max(0.,1-intent["nav"]/intent["peak"]),
                "parent_risk_state":int(path.iloc[i].risk_state),"phase":intent["phase"],
                "status":"unknown","hold_terminal_nav":np.nan,"hold_max_drawdown":np.nan,
                "soxl_order_status":"unknown","soxx_order_status":"unknown",
                "soxl_intended_raw_shares":0,"soxx_intended_raw_shares":0,
                "soxl_filled_raw_shares":0,"soxx_filled_raw_shares":0,
                "soxl_fee_usd":0.,"soxx_fee_usd":0.,
                "soxl_value":np.nan,"soxx_value":np.nan,"relative_value_soxl_minus_soxx":np.nan,
                "soxl_tail_damage":np.nan,"soxx_tail_damage":np.nan}
            if not mature:
                record["status"]="terminal_unknown";records.append(record);continue
            if not _path_ready(source,asset,intent) or plane is None:
                record["status"]="path_source_unknown";records.append(record);continue
            day_counts["mature_path_rows"]+=1
            arms=_arms(intent,asset)
            values={}
            for symbol in ("SOXL","SOXX"):
                spec,_,_=arms[symbol]
                record[symbol.lower()+"_intended_raw_shares"]=int(spec["raw_shares"])
                values[symbol]=value_lock_event(plane,bar_end_at=intent["stamp"],fill_at=slot.fill_at,
                    quantities=intent["q"],cash=intent["cash"],starting_peak=intent["initial_peak"],
                    fees=intent["fees"],fixed_intent=spec,start_at=intent["start_at"])
                record[symbol.lower()+"_order_status"]=values[symbol]["EVENT"]["status"]
            hold=values["SOXL"]["HOLD"];other=values["SOXX"]["HOLD"]
            if (abs(hold["terminal_nav"]-other["terminal_nav"])>1e-7 or
                    abs(hold["max_drawdown"]-other["max_drawdown"])>1e-10 or
                    any(abs(v["decision_nav"]-intent["nav"])>1e-7 for v in values.values())):
                raise ValueError("blocked: fixed-share same-state HOLD differs")
            record["hold_terminal_nav"]=hold["terminal_nav"]
            record["hold_max_drawdown"]=hold["max_drawdown"]
            funded=[]
            for symbol in ("SOXL","SOXX"):
                event=values[symbol]["EVENT"]
                yes=event["status"]=="funded" and len(event["orders"])==1 and event["orders"][0]["symbol"]==symbol
                funded.append(yes)
                if yes:
                    record[symbol.lower()+"_filled_raw_shares"]=int(event["orders"][0]["raw_shares"])
                    record[symbol.lower()+"_fee_usd"]=float(event["fee_usd"])
                # A no-share or cash-rejected order physically remains HOLD;
                # its local action value and risk damage are exactly zero.
                record[symbol.lower()+"_value"]=(event["terminal_nav"]-hold["terminal_nav"])/intent["nav"]
                record[symbol.lower()+"_tail_damage"]=event["max_drawdown"]-hold["max_drawdown"]
            record["relative_value_soxl_minus_soxx"]=record["soxl_value"]-record["soxx_value"]
            record["status"]="paired_funded" if all(funded) else "attempt_no_fill"
            day_counts["paired_funded"]+=int(all(funded))
            records.append(record)
        expected=fixed_support.iloc[i]
        if any(day_counts[key]!=int(expected[key]) for key in day_counts):
            raise ValueError("blocked: full fixed-share labels/source counts differ")
        if i%100==0:print("fixed_label_progress",i,len(records),round(perf_counter()-started,1),flush=True)
        if perf_counter()-started>maximum_seconds:
            raise ValueError("blocked: frozen fixed-share batch time budget exceeded")
    labels=pd.DataFrame(records)
    if len(labels)!=int(fixed_support.intent_rows.sum()):
        raise ValueError("blocked: fixed-share label row inventory differs")
    return labels,{"elapsed_seconds":perf_counter()-started,"opening_peak_max_error":peak_error,
                   "intent_rows":len(labels),
                   "mature_path_rows":int(labels.status.isin(["paired_funded","attempt_no_fill"]).sum()),
                   "paired_funded_rows":int(labels.status.eq("paired_funded").sum())}
