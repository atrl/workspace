"""Point-in-time QQQ/SMH downside-resilience join for Stage1 minute intents.

The controls include the old three-peer 09:45 breadth, current QQQ 5m return,
and unconditional SMH-minus-QQQ 5m gap. Only their QQQ-down interaction is
candidate-only. Economic targets are attached later by the frozen Qlib screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_peer_downshock(panel: pd.DataFrame, peer_bars: pd.DataFrame):
    if not isinstance(peer_bars.index, pd.MultiIndex):
        raise ValueError("blocked: Qlib peer 5m instrument/time index required")
    parts = {}
    for symbol in ("QQQ", "SMH", "NVDA", "AMD"):
        part = peer_bars.xs(symbol, level="instrument").copy()
        if not part.index.is_unique:
            raise ValueError("blocked: duplicate peer minute bar")
        parts[symbol] = part
    if not all(parts["QQQ"].index.equals(parts[s].index) for s in parts):
        raise ValueError("blocked: peer bar grids differ")
    q, s = parts["QQQ"], parts["SMH"]
    valid = q["$price_valid"].eq(1) & s["$price_valid"].eq(1)
    peer = pd.DataFrame(index=q.index)
    peer["qqq_current_5m_return"] = np.log(q["$close"] / q["$open"])
    smh_return = np.log(s["$close"] / s["$open"])
    peer["smh_qqq_current_5m_gap"] = smh_return - peer["qqq_current_5m_return"]
    peer["peer_downshock_resilience"] = np.where(
        peer.qqq_current_5m_return < 0, peer.smh_qqq_current_5m_gap, 0.)
    peer = peer.loc[valid].copy()
    peer["trade_date"] = peer.index.normalize()
    old_breadth = []
    for day, _ in peer.groupby("trade_date"):
        signs = []
        for symbol in ("AMD", "NVDA", "SMH"):
            part = parts[symbol]
            stamps = [day + pd.Timedelta(hours=9, minutes=m) for m in (35, 40, 45)]
            if (not all(stamp in part.index for stamp in stamps) or
                    not part.loc[stamps, "$price_valid"].eq(1).all()):
                raise ValueError("blocked: old three-peer 09:45 source absent")
            first = float(part.loc[stamps[0], "$open"])
            last = float(part.loc[stamps[-1], "$close"])
            if first <= 0 or last <= 0:
                raise ValueError("blocked: old breadth peer price invalid")
            signs.append(np.log(last / first) > 0)
        old_breadth.append((day, float(np.mean(signs))))
    breadth = dict(old_breadth)
    source = peer.reset_index().rename(columns={peer.index.name or "datetime":"peer_bar_end_at"})
    if "peer_bar_end_at" not in source:
        source = source.rename(columns={source.columns[0]:"peer_bar_end_at"})
    source = source.sort_values("peer_bar_end_at")
    ordered = panel.sort_values("bar_end_at").copy()
    joined = pd.merge_asof(ordered, source,
        left_on="bar_end_at", right_on="peer_bar_end_at",
        by="trade_date", direction="backward", allow_exact_matches=True)
    joined["peer_breadth_0945"] = joined.trade_date.map(breadth)
    if joined[["peer_bar_end_at", "peer_breadth_0945",
               "qqq_current_5m_return", "smh_qqq_current_5m_gap",
               "peer_downshock_resilience"]].isna().any().any():
        raise ValueError("blocked: peer source missing at minute decision")
    age = (joined.bar_end_at - joined.peer_bar_end_at).dt.total_seconds() / 60.
    if (age.lt(0).any() or age.gt(4).any() or
            (joined.peer_bar_end_at + pd.Timedelta(minutes=1)).gt(joined.decision_at).any()):
        raise ValueError("blocked: peer 5m completion is late")
    joined = joined.sort_values(["trade_date", "bar_end_at"]).reset_index(drop=True)
    annual=[]
    for year, part in joined.groupby(joined.trade_date.dt.year):
        annual.append({"year":int(year), "rows":len(part),
                       "days":int(part.trade_date.nunique()),
                       "max_peer_age_minutes":float(((part.bar_end_at - part.peer_bar_end_at)
                           .dt.total_seconds() / 60.).max())})
    return joined, {"rows":len(joined), "maximum_peer_age_minutes":float(age.max()),
                    "annual":annual, "historical_first_receipt_qualified":False}
