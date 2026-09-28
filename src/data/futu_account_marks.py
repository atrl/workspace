"""Unadjusted Qlib marks for actual broker share quantities."""
from pathlib import Path

import pandas as pd

from src.data.qlib_native import build_qlib_store_from_frame


def refresh_account_marks(*, root: Path, start: str, end: str,
                          host: str = "127.0.0.1", port: int = 11111) -> dict:
    from futu import OpenQuoteContext, RET_OK, AuType, KLType, SysConfig

    SysConfig.set_all_thread_daemon(True)
    frames = []
    context = OpenQuoteContext(host=host, port=port)
    try:
        for symbol in ("SOXL", "SOXX"):
            page = None
            while True:
                ret, frame, page = context.request_history_kline(
                    "US." + symbol, start=start, end=end, ktype=KLType.K_DAY,
                    autype=AuType.NONE, max_count=1000, page_req_key=page,
                )
                if ret != RET_OK:
                    raise RuntimeError(f"blocked: unadjusted account marks unavailable for {symbol}: {frame}")
                frame["instrument"] = symbol
                frame["datetime"] = pd.to_datetime(frame["time_key"])
                frames.append(frame)
                if page is None:
                    break
    finally:
        context.close()
    prices = pd.concat(frames, ignore_index=True)
    calendars = prices.groupby("instrument")["datetime"].apply(lambda s: tuple(sorted(s.unique())))
    if len(calendars) != 2 or calendars.iloc[0] != calendars.iloc[1]:
        raise ValueError("blocked: account mark calendars do not match")
    if str(prices["datetime"].max().date()) != end:
        raise ValueError("blocked: unadjusted account marks are stale")
    return build_qlib_store_from_frame(
        prices, provider_uri=root / "data/qlib/stage1_account_marks", market="all", region="us",
        field_map={"$open": "open", "$close": "close"}, as_of=end,
        source="Futu request_history_kline K_DAY AuType.NONE; actual broker shares",
    )
