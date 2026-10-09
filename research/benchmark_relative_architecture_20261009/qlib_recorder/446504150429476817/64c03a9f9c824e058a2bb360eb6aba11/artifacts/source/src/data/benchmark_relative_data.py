"""Qlib-native open-to-open rows with explicit information availability."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_relative_dataset(*, provider_uri, signal_start, as_of):
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    qlib.init(provider_uri=str(provider_uri), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    if calendar.empty or str(calendar[-1].date()) != as_of or pd.Timestamp(signal_start) not in calendar:
        raise ValueError("blocked: source calendar does not match declared start/as_of")
    instruments = sorted(D.list_instruments(D.instruments("all"), as_list=True))
    if instruments != ["SOXL", "SOXX"]:
        raise ValueError("blocked: benchmark-relative universe differs")
    raw = D.features(instruments, ["$open"], start_time=calendar[0], end_time=as_of, freq="day")
    prices = raw["$open"].unstack("instrument").reindex(calendar)[instruments].astype(float)
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("blocked: complete positive native open prices required")
    signal = calendar[:-2]
    frame = pd.DataFrame(index=signal)
    frame.index.name = "datetime"
    frame["anchor_feature"] = 1.
    for symbol in instruments:
        frame[f"{symbol.lower()}_return"] = prices[symbol].to_numpy()[2:] / prices[symbol].to_numpy()[1:-1] - 1.
    frame["signal_date"], frame["trade_date"], frame["exit_date"] = signal, calendar[1:-1], calendar[2:]
    frame["decision_at"] = signal.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    frame["return_available_at"] = calendar[2:].tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    frame = frame.loc[pd.Timestamp(signal_start):].copy()
    if frame.empty:
        raise ValueError("blocked: no complete open-to-open sample after signal_start")
    columns = {"feature": ["anchor_feature"], "label": ["soxl_return", "soxx_return"],
        "metadata": ["signal_date", "trade_date", "exit_date", "decision_at", "return_available_at"]}
    native = pd.concat({key: frame[value] for key, value in columns.items()}, axis=1)
    native.index = pd.MultiIndex.from_arrays([frame.index, ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP(instruments=None, start_time=frame.index[0], end_time=frame.index[-1],
        data_loader=StaticDataLoader(config=native), infer_processors=[], learn_processors=[]),
        segments={"all": (str(frame.index[0].date()), str(frame.index[-1].date()))})
    readback = dataset.prepare("all", col_set=["feature", "label", "metadata"]).copy()
    readback.columns = readback.columns.get_level_values(1)
    readback.index = pd.DatetimeIndex(readback.index.get_level_values("datetime"))
    pd.testing.assert_frame_equal(readback, frame, check_exact=True, check_freq=False)
    return dataset, readback, calendar
