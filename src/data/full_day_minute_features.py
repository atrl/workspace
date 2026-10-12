"""Causal intraday snapshots built exclusively from native Qlib price planes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _groups(native: pd.DataFrame) -> dict:
    if native.attrs.get("qlib_data_api") != ["qlib.data.D.features"]:
        raise ValueError("blocked: native Qlib features required")
    frame = native.reset_index().rename(columns={c: c.removeprefix("$") for c in native.columns})
    frame["datetime"] = pd.to_datetime(frame.datetime)
    frame["date"] = frame.datetime.dt.normalize()
    frame["clock"] = frame.datetime.dt.hour * 60 + frame.datetime.dt.minute
    return dict(tuple(frame.sort_values("datetime").groupby(["instrument", "date"], sort=False)))


def _path_features(bars: pd.DataFrame, *, cutoff: int, prefix: bool) -> dict:
    part = bars.loc[bars.clock.le(min(cutoff, 630) if prefix else cutoff)]
    if part.empty or part.price_valid.ne(1).any() or part[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("unknown_source_path")
    opening, closing = float(part.open.iloc[0]), float(part.close.iloc[-1])
    volume = float(part.volume.sum())
    result = {"ret": closing / opening - 1, "range": float(part.high.max() / part.low.min() - 1),
              "from_low": closing / float(part.low.min()) - 1, "log_volume": np.log1p(volume)}
    if prefix:
        return result
    for minutes in (15, 30, 60):
        window = part.loc[part.clock.gt(cutoff - minutes)]
        if window.empty:
            raise ValueError("unknown_rolling_path")
        returns = np.log(window.close.astype(float) / window.close.shift(1).fillna(window.open.iloc[0]))
        result[f"ret_{minutes}"] = float(window.close.iloc[-1] / window.open.iloc[0] - 1)
        result[f"rv_{minutes}"] = float(np.sqrt(np.square(returns).sum()))
        if minutes == 30:
            rv = float(np.square(returns).sum())
            result["downside_share_30"] = float(np.square(returns[returns < 0]).sum() / rv) if rv else 0.0
            result["trend_efficiency_30"] = float(abs(returns.sum()) / returns.abs().sum()) if returns.abs().sum() else 0.0
    returns = np.log(part.close.astype(float) / part.close.shift(1).fillna(opening))
    result["prefix_rv"] = float(np.sqrt(np.square(returns).sum()))
    result["drawdown_from_high"] = closing / float(part.high.max()) - 1
    return result


def build_full_day_snapshots(
    core: pd.DataFrame, context: pd.DataFrame, execution: pd.DataFrame,
    parent: pd.DataFrame, calendar: pd.DatetimeIndex, *, clocks: list[int],
    execution_delay_minutes: int, context_symbols: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """Each snapshot ends at cutoff; fills are later native 15-minute closes.

    Open snapshots use only the preceding completed session. The final label
    matures at the next session's delayed open, explicitly including overnight.
    Historical bar availability is an assumption, not live receipt proof.
    """
    core_groups, context_groups, execution_groups = _groups(core), _groups(context), _groups(execution)
    calendar = pd.DatetimeIndex(calendar).normalize()
    parent = parent.copy()
    parent["signal_date"] = pd.to_datetime(parent.signal_date)
    if parent.signal_date.duplicated().any():
        raise ValueError("blocked: duplicate frozen parent predictions")
    parent_dates = calendar.searchsorted(parent.signal_date, side="right")
    known = (parent_dates < len(calendar)) & parent.signal_date.isin(calendar)
    parent = parent.loc[known].copy()
    parent["trade_date"] = calendar[parent_dates[known]]
    if (pd.to_datetime(parent.train_end) >= parent.signal_date).any():
        raise ValueError("blocked: frozen parent prediction chronology failed")
    parent = parent.set_index("trade_date")
    rows, eligibility = [], []
    base_names, rolling_names, context_names = set(), set(), set()
    for position, day in enumerate(calendar):
        if position == 0 or day not in parent.index:
            continue
        next_day = calendar[position + 1] if position + 1 < len(calendar) else None
        source_parent = parent.loc[day]
        for slot, cutoff in enumerate(clocks):
            source_day = calendar[position - 1] if cutoff == 570 else day
            source_cutoff = 960 if cutoff == 570 else cutoff
            info_at = day + pd.Timedelta(minutes=cutoff)
            entry_at = info_at + pd.Timedelta(minutes=execution_delay_minutes)
            exit_at = (day + pd.Timedelta(minutes=clocks[slot + 1] + execution_delay_minutes)
                       if slot + 1 < len(clocks) else
                       next_day + pd.Timedelta(minutes=clocks[0] + execution_delay_minutes) if next_day is not None else pd.NaT)
            row = {"datetime": info_at, "instrument": "ALLOCATION", "trade_date": day,
                   "entry_at": entry_at, "label_available_at": exit_at,
                   "source_day": source_day, "source_cutoff_at": source_day + pd.Timedelta(minutes=source_cutoff)}
            try:
                for symbol in ["SOXL", "SOXX"]:
                    bars = core_groups[(symbol, source_day)]
                    prefix = _path_features(bars, cutoff=source_cutoff, prefix=True)
                    rolling = _path_features(bars, cutoff=source_cutoff, prefix=False)
                    for name, value in prefix.items():
                        key = "prefix_" + symbol.lower() + "_" + name
                        row[key] = value
                        base_names.add(key)
                    for name, value in rolling.items():
                        key = "rolling_" + symbol.lower() + "_" + name
                        row[key] = value
                        rolling_names.add(key)
                    entry = execution_groups[(symbol, day)].set_index("datetime").loc[entry_at, "close"]
                    exit_day = exit_at.normalize() if pd.notna(exit_at) else day
                    exit_price = (execution_groups[(symbol, exit_day)].set_index("datetime").loc[exit_at, "close"]
                                  if pd.notna(exit_at) else np.nan)
                    row["label_" + symbol.lower()] = float(exit_price / entry - 1)
                for symbol in context_symbols:
                    values = _path_features(context_groups[(symbol, source_day)], cutoff=source_cutoff, prefix=False)
                    for name in ["ret_30", "ret_60", "rv_30", "downside_share_30"]:
                        key = "context_" + symbol.lower() + "_" + name
                        row[key] = values[name]
                        context_names.add(key)
                for name in ["soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"]:
                    row["parent_" + name] = float(source_parent[name])
                    base_names.add("parent_" + name)
                row["slot"] = slot
                row["overnight_label"] = int(slot == len(clocks) - 1)
                base_names.update(["slot", "overnight_label"])
                eligibility.append({"datetime": info_at, "eligible": True, "reason": "observed_native_paths"})
                rows.append(row)
            except (KeyError, ValueError) as error:
                eligibility.append({"datetime": info_at, "eligible": False, "reason": str(error)[:100]})
    frame = pd.DataFrame(rows).sort_values("datetime").set_index(["datetime", "instrument"])
    families = {"first_hour": sorted(base_names),
                "full_day": sorted(base_names | rolling_names),
                "full_day_context": sorted(base_names | rolling_names | context_names)}
    if frame.empty or not np.isfinite(frame[families["full_day_context"]].to_numpy(float)).all():
        raise ValueError("blocked: invalid full-day native feature snapshot")
    return frame, families, pd.DataFrame(eligibility)
