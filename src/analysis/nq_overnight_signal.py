"""NQ overnight 18:00->10:00 signal for post-10:31 equity returns.

Zero models. One declared decision (10:31 ET) and one direction-fixed signal read
from the NQ 60-minute qlib store (prior-calendar-day 18:00-bar open -> 09:00-bar
close, whose bar ends at 10:00, strictly observed before 10:31).

Stock labels come from the uniform Qlib 15-minute provider. The primary label is
``session_close`` chosen by the official Futu trade-date receipt (WHOLE -> 16:00,
MORNING -> 13:00); 13:00 is an auxiliary descriptive horizon. Next-open uses the
strictly-next trading day's 09:45 open (exit_at > decision_at).

Economics are long-only (SOXL/SOXX/cash) with exact two-leg settlement: each day
opens from cash and closes back to cash through ``fee_affordable_target`` and
``settle_target_session``. Uncertainty uses fixed 20-session moving blocks over
the FULL official calendar (invalid days kept as a mask, filtered only inside the
statistic). The original night-session ``+0.22`` result is void and not reused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.nq_futures_store import build_nq_futures_store

NQ_PROVIDER = "data/qlib/nq_futures_60min_20261010_v1"
NQ_RAW = "data/raw/external/night_session_20261010_v1/NQ_F.parquet"
RAW_MANIFEST = "data/raw/external/night_session_20261010_v1/manifest.json"
TRADING_DAYS = "data/raw/external/stage1_intraday_gap_repair_20261009_v1/trading_days.json"
BAR_SPAN_AUDIT = "data/raw/external/stage1_intraday_gap_repair_20261009_v1/source_manifest.json"
UNIFORM_INTRADAY = "data/qlib/stage1_uniform_intraday_20261009"
UNIFORM_EXECUTION = "data/qlib/stage1_uniform_execution_marks_20261009"
UNIFORM_DAILY = "data/qlib/stage1_uniform_daily_marks_20261009"
OVERNIGHT_START = "18:00"
ENTRY = "10:30"
DECISION_AT = "10:31"
HORIZON_FIXED = "13:00"
COSTS_BPS = (10.0, 25.0)
DIRECTION = 1.0
BLOCK_SESSIONS = 20
BOOTSTRAP = 2000
SYMBOLS = ("SOXL", "SOXX")
DEFENSIVE = "SOXX"
PRIMARY_BOOK = "rule_NQpos_SOXL_else_SOXX"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_sources(root: Path) -> dict:
    frozen = {}
    for name, rel in (("nq_raw", NQ_RAW), ("nq_raw_manifest", RAW_MANIFEST),
                      ("trading_days", TRADING_DAYS), ("bar_span_source_audit", BAR_SPAN_AUDIT)):
        path = root / rel
        _require(path.is_file(), f"missing source {rel}")
        frozen[name] = {"path": rel, "sha256": sha256(path)}
    for name, rel in (("nq_provider", NQ_PROVIDER), ("uniform_intraday", UNIFORM_INTRADAY),
                      ("uniform_execution", UNIFORM_EXECUTION), ("uniform_daily", UNIFORM_DAILY)):
        manifest = root / rel / "qlib_store_manifest.json"
        _require(manifest.is_file(), f"missing provider manifest {rel}")
        data = json.loads(manifest.read_text())
        frozen[name] = {"path": rel, "manifest_sha256": sha256(manifest), "freq": data.get("freq"),
                        "as_of": data.get("as_of"),
                        "trading_calendar_sha256": data.get("trading_calendar_sha256")}
    return frozen


def load_trading_days(root: Path) -> tuple[dict, str]:
    path = root / TRADING_DAYS
    _require(path.is_file(), "official trading-day receipt is missing")
    digest = sha256(path)
    manifest = json.loads((root / UNIFORM_INTRADAY / "qlib_store_manifest.json").read_text())
    _require(digest == manifest.get("trading_calendar_sha256"),
             "trading-day receipt digest does not match the uniform provider")
    data = json.loads(path.read_text())
    days = {pd.Timestamp(item["time"]).normalize(): item["trade_date_type"] for item in data["days"]}
    return days, digest


def read_nq(root: Path) -> pd.DataFrame:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str((root / NQ_PROVIDER).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="60min"))
    raw = D.features(["NQ"], ["$open", "$close"], freq="60min").reset_index()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw[raw["datetime"].isin(calendar)].copy()
    # Clock metadata derived from the qlib timeframe index, never stored as float.
    raw["bar_start"] = raw["datetime"]
    raw["bar_end"] = raw["datetime"] + pd.Timedelta(hours=1)
    raw["available_at"] = raw["bar_end"]
    return raw


def _bar(nq: pd.DataFrame, day: pd.Timestamp, hhmm: str):
    match = nq[(nq["datetime"].dt.normalize() == day) & (nq["datetime"].dt.strftime("%H:%M") == hhmm)]
    return None if len(match) != 1 else match.iloc[0]


def nq_overnight_signal(nq: pd.DataFrame, *, trading_days: dict) -> pd.DataFrame:
    """Per official day: prior CALENDAR day's 18:00-bar open -> day's 09:00-bar close.

    Monday uses Sunday's 18:00 bar (not Friday). Missing/invalid bars are audited,
    never substituted from the last observed date.
    """
    _require({"datetime", "bar_end", "$open", "$close"} <= set(nq.columns), "NQ store fields missing")
    rows = []
    for day in sorted(trading_days):
        prior = pd.Timestamp(day).normalize() - pd.Timedelta(days=1)
        evening = _bar(nq, prior, OVERNIGHT_START)
        morning = _bar(nq, pd.Timestamp(day).normalize(), "09:00")
        if evening is None or morning is None:
            continue
        open_price, close_price = float(evening["$open"]), float(morning["$close"])
        if not (np.isfinite(open_price) and np.isfinite(close_price) and open_price > 0 and close_price > 0):
            continue
        if pd.Timestamp(morning["bar_end"]) >= pd.Timestamp(day).normalize() + pd.Timedelta(hours=10, minutes=31):
            continue
        rows.append({"trade_date": pd.Timestamp(day).normalize(), "prior_day": prior,
                     "nq_overnight": close_price / open_price - 1.0,
                     "signal_available_at": pd.Timestamp(morning["bar_end"]),
                     "evening_bar_start": pd.Timestamp(evening["bar_start"]),
                     "morning_bar_end": pd.Timestamp(morning["bar_end"])})
    return pd.DataFrame(rows)


def _uniform_bars(root: Path, *, start: str, end: str) -> dict:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str((root / UNIFORM_INTRADAY).resolve()), region="us", kernels=1)
    raw = D.features(list(SYMBOLS), ["$open", "$close"],
                     start_time=start, end_time=f"{end} 16:00:00", freq="15min")
    return {"open": raw["$open"].unstack("instrument"), "close": raw["$close"].unstack("instrument"),
            "calendar": pd.DatetimeIndex(D.calendar(freq="15min"))}


def _stamp(frame: pd.DataFrame, day, hhmm: str, symbol: str):
    stamp = pd.Timestamp(day).normalize() + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    if stamp not in frame.index or symbol not in frame.columns:
        return None
    value = frame.loc[stamp, symbol]
    return float(value) if np.isfinite(value) else None


def build_calendar(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full official calendar in the NQ window with per-day validity and labels."""
    trading_days, digest = load_trading_days(root)
    nq = read_nq(root)
    signal = nq_overnight_signal(nq, trading_days=trading_days)
    signal_map = {row.trade_date: row for row in signal.itertuples()}
    nq_start = pd.Timestamp(nq["datetime"].min()).normalize()
    nq_end = pd.Timestamp(nq["datetime"].max()).normalize()
    official = [d for d in sorted(trading_days) if nq_start <= d <= nq_end]
    bars = _uniform_bars(root, start=str(official[0].date()), end=str(official[-1].date()))
    closes, opens, calendar = bars["close"], bars["open"], bars["calendar"]
    open_stamps = pd.DatetimeIndex(calendar[calendar.strftime("%H:%M") == "09:45"])
    stock_days = set(pd.DatetimeIndex(calendar).normalize())

    rows = []
    for day in official:
        trade_type = trading_days[day]
        session_close_time = HORIZON_FIXED if trade_type == "MORNING" else "16:00"
        requires_1600 = trade_type != "MORNING"
        record = {"trade_date": day, "trade_date_type": trade_type, "session_close_time": session_close_time,
                  "calendar_digest": digest, "valid": True, "reasons": ""}
        reasons = []
        if day not in signal_map:
            reasons.append("nq_window_incomplete")
        else:
            rec = signal_map[day]
            record.update({"nq_overnight": float(rec.nq_overnight), "signal_available_at": rec.signal_available_at,
                           "evening_bar_start": rec.evening_bar_start})
        if day not in stock_days:
            reasons.append("stock_session_not_in_calendar")
        if not reasons:
            for symbol in SYMBOLS:
                base = _stamp(closes, day, ENTRY, symbol)
                if base is None:
                    reasons.append(f"{symbol}:missing_entry_1030"); continue
                close_stamp = _stamp(closes, day, session_close_time, symbol)
                if close_stamp is None:
                    reasons.append(f"{symbol}:missing_session_close_{session_close_time}")
                record[f"{symbol.lower()}_session_close"] = (close_stamp / base - 1.0) if close_stamp is not None else np.nan
                aux = _stamp(closes, day, HORIZON_FIXED, symbol)
                record[f"{symbol.lower()}_close_1300"] = (aux / base - 1.0) if aux is not None else np.nan
                if requires_1600:
                    close16 = _stamp(closes, day, "16:00", symbol)
                    if close16 is None:
                        reasons.append(f"{symbol}:missing_close_1600")
                    record[f"{symbol.lower()}_close_1600"] = (close16 / base - 1.0) if close16 is not None else np.nan
                later = open_stamps[open_stamps.normalize() > day]
                if len(later) == 0 or later[0] not in opens.index or not np.isfinite(opens.loc[later[0], symbol]):
                    reasons.append(f"{symbol}:missing_next_open")
                else:
                    record[f"{symbol.lower()}_next_open"] = float(opens.loc[later[0], symbol]) / base - 1.0
                    record[f"{symbol.lower()}_next_open_at"] = later[0]
        record["valid"] = not reasons
        record["reasons"] = ";".join(reasons)
        rows.append(record)
    calendar_frame = pd.DataFrame(rows)
    valid = calendar_frame[calendar_frame["valid"]].reset_index(drop=True)
    if not valid.empty:
        for symbol in SYMBOLS:
            exit_at = pd.to_datetime(valid[f"{symbol.lower()}_next_open_at"])
            _require((exit_at > valid["trade_date"] + pd.Timedelta(hours=10, minutes=31)).all(),
                     "next-open exit not strictly after the 10:31 decision")
    return calendar_frame, valid


def _spearman(x, y) -> float:
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def block_ci_full(calendar: pd.DataFrame, statistic, *, block: int = BLOCK_SESSIONS) -> dict:
    """Fixed moving-block bootstrap over the FULL calendar; statistic filters validity."""
    n = len(calendar)
    if n < block or block < 2:
        return {"point": None, "ci_low": None, "ci_high": None, "blocks": 0, "status": "needs_data",
                "reason": "fewer calendar sessions than one block"}
    point = float(statistic(calendar))
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(20261010)
    draws = []
    for _ in range(BOOTSTRAP):
        chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in chosen])[:n]
        value = statistic(calendar.iloc[idx])
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    if not draws:
        return {"point": point, "ci_low": None, "ci_high": None, "blocks": int(len(starts)),
                "status": "needs_data", "reason": "no finite bootstrap draws"}
    return {"point": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "blocks": int(len(starts)), "status": "ok"}


def _roundtrip_net(weights: np.ndarray, r_soxl: float, r_soxx: float, cost: float) -> float:
    """Exact two-leg day: cash -> target at entry -> cash at session close."""
    return _roundtrip_detail(weights, r_soxl, r_soxx, cost)["net"]


def _roundtrip_detail(weights: np.ndarray, r_soxl: float, r_soxx: float, cost: float) -> dict:
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    pretrade = np.array([0.0, 0.0, 1.0])
    entry_target = fee_affordable_target(weights, pretrade_weights=pretrade, cost_bps=cost)
    opened = settle_target_session(pretrade, open_target_weights=entry_target,
                                   asset_returns=[r_soxl, r_soxx, 0.0], cost_bps=cost)
    exit_target = fee_affordable_target(np.array([0.0, 0.0, 1.0]), pretrade_weights=opened["ending_weights"], cost_bps=cost)
    closed = settle_target_session(opened["ending_weights"], open_target_weights=exit_target,
                                   asset_returns=[0.0, 0.0, 0.0], cost_bps=cost)
    return {"net": float(opened["net_equity"] * closed["net_equity"] - 1.0),
            "entry_fee": float(opened["fees"]), "exit_fee": float(closed["fees"]),
            "entry_weights": entry_target.tolist(), "ending_weights": opened["ending_weights"].tolist(),
            "closing_cash": float(closed["ending_weights"][2]), "closing_shares_weight": float(closed["ending_weights"][:2].sum())}


def _wealth_metrics(nets: np.ndarray) -> dict:
    values = np.asarray(nets, float)
    wealth = np.cumprod(1.0 + values)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    q05 = np.quantile(values, 0.05)
    return {"sessions": int(values.size), "cumulative_return_pct": 100.0 * float(wealth[-1] - 1.0),
            "max_drawdown_pct": 100.0 * float(drawdown.min()), "worst_day_pct": 100.0 * float(values.min()),
            "cvar5_pct": 100.0 * float(values[values <= q05].mean()), "mean_daily_bps": 10_000.0 * float(values.mean())}


def _soxx_buyhold_account(root: Path, first: pd.Timestamp, last: pd.Timestamp, costs_bps) -> dict:
    """Continuous SOXX open->next-open buy-and-hold from cash over the OFFICIAL stock
    calendar in [first, last] (not compressed to signal-available days). A missing
    daily mark is reported as needs_data, never silently skipped.
    """
    import qlib
    from qlib.data import D
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    trading_days, _ = load_trading_days(root)
    qlib.init(provider_uri=str((root / UNIFORM_DAILY).resolve()), region="us", kernels=1)
    opens = D.features(["SOXX"], ["$open"], start_time=str(pd.Timestamp(first).date()),
                       end_time=str(pd.Timestamp(last).date()), freq="day")["$open"].unstack("instrument")["SOXX"].astype(float)
    opens.index = pd.DatetimeIndex(opens.index).normalize()
    stock_calendar = set(opens.index)
    sessions = [pd.Timestamp(d) for d in sorted(trading_days) if pd.Timestamp(first) <= d <= pd.Timestamp(last) and d in stock_calendar]
    missing = [str(d.date()) for d in sessions if d not in opens.index or not np.isfinite(opens.loc[d])]
    accounts = {}
    for cost in costs_bps:
        wealth, previous, path = 1.0, np.array([0.0, 0.0, 1.0]), []
        for position in range(len(sessions) - 1):
            day, nxt = sessions[position], sessions[position + 1]
            if day not in opens.index or nxt not in opens.index or not (np.isfinite(opens.loc[day]) and np.isfinite(opens.loc[nxt])):
                continue
            soxx_ret = float(opens.loc[nxt]) / float(opens.loc[day]) - 1.0
            target = (fee_affordable_target(np.array([0.0, 1.0, 0.0]), pretrade_weights=previous, cost_bps=cost)
                      if position == 0 else None)
            settled = settle_target_session(previous, open_target_weights=target,
                                            asset_returns=[0.0, soxx_ret, 0.0], cost_bps=cost)
            previous = settled["ending_weights"]
            wealth *= settled["net_equity"]
            path.append({"trade_date": day, "exit_date": nxt, "soxx_return": soxx_ret, "net_equity": wealth})
        accounts[int(cost)] = {"return_pct": 100.0 * (wealth - 1.0), "sessions": len(path),
                               "status": "needs_data" if missing else "ok", "missing_marks": missing,
                               "path": pd.DataFrame(path)}
    return accounts


def _valid(calendar: pd.DataFrame) -> pd.DataFrame:
    return calendar[calendar["valid"]].reset_index(drop=True)


def evaluate(root: Path) -> dict:
    calendar, valid = build_calendar(root)
    _require(not valid.empty, "no valid sessions for the NQ 10:31 signal")
    results = {
        "signal": "nq_overnight 18:00->10:00 (prior calendar day)", "declared_decision_at": DECISION_AT, "entry": ENTRY,
        "direction": DIRECTION, "models_trained": 0, "block_sessions": BLOCK_SESSIONS,
        "official_days_in_range": int(len(calendar)), "sessions": int(len(valid)),
        "dropped_needs_data": int((~calendar["valid"]).sum()),
        "morning_days_in_range": int((calendar["trade_date_type"] == "MORNING").sum()),
        "valid_morning_days": int((valid["trade_date_type"] == "MORNING").sum()),
        "missing_nq_signal_days": int(calendar["reasons"].str.contains("nq_window_incomplete").sum()),
        "calendar_digest": calendar["calendar_digest"].iloc[0] if len(calendar) else None,
    }

    spread_stat = lambda d: _spearman(_valid(d)["nq_overnight"].to_numpy(float),
                                      (_valid(d)["soxl_session_close"] - _valid(d)["soxx_session_close"]).to_numpy(float))
    results["spread_ic_primary"] = block_ci_full(calendar, spread_stat)
    results["individual_ic_descriptive"] = {
        "soxl_session_close": block_ci_full(calendar, lambda d: _spearman(_valid(d)["nq_overnight"].to_numpy(float), _valid(d)["soxl_session_close"].to_numpy(float))),
        "soxx_session_close": block_ci_full(calendar, lambda d: _spearman(_valid(d)["nq_overnight"].to_numpy(float), _valid(d)["soxx_session_close"].to_numpy(float))),
        "soxl_close_1300": block_ci_full(calendar, lambda d: _spearman(_valid(d)["nq_overnight"].to_numpy(float), _valid(d)["soxl_close_1300"].to_numpy(float))),
        "soxx_close_1300": block_ci_full(calendar, lambda d: _spearman(_valid(d)["nq_overnight"].to_numpy(float), _valid(d)["soxx_close_1300"].to_numpy(float))),
    }

    economic = []
    for label, chooser in (("rule_NQpos_SOXL_else_SOXX", "rule"), ("soxl_only", "soxl"), ("soxx_only", "soxx")):
        for cost in COSTS_BPS:
            column = f"net_{label}_{int(cost)}"
            nets = np.full(len(calendar), np.nan)
            for index, row in calendar.iterrows():
                if not row["valid"]:
                    continue
                if chooser == "rule":
                    weights = np.array([1.0, 0.0, 0.0]) if row["nq_overnight"] > 0 else np.array([0.0, 1.0, 0.0])
                elif chooser == "soxl":
                    weights = np.array([1.0, 0.0, 0.0])
                else:
                    weights = np.array([0.0, 1.0, 0.0])
                nets[index] = _roundtrip_net(weights, row["soxl_session_close"], row["soxx_session_close"], cost)
            calendar[column] = nets
            ci = block_ci_full(calendar, lambda d, c=column: float(np.mean(_valid(d)[c].to_numpy(float)))
                               if not _valid(d).empty else None)
            economic.append({"book": label, "cost_bps": cost,
                             "mean_net_bps": None if ci["point"] is None else 10_000.0 * ci["point"],
                             "ci_low_bps": None if ci["ci_low"] is None else 10_000.0 * ci["ci_low"],
                             "ci_high_bps": None if ci["ci_high"] is None else 10_000.0 * ci["ci_high"],
                             "ci_status": ci["status"], "legs": "cash->target at 10:30, target->cash at session close"})
    results["economic_long_only"] = economic

    relative, wealth_table, ledgers = [], [], {}
    for cost in COSTS_BPS:
        rule_col, soxx_col = f"net_{PRIMARY_BOOK}_{int(cost)}", f"net_soxx_only_{int(cost)}"
        diff_col = f"rel_rule_minus_soxx_{int(cost)}"
        calendar[diff_col] = calendar[rule_col] - calendar[soxx_col]
        ci = block_ci_full(calendar, lambda d, c=diff_col: float(np.mean(_valid(d)[c].to_numpy(float))) if not _valid(d).empty else None)
        relative.append({"cost_bps": cost,
                         "mean_diff_bps": None if ci["point"] is None else 10_000.0 * ci["point"],
                         "ci_low_bps": None if ci["ci_low"] is None else 10_000.0 * ci["ci_low"],
                         "ci_high_bps": None if ci["ci_high"] is None else 10_000.0 * ci["ci_high"],
                         "ci_status": ci["status"],
                         "note": "descriptive only; a positive relative CI does not mask absolute losses"})
        for label, column in ((PRIMARY_BOOK, rule_col), ("soxl_only", f"net_soxl_only_{int(cost)}"),
                              ("soxx_only", soxx_col)):
            wealth_table.append({"book": label, "cost_bps": cost,
                                 **_wealth_metrics(_valid(calendar)[column].to_numpy(float))})
        ledger_rows = []
        for _, row in valid.iterrows():
            weights = np.array([1.0, 0.0, 0.0]) if row["nq_overnight"] > 0 else np.array([0.0, 1.0, 0.0])
            detail = _roundtrip_detail(weights, row["soxl_session_close"], row["soxx_session_close"], cost)
            ledger_rows.append({"trade_date": row["trade_date"], "asset": "SOXL" if weights[0] else "SOXX",
                                "nq_overnight": row["nq_overnight"], "entry_fee": detail["entry_fee"],
                                "exit_fee": detail["exit_fee"], "closing_cash": detail["closing_cash"],
                                "closing_shares_weight": detail["closing_shares_weight"], "net": detail["net"]})
        ledgers[int(cost)] = pd.DataFrame(ledger_rows)
    results["relative_rule_minus_soxx"] = relative
    results["three_account_wealth"] = wealth_table
    results["native_capability"] = {
        "full_strategy_native": "needs_data",
        "reason": "root helper covers opening/recovery slots; the 10:30 entry->session_close two-leg roundtrip is not directly adaptable",
        "recorded": "real settle_target_session fee/cash/weight paths per day (normalized account)"}
    results["execution_proxy"] = ("entry mark is the predeclared historical 10:30 close; decision is 10:31 ET; "
                                  "NOT verified real 10:31 fills")
    results["direction_note"] = "the NQ sign is a directional description; economics are long-only, no short portfolio"
    accounts = _soxx_buyhold_account(root, valid["trade_date"].min(),
                                     pd.to_datetime(valid["soxl_next_open_at"]).max(), COSTS_BPS)
    results["soxx_buyhold_account"] = {k: {"return_pct": v["return_pct"], "sessions": v["sessions"],
                                           "status": v["status"], "missing_marks": v["missing_marks"]}
                                       for k, v in accounts.items()}
    return {"results": results, "calendar": calendar, "panel": valid, "buyhold": accounts, "ledgers": ledgers}

def ensure_nq_store(root: Path) -> dict:
    manifest = root / NQ_PROVIDER / "qlib_store_manifest.json"
    if not manifest.is_file():
        return build_nq_futures_store(root=root)
    return {"provider_uri": str((root / NQ_PROVIDER).resolve()), "reused": True}
