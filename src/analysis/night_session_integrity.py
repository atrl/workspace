"""Machine-verifiable integrity correction for the night-session signal card.

The original card reported a headline plus0.22 IC for an overnight NQ futures
signal. Independent review found the signal is not causal:

1. The overnight window selected bars with ``index.time < 09:30``. The Yahoo NQ
   one-hour bar labeled ``09:00`` actually spans ``09:00-10:00`` (its close is
   the 10:00 price), so the window consumed a price from *after* the cash open.
   Removing that bar drops the IC from +0.2200 to -0.0119 (excess: +0.2244 to
   -0.0072).
2. The headline return is open-to-close, but the native reconciliation uses
   open-to-next-open; these are different definitions.
3. Turnover is charged as adjacent-day target differences, so an intraday round
   trip is undercharged.
4. The benchmark labeled "SOXX buyhold" is an intraday session return.
5. The diagnostic re-selected the signal threshold inside each evaluation window.

This module recomputes the IC under a safe cutoff and records the known defects.
It does not rewrite the original metrics; it marks the card
``invalid_for_promotion`` and preserves the original numbers as invalid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

DATA_DIR = "data/raw/external/night_session_20261010_v1"
OVERNIGHT_START = "18:00"
SAFE_CUTOFF = "09:00"
UNSAFE_CUTOFF = "09:30"
KNOWN_DEFECTS = (
    "look_ahead: 09:00 one-hour bar spans 09:00-10:00 but was selected by index.time < 09:30",
    "return_convention_mismatch: headline open-to-close vs native open-to-next-open",
    "turnover_undercharged: adjacent-day target difference, not intraday round trip",
    "benchmark_mislabeled: 'SOXX buyhold' is an intraday session return",
    "threshold_reselected_in_diagnostic: threshold chosen inside the evaluation window",
    "recorder_incomplete: recorder stored only a summary yet claimed panels",
)


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class IntegritySpec:
    root: object = "."
    cutoff: str = UNSAFE_CUTOFF


# Bar labels from the source are start-hour labels. A bar is only available at
# its *end*, so a bar labeled H is available at H + the bar width.
BAR_WIDTH = {"1h": pd.Timedelta(hours=1), "15m": pd.Timedelta(minutes=15),
             "5m": pd.Timedelta(minutes=5), "1m": pd.Timedelta(minutes=1)}


def bar_available_at(label: pd.Timestamp, resolution: str) -> pd.Timestamp:
    """Return the earliest time a start-labeled bar is actually available.

    A ``09:00`` one-hour bar covers ``09:00-10:00`` and its close is only known at
    ``10:00``. Using it at ``09:30`` is look-ahead. This helper makes the
    availability explicit so a future perturbation test can assert that a bar is
    not consumed before its end.
    """
    _require(resolution in BAR_WIDTH, f"unknown bar resolution: {resolution}")
    stamp = pd.Timestamp(label)
    return stamp + BAR_WIDTH[resolution]


def selectable_at(label: pd.Timestamp, decision_at: pd.Timestamp, resolution: str) -> bool:
    """Whether a start-labeled bar may be used at ``decision_at`` without look-ahead."""
    return bar_available_at(label, resolution) <= pd.Timestamp(decision_at)


def _regular(frame: pd.DataFrame) -> pd.DataFrame:
    mask = ((frame.index.time >= pd.Timestamp("09:30").time())
            & (frame.index.time < pd.Timestamp("16:00").time()))
    block = frame.loc[mask].assign(day=lambda x: x.index.normalize())
    grouped = block.groupby("day")
    return pd.DataFrame({"open": grouped["open"].first(), "close": grouped["close"].last()})


def _futures_overnight(frame: pd.DataFrame, cutoff: str) -> pd.Series:
    work = frame.assign(day=frame.index.normalize()).sort_index()
    start = pd.Timestamp(OVERNIGHT_START).time()
    cut = pd.Timestamp(cutoff).time()
    days = sorted(work["day"].unique())
    rows = {}
    for index, day in enumerate(days):
        if index == 0:
            continue
        previous = days[index - 1]
        window = work.loc[((work["day"] == previous) & (work.index.time >= start))
                          | ((work["day"] == day) & (work.index.time < cut))]
        if len(window) >= 2:
            rows[day] = window["close"].iloc[-1] / window["open"].iloc[0] - 1.0
    return pd.Series(rows).sort_index()


def load_night_session(root) -> dict[str, pd.DataFrame]:
    from pathlib import Path

    base = Path(root) / DATA_DIR
    out = {}
    for key, name in (("soxl", "SOXL"), ("soxx", "SOXX"), ("nq", "NQ_F")):
        path = base / f"{name}.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
            frame.index = pd.to_datetime(frame.index)
            out[key] = frame
    return out


SNAPSHOT = "research/night_session_signal_20261010/night_span_source_audit.json"
AGG_RULES = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
REQUIRED_DAYS = ("2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08")
REQUIRED_BARS_PER_DAY = 12


def _aggregate_five_minute(bars: list[dict]) -> dict:
    frame = pd.DataFrame(bars)
    return {
        "open": float(frame["open"].iloc[0]),
        "high": float(frame["high"].max()),
        "low": float(frame["low"].min()),
        "close": float(frame["close"].iloc[-1]),
        "volume": float(frame["volume"].sum()),
    }


def _expected_bar_stamps(day: str) -> list[str]:
    # The aggregation window starts at 09:00 ET on the given day.
    start = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9)
    return [
        (start + pd.Timedelta(minutes=5 * i)).isoformat()
        for i in range(REQUIRED_BARS_PER_DAY)
    ]


def _validate_bars(bars: list[dict], day: str) -> str | None:
    """Return a reason string when the five-minute bars are not exact, else None."""
    if not isinstance(bars, list) or len(bars) != REQUIRED_BARS_PER_DAY:
        return f"expected exactly {REQUIRED_BARS_PER_DAY} bars, got {len(bars) if isinstance(bars, list) else 0}"
    stamps = [str(b.get("timestamp_et")) for b in bars]
    expected = _expected_bar_stamps(day)
    if stamps != expected:
        return "timestamp_et sequence is not exactly 09:00..09:55 at 5-minute steps without gaps or duplicates"
    for bar in bars:
        for field in AGG_RULES:
            value = bar.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                return f"bar field {field} is missing or non-finite"
    return None


def bar_span_audit(root, *, snapshot: str | None = None,
                   required_days: tuple[str, ...] = REQUIRED_DAYS) -> dict:
    """Re-aggregate the archived five-minute bars offline and compare five fields.

    Consumes the archived source-timestamp snapshot, re-derives the hourly OHLCV
    from the stored five-minute bars using the declared rules, and requires the
    hourly bar to match on all five fields for every required day. The snapshot's
    own ``all_ohlcv_equal`` boolean is ignored. Fewer than all required days or
    fewer than 12 bars on any day is ``needs_data`` rather than a pass.
    """
    from pathlib import Path

    path = Path(root) / (snapshot or SNAPSHOT)
    if not path.is_file():
        return {"status": "needs_data", "reason": "source span snapshot is missing"}
    doc = json.loads(path.read_text(encoding="utf-8"))
    # Bind the snapshot to the raw parquet by re-hashing the file, not by trusting
    # the stored string. A missing file or a hash mismatch is needs_data.
    raw = doc.get("local_raw_parquet") or {}
    raw_rel = raw.get("path")
    raw_sha = raw.get("sha256")
    raw_path = Path(root) / raw_rel if isinstance(raw_rel, str) else None
    if raw_path is None or not raw_path.is_file():
        return {"status": "needs_data", "reason": "bound raw parquet is missing",
                "bound_raw_parquet": raw_sha, "days": []}
    actual_raw_sha = _sha256(raw_path)
    if actual_raw_sha != raw_sha:
        return {"status": "needs_data", "reason": "bound raw parquet SHA-256 mismatch",
                "bound_raw_parquet": raw_sha, "actual_raw_parquet_sha256": actual_raw_sha,
                "days": []}
    days = doc.get("days") if isinstance(doc.get("days"), list) else list((doc.get("days") or {}).values())
    by_day = {entry.get("day"): entry for entry in days if isinstance(entry, dict)}
    missing = [d for d in required_days if d not in by_day]
    if missing:
        return {"status": "needs_data", "reason": f"required days missing: {missing}",
                "bound_raw_parquet": raw_sha, "days": []}
    rows = []
    for day in required_days:
        entry = by_day[day]
        bars = entry.get("five_minute_bars")
        hourly = entry.get("hourly_bar")
        reason = _validate_bars(bars, day)
        if reason is not None or not isinstance(hourly, dict):
            rows.append({"day": day, "status": "needs_data",
                         "reason": reason or "hourly bar is missing"})
            continue
        agg = _aggregate_five_minute(bars)
        fields = {k: bool(abs(float(hourly[k]) - agg[k]) < 1e-6) for k in AGG_RULES}
        rows.append({"day": day, "status": "ok", "hourly": {k: float(hourly[k]) for k in AGG_RULES},
                     "reaggregated": agg, "fields": fields,
                     "identical": all(fields.values())})
    ok_rows = [r for r in rows if r["status"] == "ok"]
    complete = len(ok_rows) == len(required_days)
    identical = bool(complete and all(r["identical"] for r in ok_rows))
    if not complete:
        status = "needs_data"
    elif not identical:
        status = "mismatch"
    else:
        status = "ok"
    return {
        "status": status,
        "all_five_fields_identical": identical,
        "required_days": list(required_days),
        "required_bars_per_day": REQUIRED_BARS_PER_DAY,
        "snapshot": str(path.relative_to(root)) if path.is_relative_to(Path(root)) else str(path),
        "snapshot_sha256": _sha256(path),
        "bound_raw_parquet": raw_sha,
        "verified_raw_parquet_sha256": actual_raw_sha,
        "days": rows,
        "scope": "forensic source-timestamp audit only; not research or strategy evidence",
    }


def _sha256(path) -> str:
    import hashlib
    from pathlib import Path as _P
    return hashlib.sha256(_P(path).read_bytes()).hexdigest()


def recompute_ic(root, *, cutoff: str) -> dict:
    frames = load_night_session(root)
    if not {"soxl", "soxx", "nq"}.issubset(frames):
        return {"status": "needs_data", "cutoff": cutoff}
    soxl, soxx = _regular(frames["soxl"]), _regular(frames["soxx"])
    returns = pd.DataFrame({"soxl": soxl["close"] / soxl["open"] - 1.0,
                            "soxx": soxx["close"] / soxx["open"] - 1.0})
    signal = _futures_overnight(frames["nq"], cutoff).rename("nq")
    panel = signal.to_frame().join(returns, how="inner").dropna()
    panel = panel[panel.index.dayofweek < 5]
    return {
        "status": "ok",
        "cutoff": cutoff,
        "sessions": int(len(panel)),
        "ic_soxl": float(panel["nq"].corr(panel["soxl"])),
        "ic_excess": float(panel["nq"].corr(panel["soxl"] - panel["soxx"])),
    }


def integrity_correction(root) -> dict:
    """Assemble the machine-verifiable correction record."""
    unsafe = recompute_ic(root, cutoff=UNSAFE_CUTOFF)
    safe = recompute_ic(root, cutoff=SAFE_CUTOFF)
    span = bar_span_audit(root)
    return {
        "card": "night_session_signal_20261010",
        "status": "invalid_for_promotion",
        "objective_achieved": False,
        "promotion_eligible": False,
        "known_defects": list(KNOWN_DEFECTS),
        "bar_span_audit": span,
        "span_source_snapshot": span.get("snapshot"),
        "span_source_snapshot_sha256": span.get("snapshot_sha256"),
        "ic_unsafe_cutoff_09_30": unsafe,
        "ic_safe_cutoff_09_00": safe,
        "interpretation": (
            "The +0.22 IC is produced by the 09:00 one-hour bar whose span ends at 10:00, "
            "after the 09:30 cash open. Under a safe 09:00 cutoff the IC collapses to near zero, "
            "so the signal is not causal. The original metrics are preserved as invalid."),
    }
