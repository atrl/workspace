"""Bounded post-model-seal Qlib prefix intake for the fixed P3 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.native_evidence import native_dataset
from src.data.path_area_bridge import PATH_COLUMNS, build_path_area_inputs


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def prefix_inputs_from_marks(marks, frame):
    """Use only each funded day's fifteen closed bars to define mask and area."""
    required = {"$open", "$close", "$price_valid"}
    _require(isinstance(marks.index, pd.MultiIndex)
             and list(marks.index.names) == ["datetime", "instrument"]
             and not marks.index.has_duplicates and required.issubset(marks),
             "raw prefix schema differs")
    dates, signals = pd.DatetimeIndex(frame.trade_date), pd.DatetimeIndex(frame.signal_date)
    _require(dates.tz is None and signals.tz is None and not dates.has_duplicates
             and not signals.has_duplicates and dates.is_monotonic_increasing
             and signals.is_monotonic_increasing and (signals < dates).all(),
             "funded prefix date mapping differs")
    stamps = pd.DatetimeIndex(marks.index.get_level_values("datetime"))
    minute = stamps.hour * 60 + stamps.minute
    retained = marks.loc[stamps.normalize().isin(dates) & (minute >= 571) & (minute <= 585)].copy()
    _require(set(retained.index.get_level_values("instrument")) <= {"SOXL", "SOXX"},
             "unexpected prefix instrument")
    masks = []
    for day in dates:
        grid = pd.MultiIndex.from_product(
            [pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=15, freq="min"),
             ["SOXL", "SOXX"]], names=["datetime", "instrument"])
        values = retained.reindex(grid)
        first = values.loc[(day + pd.Timedelta(hours=9, minutes=31), slice(None)), "$open"]
        mask = (values["$price_valid"].eq(1).all()
                and np.isfinite(values["$close"]).all() and values["$close"].gt(0).all()
                and np.isfinite(first).all() and first.gt(0).all())
        masks.append(int(mask))
    timing = pd.DataFrame({"trade_date": dates}, index=signals.rename("signal_date"))
    coverage = pd.DataFrame({"prefix_mask": masks}, index=timing.index)
    inputs = build_path_area_inputs(retained, timing, coverage)
    _require(pd.DatetimeIndex(inputs.trade_date).equals(dates), "prefix result row mapping differs")
    return inputs, retained


def read_evaluation_prefix(root, plan, frame, *, on_native_attempt=None):
    """One daily calendar and one bounded 3-field minute query, outside a writer.

    The caller must enforce the two-model seal before calling. The contiguous
    Qlib query temporarily decodes intervening non-prefix rows; the pure builder
    above discards those rows before missingness checks or path arithmetic.
    """
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.workflow.path_area_source_preflight import _init, bounded_raw_slices

    root = Path(root).resolve()
    try:
        active = R.exp_manager.active_experiment
    except AttributeError:
        active = None
    _require(active is None or active.active_recorder is None, "prefix intake requires no writable Recorder")
    source = plan["sources_and_phase_allowlists"]["P3b_evaluation_only_after_barrier"]
    prefix, daily = source["prefix_feature_source"], source["full_daily_calendar"]
    dates, signals = pd.DatetimeIndex(frame.trade_date), pd.DatetimeIndex(frame.signal_date)
    _require(len(frame) == 944 and dates[0] == pd.Timestamp("2023-01-04")
             and dates[-1] == pd.Timestamp("2026-10-08"), "fixed P3 funded prefix bounds differ")
    provider = root / prefix["provider_uri"]
    checks = {provider / name: digest for name, digest in prefix["current_provider_file_sha256"].items()}
    checks.update({
        provider / "qlib_store_manifest.json": prefix["manifest_sha256"],
        root / daily["calendar_path"]: daily["calendar_sha256"],
        root / daily["instrument_path"]: daily["instrument_sha256"],
    })
    for binding in (prefix["source_inventory"], prefix["source_byte_identity"]):
        checks[root / binding["path"]] = binding["sha256"]
    for path, digest in checks.items():
        _require(path.is_file() and not path.is_symlink() and _sha(path) == digest,
                 "prefix source bytes changed: " + str(path.relative_to(root)))
    source_manifest = json.loads((provider / "qlib_store_manifest.json").read_text())
    # The existing source manifest schema is preserved; its pinned digest is
    # authoritative even when snapshot IDs are carried by a separate inventory.
    _require(source_manifest.get("freq") == "1min", "raw prefix source frequency differs")
    _init(root / daily["provider_uri"])
    if on_native_attempt is not None:
        on_native_attempt("D.calendar")
    calendar = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    _require(calendar.tz is None and calendar.is_unique and calendar.is_monotonic_increasing,
             "native daily calendar differs")
    _require(dates.equals(calendar[(calendar >= dates[0]) & (calendar <= dates[-1])]),
             "funded dates omit native sessions")
    _init(provider)
    start = dates[0] + pd.Timedelta(hours=9, minutes=31)
    end = dates[-1] + pd.Timedelta(hours=9, minutes=45)
    fields = ["$open", "$close", "$price_valid"]
    with bounded_raw_slices(provider, freq="1min", fields=fields, instruments=["SOXL", "SOXX"],
                            start=start, end=end) as observed:
        if on_native_attempt is not None:
            on_native_attempt("D.features")
        raw = D.features(["SOXL", "SOXX"], fields, start_time=start, end_time=end,
                         freq="1min", disk_cache=0)
    _require(raw.index.names == ["instrument", "datetime"], "native Qlib minute index differs")
    raw = raw.reorder_levels(["datetime", "instrument"]).sort_index()
    decoded_rows = len(raw)
    inputs, retained = prefix_inputs_from_marks(raw, frame)
    del raw
    _require(len(observed["reads"]) == 6, "expected one slice per two-asset/three-field pair")
    for path, digest in checks.items():
        _require(_sha(path) == digest, "source bytes changed during prefix read")
    minute_frame = pd.concat({"feature": retained}, axis=1)
    prefix_dataset = native_dataset(minute_frame,
        segments={"all": (str(start), str(end))})
    path_metadata = inputs[["trade_date"]].copy()
    path_metadata["prefix_available_at"] = dates.tz_localize("America/New_York") + pd.Timedelta(
        hours=9, minutes=46)
    path_frame = pd.concat({
        "feature": inputs[[*PATH_COLUMNS, "prefix_mask"]],
        "meta": path_metadata,
    }, axis=1)
    path_frame.index = pd.MultiIndex.from_arrays([signals, ["SOXL_SOXX_CASH"] * len(signals)],
                                                names=["datetime", "instrument"])
    path_dataset = native_dataset(path_frame,
        segments={"all": (str(signals[0]), str(signals[-1]))})
    return {
        "path_inputs": inputs, "prefix_dataset": prefix_dataset,
        "path_dataset": path_dataset, "daily_calendar": calendar,
        "audit": {
            "D_calendar_calls": 1, "D_features_calls": 1, "numeric_field_slices": observed["reads"],
            "query_start": str(start), "query_end": str(end),
            "decoded_contiguous_rows": decoded_rows, "retained_prefix_rows": len(retained),
            "funded_sessions": len(inputs), "valid_prefix_sessions": int(inputs.prefix_mask.sum()),
            "missing_prefix_sessions": int(inputs.prefix_mask.eq(0).sum()),
            "intervening_rows_used_in_features_or_masks": False,
            "prefix_available_at_rule": "actual trade_date 09:46 America/New_York",
            "source_bytes_verified_before_and_after": True,
            "historical_PIT_qualified": False, "independent_validation_qualified": False,
            "source_byte_identity_sha256": prefix["source_byte_identity"]["sha256"],
        },
    }
