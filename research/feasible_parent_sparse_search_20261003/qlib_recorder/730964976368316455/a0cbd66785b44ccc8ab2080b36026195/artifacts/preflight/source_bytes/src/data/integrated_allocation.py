"""Frozen Qlib inputs for integrated allocation, auxiliary learning and OOF heads.

Only Qlib providers supply observations. The reused constituent feature builder
provides the common causal encoder inputs; its synthetic/classification labels
are deliberately excluded. Label availability follows each target's last
required observed price, rather than a guessed number of purge rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.factor import build_constituent_2x_tail_training_frame
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES


PRICE_FIELDS = ("$open", "$high", "$low", "$close", "$volume")
COMMON_PRIMARY_FEATURES = tuple(f"soxx_{name}" for name in COMMON_FEATURES)
SPECIFIC_FEATURES = (
    "soxl_gap1", "soxl_intraday1", "soxl_volume_z20", "soxl_range_pct",
    "relative_log_return_1d", "relative_log_return_5d", "relative_log_return_20d",
    "relative_volatility_20", "soxx_efficiency_20",
)
FORECAST_FEATURES = ("cash_advantage_prediction", "future_vol_prediction", "forecast_available")
PRIMARY_FEATURES = (*COMMON_PRIMARY_FEATURES, *SPECIFIC_FEATURES, *FORECAST_FEATURES)
RETURN_LABELS = ("soxl_return", "soxx_return")
FORECAST_LABELS = ("cash_advantage_1d", "future_vol_5d")
AUXILIARY_TARGETS = ("underlying_return_5d", "underlying_vol_5d")
PRIMARY_LABELS = (*RETURN_LABELS, *FORECAST_LABELS, *AUXILIARY_TARGETS)
LABEL_AVAILABILITY = {**{name: "return_available_at" for name in (*RETURN_LABELS, "cash_advantage_1d")},
                      "future_vol_5d": "forecast_available_at",
                      **{name: "auxiliary_available_at" for name in AUXILIARY_TARGETS}}
PRIMARY_METADATA = ("signal_date", "trade_date", "exit_date", "decision_at", "feature_available_at",
                    "return_available_at", "forecast_available_at", "auxiliary_available_at")
AUXILIARY_METADATA = ("signal_date", "feature_available_at", "auxiliary_available_at",
                      "membership_observed_at", "training_scope", "sample_weight")


@dataclass(frozen=True)
class IntegratedAllocationDataSpec:
    root: Path
    as_of: str = "2026-09-25"
    train_start: str = "2011-01-03"
    primary_provider: str = "data/qlib/soxl"
    execution_provider: str = "data/qlib/soxl_soxx_soxs"
    auxiliary_provider: str = "data/qlib/soxx_pit_leveraged_training_v1"


def frame_sha256(frame: pd.DataFrame) -> str:
    schema = json.dumps({"columns": [str(c) for c in frame.columns], "dtypes": [str(t) for t in frame.dtypes],
                         "index_names": list(frame.index.names)}, sort_keys=True).encode()
    return hashlib.sha256(schema + pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _manifest(provider: Path, *, as_of: str, require_end: bool) -> tuple[dict, str]:
    path = provider / "qlib_store_manifest.json"
    if not provider.is_dir() or not path.is_file():
        raise ValueError(f"blocked: audited Qlib provider is unavailable: {provider}")
    raw = path.read_bytes()
    value = json.loads(raw)
    for key in ("as_of", "calendar_start", "calendar_end", "instruments", "fields"):
        if key not in value:
            raise ValueError(f"blocked: Qlib provider manifest lacks {key}")
    if pd.Timestamp(value["as_of"]) > pd.Timestamp(as_of) or pd.Timestamp(value["calendar_end"]) > pd.Timestamp(as_of):
        raise ValueError("blocked: provider contains observations beyond the frozen as_of")
    if require_end and pd.Timestamp(value["calendar_end"]) != pd.Timestamp(as_of):
        raise ValueError("blocked: primary provider does not reach the frozen as_of")
    return value, hashlib.sha256(raw).hexdigest()


def _native_observations(provider: Path, *, fields: Sequence[str], instruments: Sequence[str], as_of: str,
                         require_end: bool = True) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    from qlib.data import D

    manifest, before = _manifest(provider, as_of=as_of, require_end=require_end)
    if not set(fields).issubset(manifest["fields"]) or not set(instruments).issubset(manifest["instruments"]):
        raise ValueError("blocked: requested fields/instruments absent from Qlib manifest")
    frame = load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider), instruments=list(instruments),
        fields=list(fields), start=manifest["calendar_start"], end=manifest["calendar_end"])
    calendar = pd.DatetimeIndex(D.calendar(start_time=manifest["calendar_start"], end_time=manifest["calendar_end"], freq="day"))
    if calendar.empty or calendar[-1] != pd.Timestamp(manifest["calendar_end"]):
        raise ValueError("blocked: Qlib calendar and manifest end disagree")
    if hashlib.sha256((provider / "qlib_store_manifest.json").read_bytes()).hexdigest() != before:
        raise ValueError("blocked: provider manifest changed during Qlib read")
    if frame.index.has_duplicates:
        raise ValueError("blocked: duplicate Qlib observations")
    return frame, calendar, {"provider_uri": str(provider), "manifest_sha256": before,
        "provider_as_of": manifest["as_of"], "calendar_start": str(calendar[0].date()),
        "calendar_end": str(calendar[-1].date()), "fields": list(fields), "instruments": list(instruments),
        "readback_sha256": frame_sha256(frame), "source_api": ["qlib.data.D.features", "qlib.data.D.calendar"]}


def _at_market_time(days, *, hour: int, minute: int = 0) -> pd.Series:
    source = days if isinstance(days, pd.Series) else pd.Series(days)
    stamps = pd.DatetimeIndex(pd.to_datetime(source)).tz_localize("America/New_York")
    return pd.Series(stamps + pd.Timedelta(hours=hour, minutes=minute), index=source.index)


def _future_targets(price: pd.Series) -> tuple[pd.Series, pd.Series]:
    one = price.pct_change(fill_method=None)
    forward = pd.concat([one.shift(-offset) for offset in range(1, 6)], axis=1)
    complete = forward.notna().all(axis=1)
    return (price.shift(-5) / price - 1.0).where(complete), forward.std(axis=1, ddof=1).where(complete)


def primary_frame_from_qlib(raw: pd.DataFrame, execution: pd.DataFrame, calendar: pd.DatetimeIndex,
                            *, train_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure transform of already Qlib-read frames; useful for causal tests."""
    parts = {}
    for symbol in ("SOXL", "SOXX"):
        part = raw.xs(symbol, level="instrument").reindex(calendar)
        prices = part.loc[:, list(PRICE_FIELDS)]
        if (not np.isfinite(prices.to_numpy(dtype=float)).all()
                or (prices.loc[:, ["$open", "$high", "$low", "$close"]] <= 0).any().any()
                or (prices["$volume"] < 0).any()):
            raise ValueError(f"blocked: incomplete primary OHLCV for {symbol}")
        common_source = part.reset_index().rename(columns={part.index.name or "index": "datetime"})
        common_source["instrument"] = symbol
        common_source["$adj_close"] = common_source["$close"]
        for field, value in {"$pit_weight_pct": 100., "$pit_available": 0., "$pit_member": 1.,
                             "$instrument_type": 0., "$leverage_multiple": 1., "$is_actual_2x": 0.}.items():
            common_source[field] = value
        computed = build_constituent_2x_tail_training_frame(common_source).set_index("datetime")
        parts[symbol] = (part, computed)
    frame = pd.DataFrame(index=calendar)
    frame.index.name = "datetime"
    for name in COMMON_FEATURES:
        frame[f"soxx_{name}"] = parts["SOXX"][1][name]
    for name in ("gap1", "intraday1", "volume_z20", "range_pct"):
        frame[f"soxl_{name}"] = parts["SOXL"][1][name]
    log_soxl, log_soxx = (np.log(parts[s][0]["$close"].astype(float)) for s in ("SOXL", "SOXX"))
    for horizon in (1, 5, 20):
        frame[f"relative_log_return_{horizon}d"] = log_soxl.diff(horizon) - 3.0 * log_soxx.diff(horizon)
    frame["relative_volatility_20"] = parts["SOXL"][1]["vol20"] / parts["SOXX"][1]["vol20"].replace(0., np.nan)
    frame["soxx_efficiency_20"] = log_soxx.diff(20).abs() / log_soxx.diff().abs().rolling(20, min_periods=20).sum().clip(lower=1e-12)
    # Neutral slots indicate absence of a trained OOF head, not predicted zero.
    frame.loc[:, list(FORECAST_FEATURES)] = 0.0
    prices = execution.loc[:, ["$open", "$close"]].unstack("instrument").reindex(calendar)
    prices.columns = pd.MultiIndex.from_tuples([(field.removeprefix("$"), symbol) for field, symbol in prices.columns])
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("blocked: incomplete native execution prices")
    dates = pd.Series(calendar, index=calendar)
    frame["signal_date"] = dates
    frame["trade_date"], frame["exit_date"] = dates.shift(-1), dates.shift(-2)
    frame["feature_available_at"] = _at_market_time(dates, hour=16)
    frame["decision_at"] = _at_market_time(frame.trade_date, hour=9, minute=31)
    frame["return_available_at"] = _at_market_time(frame.exit_date, hour=9, minute=30)
    frame["forecast_available_at"] = _at_market_time(dates.shift(-6), hour=9, minute=30)
    frame["auxiliary_available_at"] = _at_market_time(dates.shift(-5), hour=16)
    for symbol in ("SOXL", "SOXX"):
        opening = prices[("open", symbol)]
        frame[f"{symbol.lower()}_return"] = opening.shift(-2) / opening.shift(-1) - 1.0
    frame["cash_advantage_1d"] = -np.log1p(frame["soxx_return"])
    forward = pd.concat([frame.soxx_return.shift(-offset) for offset in range(5)], axis=1)
    frame["future_vol_5d"] = forward.std(axis=1, ddof=1).where(forward.notna().all(axis=1))
    frame["underlying_return_5d"], frame["underlying_vol_5d"] = _future_targets(parts["SOXX"][0]["$close"].astype(float))
    frame = frame.loc[pd.Timestamp(train_start):].copy()
    if frame.empty or not np.isfinite(frame.loc[:, list(PRIMARY_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("blocked: primary history lacks complete causal features after train_start")
    return frame, prices


def auxiliary_frame_from_qlib(raw: pd.DataFrame, calendar: pd.DatetimeIndex, *, train_start: str) -> pd.DataFrame:
    """Reuse common features, exclude union history, conservatively lag membership.

    Persisted PIT flags have filing dates but no intraday publication timestamp.
    Only the previous Qlib session's published membership is used at today's
    signal; this avoids claiming that a same-day SEC filing was known at close.
    """
    computed = build_constituent_2x_tail_training_frame(raw)
    parts = []
    previous_calendar = pd.Series(calendar, index=calendar).shift(1)
    for symbol, group in computed.groupby("instrument", sort=True):
        part = group.set_index("datetime").sort_index().reindex(calendar)
        available = part["$pit_available"].shift(1).gt(.5)
        member = part["$pit_member"].shift(1).gt(.5)
        eligible = available & member & part["$instrument_type"].eq(0.) & part["$is_actual_2x"].eq(0.)
        frame = part.loc[:, list(COMMON_FEATURES)].copy()
        frame["underlying_return_5d"], frame["underlying_vol_5d"] = _future_targets(part["$adj_close"].astype(float))
        dates = pd.Series(calendar, index=calendar)
        frame["signal_date"] = dates
        frame["feature_available_at"] = _at_market_time(dates, hour=16)
        frame["auxiliary_available_at"] = _at_market_time(dates.shift(-5), hour=16)
        frame["membership_observed_at"] = _at_market_time(previous_calendar, hour=16)
        frame["training_scope"] = "lagged_published_pit_member"
        finite = np.isfinite(frame.loc[:, list(COMMON_FEATURES)].to_numpy(dtype=float)).all(axis=1)
        frame = frame.loc[eligible & finite & dates.ge(pd.Timestamp(train_start))].copy()
        frame["instrument"] = symbol
        parts.append(frame)
    if not parts:
        raise ValueError("blocked: no PIT constituent observations")
    result = pd.concat(parts).reset_index(drop=True).set_index(["signal_date", "instrument"]).sort_index()
    result.index.names = ["datetime", "instrument"]
    result["signal_date"] = result.index.get_level_values("datetime")
    result["sample_weight"] = 1.0 / result.groupby(level="datetime")["training_scope"].transform("size")
    if result.empty:
        raise ValueError("blocked: no published PIT member rows with complete features")
    return result


def build_native_dataset(frame: pd.DataFrame, *, feature_columns: Sequence[str], label_columns: Sequence[str],
                         metadata_columns: Sequence[str], as_of: str):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = pd.concat({"feature": frame.loc[:, list(feature_columns)], "label": frame.loc[:, list(label_columns)],
                        "metadata": frame.loc[:, list(metadata_columns)]}, axis=1)
    if not isinstance(static.index, pd.MultiIndex):
        static.index = pd.MultiIndex.from_arrays([frame.index, np.full(len(frame), "SOXL_SOXX_CASH")], names=["datetime", "instrument"])
    dates = static.index.get_level_values("datetime")
    segments = {"all": (str(dates.min().date()), str(dates.max().date())),
                "development": ("2016-01-01", "2022-12-31"), "diagnostic": ("2023-01-01", as_of)}
    dataset = DatasetH(handler=DataHandlerLP(instruments=None, start_time=dates.min(), end_time=dates.max(),
        data_loader=StaticDataLoader(config=static), infer_processors=[], learn_processors=[]), segments=segments)
    if len(dataset.prepare("all")) != len(frame):
        raise ValueError("blocked: native Dataset changed the frozen row count")
    return dataset


def prepare_mature_rows(dataset, *, decision_at: str | pd.Timestamp, label_columns: Sequence[str],
                        availability_column: str, start: str = "2011-01-03", end: str | None = None):
    """Prepare Qlib rows whose selected labels were known strictly before fit.

    Passing January 1 New York of an annual fit year supplies the conservative
    annual information cutoff. No processor/scaler is fitted on future rows.
    Return (flat_frame, audit); empty pre-PIT auxiliary sets are legitimate.
    """
    cutoff = pd.Timestamp(decision_at)
    if cutoff.tzinfo is None:
        raise ValueError("blocked: training cutoff requires an explicit timezone")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    feature_columns = list(prepared["feature"].columns)
    result = prepared.copy()
    result.columns = [str(column[-1]) for column in result.columns]
    dates = result.index.get_level_values("datetime")
    temporal = pd.Series(dates >= pd.Timestamp(start), index=result.index)
    if end is not None:
        temporal &= dates <= pd.Timestamp(end)
    eligible = result.loc[temporal].copy()
    required_availability = {availability_column, *(LABEL_AVAILABILITY[name] for name in label_columns)}
    availability = pd.concat([pd.to_datetime(eligible[name], utc=True).rename(name)
                              for name in sorted(required_availability)], axis=1)
    available = availability.max(axis=1).where(availability.notna().all(axis=1))
    feature_known = pd.to_datetime(eligible["feature_available_at"], utc=True).lt(cutoff.tz_convert("UTC"))
    finite = np.isfinite(eligible.loc[:, [*feature_columns, *label_columns]].to_numpy(dtype=float)).all(axis=1)
    mature = available.notna() & available.lt(cutoff.tz_convert("UTC")) & feature_known
    rows = eligible.loc[mature & finite].copy()
    if "sample_weight" in rows:
        rows["sample_weight"] = 1.0 / rows.groupby(level="datetime")["sample_weight"].transform("size")
    return rows, {"decision_at": cutoff.isoformat(), "availability_column": availability_column,
        "rows_before_maturity": len(eligible), "rows": len(rows), "excluded_unmatured": int((~mature).sum()),
        "label_available_at_max": None if rows.empty else available.loc[rows.index].max().isoformat(),
        "required_availability_columns": sorted(required_availability),
        "first_signal_date": None if rows.empty else str(rows.index.get_level_values("datetime").min().date()),
        "last_signal_date": None if rows.empty else str(rows.index.get_level_values("datetime").max().date()),
        "purge_rule": "each selected label availability strictly before cutoff; no fixed extra gap",
        "date_weighting": "sum sample_weight per observed date equals one" if "sample_weight" in rows else "one primary row per session"}


def load_integrated_data(spec: IntegratedAllocationDataSpec) -> dict:
    root = Path(spec.root).resolve()
    primary, calendar, primary_audit = _native_observations(root / spec.primary_provider,
        fields=PRICE_FIELDS, instruments=("SOXL", "SOXX"), as_of=spec.as_of)
    execution, execution_calendar, execution_audit = _native_observations(root / spec.execution_provider,
        fields=("$open", "$close"), instruments=("SOXL", "SOXX"), as_of=spec.as_of)
    if not calendar.equals(execution_calendar):
        raise ValueError("blocked: primary and execution Qlib calendars differ")
    frame, prices = primary_frame_from_qlib(primary, execution, calendar, train_start=spec.train_start)
    auxiliary_path = root / spec.auxiliary_provider
    aux_manifest, _ = _manifest(auxiliary_path, as_of=spec.as_of, require_end=False)
    from src.factor.soxx_pit_tail_factors import PIT_TAIL_RAW_FIELDS
    aux_raw, aux_calendar, aux_audit = _native_observations(auxiliary_path,
        fields=PIT_TAIL_RAW_FIELDS, instruments=aux_manifest["instruments"], as_of=spec.as_of, require_end=False)
    aux_frame = auxiliary_frame_from_qlib(aux_raw.reset_index(), aux_calendar, train_start=spec.train_start)
    primary_dataset = build_native_dataset(frame, feature_columns=PRIMARY_FEATURES, label_columns=PRIMARY_LABELS,
        metadata_columns=PRIMARY_METADATA, as_of=spec.as_of)
    auxiliary_dataset = build_native_dataset(aux_frame, feature_columns=COMMON_FEATURES, label_columns=AUXILIARY_TARGETS,
        metadata_columns=AUXILIARY_METADATA, as_of=spec.as_of)
    audit = {"as_of": spec.as_of, "train_start": spec.train_start,
        "source_api": ["qlib.data.D.features", "qlib.data.D.calendar", "qlib.data.dataset.DatasetH",
                       "qlib.data.dataset.handler.DataHandlerLP", "qlib.data.dataset.loader.StaticDataLoader"],
        "primary": primary_audit, "execution": execution_audit, "auxiliary": aux_audit,
        "primary_frame_sha256": frame_sha256(frame), "execution_prices_sha256": frame_sha256(prices),
        "auxiliary_frame_sha256": frame_sha256(aux_frame), "primary_rows": len(frame), "auxiliary_rows": len(aux_frame),
        "primary_feature_end": str(frame.index.max().date()),
        "primary_return_label_end": str(frame.dropna(subset=list(RETURN_LABELS)).index.max().date()),
        "primary_forecast_label_end": str(frame.dropna(subset=list(FORECAST_LABELS)).index.max().date()),
        "auxiliary_first_signal": str(aux_frame.index.get_level_values("datetime").min().date()),
        "auxiliary_last_signal": str(aux_frame.index.get_level_values("datetime").max().date()),
        "common_feature_indices": list(range(len(COMMON_FEATURES))), "feature_columns": list(PRIMARY_FEATURES),
        "auxiliary_features": list(COMMON_FEATURES), "auxiliary_targets": list(AUXILIARY_TARGETS),
        "pit_membership": "previous Qlib session's filing-date-derived published membership; no same-day filing-time assumption",
        "pre_pit_union_backfill_used": False, "actual_2x_rows_used": False,
        "auxiliary_target_basis": "five future adjusted close returns; continuous return and sample standard deviation",
        "forecast_target_basis": "cash advantage is negative next-open log return; vol is sample std of five next-open returns",
        "unavailable_forecast_features": "explicit neutral zero plus forecast_available=0; not invented predictions",
        "label_timing": "signal close < next-open decision < label observation; incomplete tail labels remain missing",
        "qlib_docs_reference": "https://qlib.readthedocs.io/en/latest/component/data.html"}
    return {"primary_frame": frame, "auxiliary_frame": aux_frame, "execution_prices": prices,
            "primary_dataset": primary_dataset, "auxiliary_dataset": auxiliary_dataset,
            "primary_features": PRIMARY_FEATURES, "common_features": COMMON_FEATURES,
            "common_feature_indices": tuple(range(len(COMMON_FEATURES))), "auxiliary_targets": AUXILIARY_TARGETS,
            "return_labels": RETURN_LABELS, "forecast_labels": FORECAST_LABELS, "audit": audit}
