from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core import stable_hash


class QlibUnavailableError(RuntimeError):
    """Raised when the qlib-native path cannot be initialized safely."""


@dataclass(frozen=True)
class QlibProviderConfig:
    provider_uri: Path
    region: str = "us"
    market: str = "all"
    freq: str = "day"
    kernels: int = 1

    @property
    def resolved_provider_uri(self) -> Path:
        return Path(self.provider_uri).resolve()


def _require_provider(config: QlibProviderConfig) -> Path:
    provider_uri = config.resolved_provider_uri
    required = [
        provider_uri,
        provider_uri / "calendars" / f"{config.freq}.txt",
        provider_uri / "instruments" / f"{config.market}.txt",
        provider_uri / "features",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise QlibUnavailableError(f"blocked: qlib unavailable: missing provider files {missing}")
    return provider_uri


def init_qlib_provider(config: QlibProviderConfig) -> dict[str, Any]:
    provider_uri = _require_provider(config)

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri.as_posix(), region=config.region, kernels=config.kernels)
    calendar = D.calendar(freq=config.freq)
    instruments = D.instruments(config.market)
    if len(calendar) == 0:
        raise QlibUnavailableError("blocked: qlib unavailable: empty qlib calendar")
    return {
        "provider_uri": provider_uri.as_posix(),
        "region": config.region,
        "market": config.market,
        "freq": config.freq,
        "kernels": int(config.kernels),
        "calendar_start": pd.Timestamp(calendar[0]).date().isoformat(),
        "calendar_end": pd.Timestamp(calendar[-1]).date().isoformat(),
        "instruments": instruments,
        "qlib_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.instruments"],
    }


def load_qlib_feature_frame(
    config: QlibProviderConfig,
    *,
    instruments: list[str],
    fields: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    init_qlib_provider(config)
    from qlib.data import D

    frame = D.features(
        instruments,
        fields,
        start_time=start,
        end_time=end,
        freq=config.freq,
    )
    frame.attrs["provider_uri"] = config.resolved_provider_uri.as_posix()
    frame.attrs["qlib_data_api"] = ["qlib.data.D.features"]
    frame.attrs["fields"] = fields
    return frame


def audit_qlib_provider(
    config: QlibProviderConfig,
    *,
    instruments: list[str],
    fields: list[str],
    start: str,
    end: str,
) -> dict[str, Any]:
    provider_meta = init_qlib_provider(config)
    frame = load_qlib_feature_frame(config, instruments=instruments, fields=fields, start=start, end=end)
    normalized = frame.reset_index()
    if "datetime" not in normalized.columns:
        raise QlibUnavailableError("blocked: qlib unavailable: qlib feature frame missing datetime index")
    normalized["datetime"] = pd.to_datetime(normalized["datetime"])

    missing_features: dict[str, list[str]] = {}
    for instrument in instruments:
        instrument_frame = normalized.loc[normalized["instrument"] == instrument]
        missing = [
            field
            for field in fields
            if field not in instrument_frame.columns or instrument_frame[field].dropna().empty
        ]
        if missing:
            missing_features[instrument] = missing
    max_datetime = normalized["datetime"].max().date().isoformat()
    snapshot_payload = {
        "provider_uri": provider_meta["provider_uri"],
        "market": config.market,
        "region": config.region,
        "instruments": instruments,
        "fields": fields,
        "start": start,
        "end": end,
        "max_datetime": max_datetime,
        "row_count": int(len(normalized)),
    }
    return {
        "status": "ok" if not missing_features else "missing_features",
        "provider_uri": provider_meta["provider_uri"],
        "market": config.market,
        "region": config.region,
        "calendar_start": provider_meta["calendar_start"],
        "calendar_end": provider_meta["calendar_end"],
        "instruments": instruments,
        "fields": fields,
        "feature_rows": int(len(normalized)),
        "max_datetime": max_datetime,
        "missing_features": missing_features,
        "snapshot_hash": stable_hash(snapshot_payload),
        "source_api": [
            "qlib.init",
            "qlib.data.D.calendar",
            "qlib.data.D.instruments",
            "qlib.data.D.features",
        ],
    }


def write_snapshot_manifest(audit: dict[str, Any], output_dir: Path, *, as_of: str) -> Path:
    if audit["max_datetime"] > as_of:
        raise QlibUnavailableError(
            f"blocked: qlib unavailable: snapshot max_datetime {audit['max_datetime']} exceeds as_of {as_of}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "qlib.data.D.features",
        "provider_uri": audit["provider_uri"],
        "market": audit["market"],
        "region": audit["region"],
        "as_of": as_of,
        "calendar_start": audit["calendar_start"],
        "calendar_end": audit["calendar_end"],
        "instruments": audit["instruments"],
        "fields": audit["fields"],
        "feature_rows": audit["feature_rows"],
        "max_datetime": audit["max_datetime"],
        "missing_features": audit["missing_features"],
        "snapshot_hash": audit["snapshot_hash"],
        "source_api": audit["source_api"],
    }
    path = output_dir / "snapshot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_data_update_manifest(audit: dict[str, Any], *, as_of: str) -> dict[str, Any]:
    if audit["max_datetime"] > as_of:
        raise QlibUnavailableError(
            f"blocked: qlib unavailable: update max_datetime {audit['max_datetime']} exceeds as_of {as_of}"
        )
    return {
        "operation": "qlib_data_update",
        "source": "qlib.data.D.features",
        "status": audit["status"],
        "provider_uri": audit["provider_uri"],
        "market": audit["market"],
        "region": audit["region"],
        "as_of": as_of,
        "calendar_start": audit["calendar_start"],
        "calendar_end": audit["calendar_end"],
        "instruments": audit["instruments"],
        "fields": audit["fields"],
        "feature_rows": audit["feature_rows"],
        "max_datetime": audit["max_datetime"],
        "missing_features": audit["missing_features"],
        "snapshot_hash": audit["snapshot_hash"],
        "source_api": audit["source_api"],
    }


def _normalize_qlib_field(field: str) -> str:
    return field.removeprefix("$").lower()


def _qlib_query_field(field: str) -> str:
    return f"${_normalize_qlib_field(field)}"


def _write_qlib_provider_metadata(
    provider_uri: Path,
    *,
    calendar: list[str],
    instruments: dict[str, tuple[str, str]],
    market: str,
    freq: str,
) -> None:
    (provider_uri / "calendars").mkdir(parents=True, exist_ok=True)
    (provider_uri / "instruments").mkdir(parents=True, exist_ok=True)
    (provider_uri / "features").mkdir(parents=True, exist_ok=True)
    (provider_uri / "calendars" / f"{freq}.txt").write_text("\n".join(calendar) + "\n", encoding="utf-8")
    instrument_lines = [
        f"{instrument}\t{date_range[0]}\t{date_range[1]}" for instrument, date_range in sorted(instruments.items())
    ]
    (provider_uri / "instruments" / f"{market}.txt").write_text(
        "\n".join(instrument_lines) + "\n",
        encoding="utf-8",
    )
    (provider_uri / "qlib.yaml").write_text(
        "# Qlib configuration\n"
        f"provider_uri: {provider_uri.as_posix()}\n"
        "redis:\n"
        "  host: null\n"
        "  port: null\n"
        "market:\n"
        f"  market: {market}\n"
        "  market_type: stock\n",
        encoding="utf-8",
    )


def _resolve_field_map(
    frame: pd.DataFrame,
    *,
    instrument_col: str,
    datetime_col: str,
    field_map: dict[str, str] | None,
) -> dict[str, str]:
    if field_map is not None:
        normalized = {_normalize_qlib_field(qlib_field): raw_field for qlib_field, raw_field in field_map.items()}
    else:
        ignored = {instrument_col, datetime_col}
        normalized = {
            _normalize_qlib_field(column): column
            for column in frame.columns
            if column not in ignored and pd.api.types.is_numeric_dtype(frame[column])
        }
    missing = [raw_field for raw_field in normalized.values() if raw_field not in frame.columns]
    if missing:
        raise QlibUnavailableError(f"blocked: qlib unavailable: raw fields missing for qlib store build {missing}")
    if not normalized:
        raise QlibUnavailableError("blocked: qlib unavailable: no numeric raw fields for qlib store build")
    return normalized


def build_qlib_store_from_frame(
    frame: pd.DataFrame,
    *,
    provider_uri: Path,
    market: str = "all",
    region: str = "us",
    freq: str = "day",
    instrument_col: str = "instrument",
    datetime_col: str = "datetime",
    field_map: dict[str, str] | None = None,
    as_of: str,
    source: str = "raw_frame",
) -> dict[str, Any]:
    """Build a qlib provider from raw tabular input, then audit it through qlib D APIs."""

    required_cols = {instrument_col, datetime_col}
    missing_required = sorted(required_cols - set(frame.columns))
    if missing_required:
        raise QlibUnavailableError(
            f"blocked: qlib unavailable: raw input missing qlib store columns {missing_required}"
        )
    fields = _resolve_field_map(
        frame,
        instrument_col=instrument_col,
        datetime_col=datetime_col,
        field_map=field_map,
    )
    normalized = frame.copy()
    normalized[instrument_col] = normalized[instrument_col].astype(str).str.upper()
    normalized[datetime_col] = pd.to_datetime(normalized[datetime_col]).dt.normalize()
    if normalized.duplicated([instrument_col, datetime_col]).any():
        raise QlibUnavailableError("blocked: qlib unavailable: duplicate instrument/datetime raw rows")
    max_datetime = normalized[datetime_col].max().date().isoformat()
    if max_datetime > as_of:
        raise QlibUnavailableError(
            f"blocked: qlib unavailable: raw max_datetime {max_datetime} exceeds as_of {as_of}"
        )

    provider_uri = Path(provider_uri).resolve()
    calendar = sorted(normalized[datetime_col].dt.date.astype(str).unique().tolist())
    instrument_ranges = {
        instrument: (
            instrument_frame[datetime_col].min().date().isoformat(),
            instrument_frame[datetime_col].max().date().isoformat(),
        )
        for instrument, instrument_frame in normalized.groupby(instrument_col)
    }
    _write_qlib_provider_metadata(
        provider_uri,
        calendar=calendar,
        instruments=instrument_ranges,
        market=market,
        freq=freq,
    )

    import qlib
    from qlib.data.storage.file_storage import FileFeatureStorage

    qlib.init(provider_uri=provider_uri.as_posix(), region=region)
    calendar_index = pd.Index(pd.to_datetime(calendar), name=datetime_col)
    feature_file_count = 0
    for instrument, instrument_frame in normalized.groupby(instrument_col):
        instrument_frame = instrument_frame.set_index(datetime_col).sort_index()
        for qlib_field, raw_field in fields.items():
            values = pd.to_numeric(instrument_frame[raw_field], errors="coerce")
            values = values.reindex(calendar_index).astype("float32").to_numpy()
            if np.isnan(values).all():
                raise QlibUnavailableError(
                    f"blocked: qlib unavailable: all-null raw field {raw_field} for {instrument}"
                )
            storage = FileFeatureStorage(instrument, qlib_field, freq)
            storage.uri.parent.mkdir(parents=True, exist_ok=True)
            if storage.uri.exists():
                storage.uri.unlink()
            storage.write(values, index=0)
            feature_file_count += 1

    qlib_fields = [_qlib_query_field(field) for field in fields]
    audit = audit_qlib_provider(
        QlibProviderConfig(provider_uri=provider_uri, region=region, market=market, freq=freq),
        instruments=sorted(instrument_ranges),
        fields=qlib_fields,
        start=calendar[0],
        end=calendar[-1],
    )
    payload = {
        "provider_uri": provider_uri.as_posix(),
        "market": market,
        "region": region,
        "freq": freq,
        "as_of": as_of,
        "source": source,
        "instruments": sorted(instrument_ranges),
        "fields": qlib_fields,
        "calendar_start": calendar[0],
        "calendar_end": calendar[-1],
        "row_count": int(len(normalized)),
        "feature_file_count": feature_file_count,
        "audit_snapshot_hash": audit["snapshot_hash"],
    }
    manifest = payload | {
        "operation": "qlib_store_build",
        "status": audit["status"],
        "hash": stable_hash(payload),
        "source_api": [
            "qlib.init",
            "qlib.data.storage.file_storage.FileFeatureStorage",
            "qlib.data.D.calendar",
            "qlib.data.D.instruments",
            "qlib.data.D.features",
        ],
    }
    manifest_path = provider_uri / "qlib_store_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_qlib_store_from_csv(
    raw_csv: Path,
    *,
    provider_uri: Path,
    market: str = "all",
    region: str = "us",
    freq: str = "day",
    instrument_col: str = "instrument",
    datetime_col: str = "datetime",
    field_map: dict[str, str] | None = None,
    as_of: str,
) -> dict[str, Any]:
    raw_csv = Path(raw_csv)
    if not raw_csv.exists():
        raise QlibUnavailableError(f"blocked: qlib unavailable: raw csv missing {raw_csv.as_posix()}")
    frame = pd.read_csv(raw_csv)
    return build_qlib_store_from_frame(
        frame,
        provider_uri=provider_uri,
        market=market,
        region=region,
        freq=freq,
        instrument_col=instrument_col,
        datetime_col=datetime_col,
        field_map=field_map,
        as_of=as_of,
        source=raw_csv.resolve().as_posix(),
    )


def build_qlib_dataset(
    config: QlibProviderConfig,
    *,
    instruments: list[str],
    features: list[str],
    label: str,
    train: tuple[str, str],
    valid: tuple[str, str] | None = None,
    test: tuple[str, str],
):
    init_qlib_provider(config)

    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    loader = QlibDataLoader(config={"feature": features, "label": [label]}, freq=config.freq)
    segments = {"train": train, "test": test}
    if valid is not None:
        segments["valid"] = valid
    segment_ranges = list(segments.values())
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=min(date_range[0] for date_range in segment_ranges),
        end_time=max(date_range[1] for date_range in segment_ranges),
        data_loader=loader,
    )
    dataset = DatasetH(
        handler=handler,
        segments=segments,
    )
    dataset.qlib_provider_uri = config.resolved_provider_uri.as_posix()
    dataset.qlib_dataset_api = ["DatasetH", "DataHandlerLP", "QlibDataLoader"]
    dataset.qlib_segments = segments
    return dataset


def alpha158_legacy_feature_config() -> tuple[list[str], list[str], dict[str, Any]]:
    """Return qlib Alpha158 expressions plus legacy SOXL rulebase features."""

    from qlib.contrib.data.handler import Alpha158

    alpha_fields, alpha_names = object.__new__(Alpha158).get_feature_config()
    gain_expr = "Greater($close-Ref($close, 1), 0)"
    loss_expr = "Greater(Ref($close, 1)-$close, 0)"
    rsi_expr = f"100-100/(1+Mean({gain_expr}, 6)/(Mean({loss_expr}, 6)+1e-5))"
    return_1d = "$close/Ref($close, 1)-1"
    legacy_fields = [
        "$vix",
        rsi_expr,
        f"Std({return_1d}, 20)*15.874507866387544",
        f"Std({return_1d}, 30)*15.874507866387544",
        f"Std({return_1d}, 20)",
        f"($vix-20)*10+({rsi_expr}-50)*0.4",
        "$close/Ref($close, 5)-1",
        "$volume/(Mean($volume, 5)+1e-12)",
    ]
    legacy_names = [
        "VIX",
        "RSI6",
        "HV_20",
        "HV_30",
        "VOL_COEF",
        "PANIC_SCORE",
        "SOXL_RET_5",
        "VOLUME_RATIO_5",
    ]
    metadata = {
        "source": "Alpha158.get_feature_config+legacy_soxl_features",
        "alpha158_count": len(alpha_names),
        "legacy_feature_names": legacy_names,
        "feature_count": len(alpha_names) + len(legacy_names),
    }
    return alpha_fields + legacy_fields, alpha_names + legacy_names, metadata


def build_alpha158_legacy_dataset(
    config: QlibProviderConfig,
    *,
    instruments: list[str],
    label: str,
    train: tuple[str, str],
    valid: tuple[str, str] | None = None,
    test: tuple[str, str],
):
    init_qlib_provider(config)

    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    feature_fields, feature_names, feature_set = alpha158_legacy_feature_config()
    loader = QlibDataLoader(
        config={
            "feature": (feature_fields, feature_names),
            "label": [label],
        },
        freq=config.freq,
    )
    segments = {"train": train, "test": test}
    if valid is not None:
        segments["valid"] = valid
    segment_ranges = list(segments.values())
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=min(date_range[0] for date_range in segment_ranges),
        end_time=max(date_range[1] for date_range in segment_ranges),
        data_loader=loader,
    )
    dataset = DatasetH(
        handler=handler,
        segments=segments,
    )
    dataset.qlib_provider_uri = config.resolved_provider_uri.as_posix()
    dataset.qlib_dataset_api = [
        "DatasetH",
        "DataHandlerLP",
        "QlibDataLoader",
        "Alpha158.get_feature_config",
    ]
    dataset.qlib_feature_set = feature_set
    dataset.qlib_segments = segments
    return dataset
