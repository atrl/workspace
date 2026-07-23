from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.factor.taleb_fragility import (
    FEATURE_DEFINITIONS,
    TALEB_FRAGILITY_FEATURES,
    build_taleb_fragility_feature_frame,
)

from .qlib_native import (
    QlibProviderConfig,
    build_qlib_store_from_frame,
    load_qlib_feature_frame,
)


@dataclass(frozen=True)
class TalebFragilityDataSpec:
    root: Path
    as_of: str = "2026-07-02"
    start: str = "2010-03-11"
    market: str = "all"
    region: str = "us"

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "taleb_fragility_v1"

    @property
    def raw_external_dir(self) -> Path:
        return self.root / "data" / "raw" / "external"


def _daily_price_frame(spec: TalebFragilityDataSpec) -> pd.DataFrame:
    fields = ["$open", "$high", "$low", "$close", "$volume", "$vix"]
    qlib_frame = load_qlib_feature_frame(
        QlibProviderConfig(
            provider_uri=spec.daily_provider_uri,
            market=spec.market,
            region=spec.region,
        ),
        instruments=["SOXL", "SOXX"],
        fields=fields,
        start=spec.start,
        end=spec.as_of,
    ).reset_index()
    qlib_frame["datetime"] = pd.to_datetime(
        qlib_frame["datetime"], errors="raise"
    ).dt.normalize()
    pieces: list[pd.DataFrame] = []
    for instrument in ("SOXL", "SOXX"):
        subset = qlib_frame.loc[
            qlib_frame["instrument"].astype(str).str.upper().eq(instrument),
            ["datetime", *fields],
        ].copy()
        subset = subset.rename(
            columns={
                field: f"{instrument.lower()}_{field.removeprefix('$')}"
                for field in fields
                if field != "$vix"
            }
        )
        if instrument == "SOXL":
            subset = subset.rename(columns={"$vix": "vix"})
        else:
            subset = subset.drop(columns="$vix")
        pieces.append(subset)
    return pieces[0].merge(
        pieces[1], on="datetime", how="inner", validate="one_to_one"
    )


def _read_external(
    spec: TalebFragilityDataSpec, name: str
) -> pd.DataFrame:
    path = spec.raw_external_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"blocked: missing Taleb external source {path.as_posix()}"
        )
    return pd.read_csv(path)


def build_taleb_fragility_qlib_store(
    *, root: Path | str, as_of: str = "2026-07-02"
) -> dict[str, Any]:
    spec = TalebFragilityDataSpec(
        root=Path(root).resolve(), as_of=as_of
    )
    daily = _daily_price_frame(spec)
    features = build_taleb_fragility_feature_frame(
        daily=daily,
        vix9d=_read_external(spec, "VIX9D"),
        vix3m=_read_external(spec, "VIX3M"),
        vvix=_read_external(spec, "VVIX"),
        skew=_read_external(spec, "SKEW"),
    )
    features = features.loc[
        features["datetime"].le(pd.Timestamp(spec.as_of))
    ].copy()
    features.insert(1, "instrument", "TALEB")
    manifest = build_qlib_store_from_frame(
        features,
        provider_uri=spec.provider_uri,
        market=spec.market,
        region=spec.region,
        instrument_col="instrument",
        datetime_col="datetime",
        field_map={feature: feature for feature in TALEB_FRAGILITY_FEATURES},
        as_of=spec.as_of,
        source=(
            "qlib:data/qlib/soxl+"
            "data/raw/external/VIX9D,VIX3M,VVIX,SKEW"
        ),
    )
    coverage = {}
    for name in ("VIX9D", "VIX3M", "VVIX", "SKEW"):
        frame = _read_external(spec, name)
        dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
        coverage[name] = {
            "start": dates.min().date().isoformat(),
            "end": dates.max().date().isoformat(),
            "rows": int(dates.notna().sum()),
        }
    manifest.update(
        {
            "feature_definitions": dict(FEATURE_DEFINITIONS),
            "point_in_time": True,
            "raw_source_coverage": coverage,
            "excluded_sources": {
                "intraday_external_context": (
                    "excluded_from_selection_history_starts_2024"
                ),
                "soxx_constituents": (
                    "excluded_non_PIT_current_constituents_backfill"
                ),
            },
        }
    )
    manifest_path = spec.provider_uri / "qlib_store_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
