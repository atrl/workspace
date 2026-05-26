from __future__ import annotations

"""Dataset paths plus feature-engineering helpers."""

from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - validation reports missing optional deps.
    yaml = None

from ..data import (
    FEATURE_COLUMNS,
    STATIC_INDEX,
    build_dataset,
    engineer_feature_frame,
    feature_columns_for_packs,
    feature_schema_signature,
    flatten_feature_frame,
    init_qlib,
    refresh_qlib_binary,
    segment_frame,
    static_index_for_packs,
)
from ..settings import DATA_DIR, ETF_UNIVERSE

DATASET_REGISTRY_PATH = DATA_DIR / "datasets.yaml"


def slice_with_lookback(frame, start: str, end: str, *, lookback: int):
    """Return ``frame[start:end]`` plus up to ``lookback`` rows of prior context."""
    if int(lookback) <= 0:
        return frame.loc[start:end].copy()
    sorted_frame = frame.sort_index()
    window = sorted_frame.loc[:end]
    start_ts = next(iter(sorted_frame.loc[start:end].index), None)
    if start_ts is None:
        return sorted_frame.loc[start:end].copy()
    start_pos = window.index.get_loc(start_ts)
    if isinstance(start_pos, slice):
        start_pos = start_pos.start
    context_start = max(int(start_pos) - int(lookback), 0)
    return window.iloc[context_start:].copy()


def load_dataset_registry(path: Path | None = None) -> dict:
    registry_path = path or DATASET_REGISTRY_PATH
    if not registry_path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read data/datasets.yaml")
    with registry_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _asset_key(asset: str) -> str:
    return asset.lower()


def get_symbol_csv_path(asset: str) -> Path:
    return DATA_DIR / "raw" / "symbols" / f"{asset.upper()}.csv"


def get_market_panel_path(asset: str) -> Path:
    return DATA_DIR / "processed" / "panels" / f"{_asset_key(asset)}_with_underlying_vix.csv"


def get_qlib_dir(asset: str) -> Path:
    return DATA_DIR / "qlib" / _asset_key(asset)


def get_training_cache_path(profile: str) -> Path:
    return DATA_DIR / "cache" / "training" / f"{profile}.pkl"


def get_universe_assets(universe: str = "3x_etf_core") -> tuple[str, ...]:
    if universe != "3x_etf_core":
        raise ValueError(f"unknown ETF universe: {universe}")
    return tuple(ETF_UNIVERSE)

__all__ = [
    "FEATURE_COLUMNS",
    "STATIC_INDEX",
    "build_dataset",
    "engineer_feature_frame",
    "feature_columns_for_packs",
    "feature_schema_signature",
    "flatten_feature_frame",
    "init_qlib",
    "load_dataset_registry",
    "get_market_panel_path",
    "get_qlib_dir",
    "get_symbol_csv_path",
    "get_training_cache_path",
    "get_universe_assets",
    "refresh_qlib_binary",
    "segment_frame",
    "slice_with_lookback",
    "static_index_for_packs",
]
